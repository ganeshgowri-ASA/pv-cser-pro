"""
Database Connection and Session Management for PV-CSER Pro.

Provides SQLAlchemy engine creation, session factory, and database
initialization utilities for Railway PostgreSQL integration.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration container."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        echo: bool = False,
    ):
        """
        Initialize database configuration.

        Args:
            database_url: PostgreSQL connection URL. Defaults to DATABASE_URL env var.
            pool_size: Number of connections to keep in the pool.
            max_overflow: Max connections above pool_size during high demand.
            pool_timeout: Seconds to wait for a connection from the pool.
            pool_recycle: Seconds after which to recycle connections.
            echo: Enable SQL query logging.
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL not provided. Set it via environment variable or constructor."
            )

        # Handle Railway's postgres:// vs postgresql:// URL format
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)

        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo


class Database:
    """Database connection manager with connection pooling."""

    _instance: Optional["Database"] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __new__(cls, config: Optional[DatabaseConfig] = None) -> "Database":
        """Singleton pattern for database instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        """
        Initialize database connection.

        Args:
            config: Database configuration. Uses defaults if not provided.
        """
        if self._initialized and config is None:
            return

        if config is None:
            config = DatabaseConfig()

        self._config = config
        self._create_engine()
        self._create_session_factory()
        self._initialized = True

    def _create_engine(self) -> None:
        """Create SQLAlchemy engine with connection pooling."""
        self._engine = create_engine(
            self._config.database_url,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            pool_timeout=self._config.pool_timeout,
            pool_recycle=self._config.pool_recycle,
            pool_pre_ping=True,  # Enable connection health checks
            echo=self._config.echo,
        )

        # Register event listeners
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set connection-level options if needed."""
            pass  # Placeholder for PostgreSQL-specific settings

        logger.info("Database engine created successfully")

    def _create_session_factory(self) -> None:
        """Create session factory."""
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized")
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get the session factory."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized")
        return self._session_factory

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db.session_scope() as session:
                session.add(obj)
                session.commit()

        Yields:
            Database session that will be committed on success,
            rolled back on exception.
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def init_db(self, drop_existing: bool = False) -> None:
        """
        Initialize database schema.

        Creates all tables defined in models.py.

        Args:
            drop_existing: If True, drops all existing tables first.
        """
        if drop_existing:
            logger.warning("Dropping all existing tables...")
            Base.metadata.drop_all(bind=self._engine)

        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self._engine)
        logger.info("Database initialization complete")

    def check_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def dispose(self) -> None:
        """Dispose of the engine and close all connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections disposed")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance:
            cls._instance.dispose()
            cls._instance = None
            cls._engine = None
            cls._session_factory = None


# Global database instance (lazy initialization)
_db: Optional[Database] = None


def get_database(config: Optional[DatabaseConfig] = None) -> Database:
    """
    Get or create the global database instance.

    Args:
        config: Optional database configuration.

    Returns:
        Database instance.
    """
    global _db
    if _db is None:
        _db = Database(config)
    return _db


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI/Streamlit to get a database session.

    Yields:
        Database session.

    Usage:
        # FastAPI
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

        # Streamlit
        db = next(get_db())
        try:
            items = db.query(Item).all()
        finally:
            db.close()
    """
    db = get_database()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


def init_db(drop_existing: bool = False) -> None:
    """
    Initialize the database schema.

    Creates all tables defined in models.

    Args:
        drop_existing: If True, drops all existing tables first.
    """
    db = get_database()
    db.init_db(drop_existing=drop_existing)


def check_connection() -> bool:
    """
    Check database connection health.

    Returns:
        True if connection successful.
    """
    db = get_database()
    return db.check_connection()


# Session context manager for simple usage
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_session() as session:
            modules = session.query(PVModule).all()

    Yields:
        Database session with automatic commit/rollback.
    """
    db = get_database()
    with db.session_scope() as session:
        yield session
