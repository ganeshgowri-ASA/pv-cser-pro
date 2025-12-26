"""
Database management for PV-CSER Pro application.

Provides PostgreSQL database connectivity and ORM models
for storing module data, calculations, and results.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class PVModule(Base):
    """PV Module specifications table."""

    __tablename__ = "pv_modules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Module identification
    manufacturer = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    serial_number = Column(String(100))

    # Electrical specifications at STC
    pmax_stc = Column(Float, nullable=False)  # Wp
    voc_stc = Column(Float)  # V
    isc_stc = Column(Float)  # A
    vmp_stc = Column(Float)  # V
    imp_stc = Column(Float)  # A

    # Temperature coefficients
    temp_coeff_pmax = Column(Float)  # %/°C
    temp_coeff_voc = Column(Float)   # %/°C or mV/°C
    temp_coeff_isc = Column(Float)   # %/°C or mA/°C

    # Physical specifications
    module_area = Column(Float)      # m²
    cell_type = Column(String(50))   # mono-Si, poly-Si, CdTe, etc.
    num_cells = Column(Integer)

    # NMOT/NOCT
    nmot = Column(Float)  # °C

    # Additional data stored as JSON
    additional_data = Column(JSONB)

    # Relationships
    power_matrices = relationship("PowerMatrix", back_populates="module")
    calculations = relationship("CSERCalculation", back_populates="module")


class PowerMatrix(Base):
    """Power matrix data (IEC 61853-1)."""

    __tablename__ = "power_matrices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, ForeignKey("pv_modules.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Matrix data stored as JSON
    irradiance_levels = Column(JSONB)  # List of W/m² values
    temperature_levels = Column(JSONB)  # List of °C values
    power_values = Column(JSONB)  # 2D array of power values

    # Measurement conditions
    measurement_date = Column(DateTime)
    measurement_location = Column(String(255))
    notes = Column(Text)

    module = relationship("PVModule", back_populates="power_matrices")


class ClimateProfile(Base):
    """Climate profile data (IEC 61853-4)."""

    __tablename__ = "climate_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Profile identification
    profile_name = Column(String(255), nullable=False)
    profile_type = Column(String(50))  # standard, custom
    location = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)

    # Climate data as JSON (hourly or binned data)
    irradiance_data = Column(JSONB)
    temperature_data = Column(JSONB)
    spectral_data = Column(JSONB)
    wind_data = Column(JSONB)

    # Metadata
    is_standard = Column(Boolean, default=False)
    description = Column(Text)

    calculations = relationship("CSERCalculation", back_populates="climate_profile")


class CSERCalculation(Base):
    """CSER calculation results."""

    __tablename__ = "cser_calculations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, ForeignKey("pv_modules.id"), nullable=False)
    climate_profile_id = Column(Integer, ForeignKey("climate_profiles.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Results
    cser_value = Column(Float)  # kWh/kWp
    annual_energy_yield = Column(Float)  # kWh
    performance_ratio = Column(Float)  # %

    # Detailed results as JSON
    monthly_yields = Column(JSONB)
    hourly_yields = Column(JSONB)
    loss_breakdown = Column(JSONB)

    # Calculation parameters
    calculation_method = Column(String(100))
    parameters = Column(JSONB)

    module = relationship("PVModule", back_populates="calculations")
    climate_profile = relationship("ClimateProfile", back_populates="calculations")


class DatabaseManager:
    """
    Database manager for handling all database operations.

    Provides methods for CRUD operations on PV modules, power matrices,
    climate profiles, and CSER calculations.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            connection_string: PostgreSQL connection string.
                             If None, uses environment variable DATABASE_URL.
        """
        import os

        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL",
            "postgresql://localhost/pv_cser_pro"
        )

        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database connection and create tables."""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

            # Create tables
            Base.metadata.create_all(bind=self.engine)
            self._initialized = True
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._initialized = False
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session as context manager.

        Yields:
            SQLAlchemy Session instance
        """
        if not self._initialized:
            self.initialize()

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            if not self._initialized:
                self.initialize()

            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    # Module operations
    def save_module(self, module_data: Dict[str, Any]) -> int:
        """
        Save PV module to database.

        Args:
            module_data: Dictionary with module specifications

        Returns:
            ID of saved module
        """
        with self.get_session() as session:
            module = PVModule(**module_data)
            session.add(module)
            session.flush()
            return module.id

    def get_module(self, module_id: int) -> Optional[Dict[str, Any]]:
        """Get module by ID."""
        with self.get_session() as session:
            module = session.query(PVModule).filter_by(id=module_id).first()
            if module:
                return {
                    "id": module.id,
                    "manufacturer": module.manufacturer,
                    "model_name": module.model_name,
                    "pmax_stc": module.pmax_stc,
                    "voc_stc": module.voc_stc,
                    "isc_stc": module.isc_stc,
                    "vmp_stc": module.vmp_stc,
                    "imp_stc": module.imp_stc,
                    "temp_coeff_pmax": module.temp_coeff_pmax,
                    "temp_coeff_voc": module.temp_coeff_voc,
                    "temp_coeff_isc": module.temp_coeff_isc,
                    "module_area": module.module_area,
                    "cell_type": module.cell_type,
                    "nmot": module.nmot,
                    "additional_data": module.additional_data,
                }
            return None

    def list_modules(self) -> List[Dict[str, Any]]:
        """List all modules."""
        with self.get_session() as session:
            modules = session.query(PVModule).all()
            return [
                {
                    "id": m.id,
                    "manufacturer": m.manufacturer,
                    "model_name": m.model_name,
                    "pmax_stc": m.pmax_stc,
                }
                for m in modules
            ]

    # Power matrix operations
    def save_power_matrix(
        self,
        module_id: int,
        irradiance_levels: List[float],
        temperature_levels: List[float],
        power_values: List[List[float]],
        notes: Optional[str] = None,
    ) -> int:
        """Save power matrix data."""
        with self.get_session() as session:
            matrix = PowerMatrix(
                module_id=module_id,
                irradiance_levels=irradiance_levels,
                temperature_levels=temperature_levels,
                power_values=power_values,
                notes=notes,
            )
            session.add(matrix)
            session.flush()
            return matrix.id

    # Climate profile operations
    def save_climate_profile(self, profile_data: Dict[str, Any]) -> int:
        """Save climate profile."""
        with self.get_session() as session:
            profile = ClimateProfile(**profile_data)
            session.add(profile)
            session.flush()
            return profile.id

    def get_climate_profiles(self) -> List[Dict[str, Any]]:
        """Get all climate profiles."""
        with self.get_session() as session:
            profiles = session.query(ClimateProfile).all()
            return [
                {
                    "id": p.id,
                    "profile_name": p.profile_name,
                    "profile_type": p.profile_type,
                    "location": p.location,
                    "is_standard": p.is_standard,
                }
                for p in profiles
            ]

    # CSER calculation operations
    def save_calculation(
        self,
        module_id: int,
        climate_profile_id: int,
        results: Dict[str, Any],
    ) -> int:
        """Save CSER calculation results."""
        with self.get_session() as session:
            calc = CSERCalculation(
                module_id=module_id,
                climate_profile_id=climate_profile_id,
                cser_value=results.get("cser_value"),
                annual_energy_yield=results.get("annual_energy_yield"),
                performance_ratio=results.get("performance_ratio"),
                monthly_yields=results.get("monthly_yields"),
                loss_breakdown=results.get("loss_breakdown"),
                calculation_method=results.get("calculation_method"),
                parameters=results.get("parameters"),
            )
            session.add(calc)
            session.flush()
            return calc.id

    def get_calculations_for_module(self, module_id: int) -> List[Dict[str, Any]]:
        """Get all calculations for a module."""
        with self.get_session() as session:
            calcs = session.query(CSERCalculation).filter_by(
                module_id=module_id
            ).all()
            return [
                {
                    "id": c.id,
                    "climate_profile_id": c.climate_profile_id,
                    "cser_value": c.cser_value,
                    "annual_energy_yield": c.annual_energy_yield,
                    "performance_ratio": c.performance_ratio,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in calcs
            ]


def get_db_connection(connection_string: Optional[str] = None) -> DatabaseManager:
    """
    Get database connection manager instance.

    Args:
        connection_string: Optional PostgreSQL connection string

    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(connection_string)
