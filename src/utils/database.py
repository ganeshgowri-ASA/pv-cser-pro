"""
Database management for PV-CSER Pro application.

Provides PostgreSQL database connectivity and ORM models
for storing module data, calculations, and results.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

Base = declarative_base()


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CellType(str, Enum):
    """PV cell technology types."""
    MONO_SI = "mono-Si"
    POLY_SI = "poly-Si"
    PERC = "PERC"
    HJT = "HJT"
    TOPCON = "TOPCon"
    CDTE = "CdTe"
    CIGS = "CIGS"
    A_SI = "a-Si"
    OTHER = "other"


class CalculationStatus(str, Enum):
    """Status of CSER calculations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProfileType(str, Enum):
    """Climate profile types."""
    STANDARD = "standard"
    CUSTOM = "custom"
    IMPORTED = "imported"


class TemperatureModelType(str, Enum):
    """Temperature model types."""
    NOCT = "NOCT"
    NMOT = "NMOT"
    FAIMAN = "Faiman"
    PVSYST = "PVsyst"
    SANDIA = "Sandia"
    ROSS = "Ross"


# =============================================================================
# DATABASE MODELS
# =============================================================================

class PVModule(Base):
    """PV Module specifications table."""

    __tablename__ = "pv_modules"
    __table_args__ = (
        UniqueConstraint("manufacturer", "model_name", "serial_number", name="uq_module_identity"),
        Index("idx_module_manufacturer", "manufacturer"),
        Index("idx_module_model", "model_name"),
        Index("idx_module_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Module identification
    manufacturer = Column(String(255), nullable=False, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    serial_number = Column(String(100))

    # Electrical specifications at STC
    pmax_stc = Column(Float, nullable=False)  # Wp
    voc_stc = Column(Float)  # V
    isc_stc = Column(Float)  # A
    vmp_stc = Column(Float)  # V
    imp_stc = Column(Float)  # A
    fill_factor = Column(Float)  # Calculated: (Vmp * Imp) / (Voc * Isc)

    # Temperature coefficients (relative, %/°C)
    temp_coeff_pmax = Column(Float)  # %/°C (negative value)
    temp_coeff_voc = Column(Float)   # %/°C (negative value)
    temp_coeff_isc = Column(Float)   # %/°C (positive value, typically)

    # Temperature coefficients (absolute)
    temp_coeff_pmax_abs = Column(Float)  # W/°C
    temp_coeff_voc_abs = Column(Float)   # V/°C or mV/°C
    temp_coeff_isc_abs = Column(Float)   # A/°C or mA/°C

    # Physical specifications
    module_area = Column(Float)      # m²
    cell_area = Column(Float)        # m² (total active cell area)
    cell_type = Column(String(50))   # mono-Si, poly-Si, CdTe, etc.
    num_cells = Column(Integer)
    num_strings = Column(Integer)    # Number of cell strings in series

    # Dimensions
    length = Column(Float)  # mm
    width = Column(Float)   # mm
    thickness = Column(Float)  # mm
    weight = Column(Float)  # kg

    # NMOT/NOCT
    nmot = Column(Float)  # °C
    noct = Column(Float)  # °C (if different from NMOT)

    # Efficiency
    efficiency_stc = Column(Float)  # % (calculated: Pmax / (area * 1000) * 100)

    # Certification & Standards
    iec_certification = Column(String(255))  # IEC 61215, IEC 61730, etc.
    certification_date = Column(DateTime)
    bifacial = Column(Boolean, default=False)
    bifaciality_factor = Column(Float)  # For bifacial modules

    # Additional data stored as JSON
    additional_data = Column(JSONB)

    # Soft delete
    is_active = Column(Boolean, default=True)

    # Relationships
    power_matrices = relationship("PowerMatrix", back_populates="module", cascade="all, delete-orphan")
    spectral_responses = relationship("SpectralResponse", back_populates="module", cascade="all, delete-orphan")
    iam_data = relationship("IAMData", back_populates="module", cascade="all, delete-orphan")
    calculations = relationship("CSERCalculation", back_populates="module", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<PVModule(id={self.id}, manufacturer='{self.manufacturer}', model='{self.model_name}', Pmax={self.pmax_stc}W)>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert module to dictionary."""
        return {
            "id": self.id,
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "serial_number": self.serial_number,
            "pmax_stc": self.pmax_stc,
            "voc_stc": self.voc_stc,
            "isc_stc": self.isc_stc,
            "vmp_stc": self.vmp_stc,
            "imp_stc": self.imp_stc,
            "fill_factor": self.fill_factor,
            "temp_coeff_pmax": self.temp_coeff_pmax,
            "temp_coeff_voc": self.temp_coeff_voc,
            "temp_coeff_isc": self.temp_coeff_isc,
            "module_area": self.module_area,
            "cell_type": self.cell_type,
            "num_cells": self.num_cells,
            "nmot": self.nmot,
            "efficiency_stc": self.efficiency_stc,
            "bifacial": self.bifacial,
            "additional_data": self.additional_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PowerMatrix(Base):
    """Power matrix data (IEC 61853-1)."""

    __tablename__ = "power_matrices"
    __table_args__ = (
        Index("idx_power_matrix_module", "module_id"),
        Index("idx_power_matrix_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Matrix dimensions
    num_irradiance_levels = Column(Integer, nullable=False)
    num_temperature_levels = Column(Integer, nullable=False)

    # Matrix data stored as JSON
    irradiance_levels = Column(JSONB, nullable=False)  # List of W/m² values
    temperature_levels = Column(JSONB, nullable=False)  # List of °C values
    power_values = Column(JSONB, nullable=False)  # 2D array of power values (W)

    # Optional current and voltage matrices
    current_values = Column(JSONB)  # 2D array of Imp values (A)
    voltage_values = Column(JSONB)  # 2D array of Vmp values (V)
    voc_values = Column(JSONB)  # 2D array of Voc values (V)
    isc_values = Column(JSONB)  # 2D array of Isc values (A)

    # Normalized power matrix (P/Pmax at STC)
    normalized_power = Column(JSONB)

    # Measurement conditions
    measurement_date = Column(DateTime)
    measurement_location = Column(String(255))
    measurement_uncertainty = Column(Float)  # % uncertainty
    laboratory = Column(String(255))

    # Validation status
    is_validated = Column(Boolean, default=False)
    validation_notes = Column(Text)

    # Notes
    notes = Column(Text)

    module = relationship("PVModule", back_populates="power_matrices")

    def __repr__(self):
        return f"<PowerMatrix(id={self.id}, module_id={self.module_id}, {self.num_irradiance_levels}x{self.num_temperature_levels})>"


class SpectralResponse(Base):
    """Spectral response data (IEC 61853-2)."""

    __tablename__ = "spectral_responses"
    __table_args__ = (
        Index("idx_spectral_module", "module_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Spectral response data
    wavelengths = Column(JSONB, nullable=False)  # nm
    response_values = Column(JSONB, nullable=False)  # A/W or normalized
    is_normalized = Column(Boolean, default=False)

    # Reference spectrum
    reference_spectrum = Column(String(100))  # e.g., "AM1.5G"

    # Measurement metadata
    measurement_date = Column(DateTime)
    measurement_location = Column(String(255))
    notes = Column(Text)

    module = relationship("PVModule", back_populates="spectral_responses")


class IAMData(Base):
    """Incidence Angle Modifier data (IEC 61853-2)."""

    __tablename__ = "iam_data"
    __table_args__ = (
        Index("idx_iam_module", "module_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # IAM data
    angles = Column(JSONB, nullable=False)  # degrees
    iam_values = Column(JSONB, nullable=False)  # dimensionless

    # Model parameters (if fitted)
    model_type = Column(String(50))  # e.g., "ASHRAE", "physical", "Martin-Ruiz"
    model_parameters = Column(JSONB)

    # Measurement metadata
    measurement_date = Column(DateTime)
    notes = Column(Text)

    module = relationship("PVModule", back_populates="iam_data")


class ClimateProfile(Base):
    """Climate profile data (IEC 61853-4)."""

    __tablename__ = "climate_profiles"
    __table_args__ = (
        UniqueConstraint("profile_name", "location", name="uq_climate_profile"),
        Index("idx_climate_profile_type", "profile_type"),
        Index("idx_climate_location", "location"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Profile identification
    profile_name = Column(String(255), nullable=False)
    profile_code = Column(String(50))  # e.g., "Tropical", "Arid", "Temperate"
    profile_type = Column(String(50), default="standard")  # standard, custom, imported
    location = Column(String(255))
    country = Column(String(100))

    # Geographic coordinates
    latitude = Column(Float)
    longitude = Column(Float)
    elevation = Column(Float)  # meters
    timezone = Column(String(50))

    # Annual summary data
    annual_ghi = Column(Float)  # kWh/m²
    annual_dni = Column(Float)  # kWh/m²
    annual_dhi = Column(Float)  # kWh/m²
    avg_temperature = Column(Float)  # °C
    avg_wind_speed = Column(Float)  # m/s

    # Hourly climate data as JSON (8760 hours)
    ghi_data = Column(JSONB)  # W/m²
    dni_data = Column(JSONB)  # W/m²
    dhi_data = Column(JSONB)  # W/m²
    temperature_data = Column(JSONB)  # °C
    wind_speed_data = Column(JSONB)  # m/s
    relative_humidity_data = Column(JSONB)  # %

    # Spectral data (if available)
    spectral_data = Column(JSONB)
    average_am = Column(Float)  # Average air mass

    # Metadata
    is_standard = Column(Boolean, default=False)
    source = Column(String(255))  # e.g., "IEC 61853-4", "PVGIS", "TMY3"
    description = Column(Text)
    is_active = Column(Boolean, default=True)

    calculations = relationship("CSERCalculation", back_populates="climate_profile")

    def __repr__(self):
        return f"<ClimateProfile(id={self.id}, name='{self.profile_name}', location='{self.location}')>"


class CSERCalculation(Base):
    """CSER calculation results."""

    __tablename__ = "cser_calculations"
    __table_args__ = (
        Index("idx_cser_module", "module_id"),
        Index("idx_cser_climate", "climate_profile_id"),
        Index("idx_cser_status", "status"),
        Index("idx_cser_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False)
    climate_profile_id = Column(Integer, ForeignKey("climate_profiles.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)

    # Calculation status
    status = Column(String(50), default="completed")

    # Main results
    cser_value = Column(Float)  # kWh/kWp
    annual_energy_yield = Column(Float)  # kWh
    annual_dc_energy = Column(Float)  # kWh (before inverter)
    specific_yield = Column(Float)  # kWh/kWp
    performance_ratio = Column(Float)  # %
    capacity_factor = Column(Float)  # %

    # Temperature analysis
    avg_cell_temperature = Column(Float)  # °C
    max_cell_temperature = Column(Float)  # °C
    operating_hours = Column(Integer)  # hours with irradiance > 0

    # Detailed results as JSON
    monthly_yields = Column(JSONB)  # Monthly energy breakdown
    monthly_cser = Column(JSONB)  # Monthly CSER values
    hourly_yields = Column(JSONB)  # Optional: 8760 hourly values

    # Loss breakdown
    loss_breakdown = Column(JSONB)
    temperature_loss = Column(Float)  # %
    low_irradiance_loss = Column(Float)  # %
    spectral_loss = Column(Float)  # %
    iam_loss = Column(Float)  # %
    soiling_loss = Column(Float)  # %
    total_losses = Column(Float)  # %

    # Calculation parameters
    calculation_method = Column(String(100))
    temperature_model = Column(String(50))
    parameters = Column(JSONB)

    # Error information (if calculation failed)
    error_message = Column(Text)

    module = relationship("PVModule", back_populates="calculations")
    climate_profile = relationship("ClimateProfile", back_populates="calculations")

    def __repr__(self):
        return f"<CSERCalculation(id={self.id}, module_id={self.module_id}, cser={self.cser_value} kWh/kWp)>"


class CalculationLog(Base):
    """Log of calculation events for auditing."""

    __tablename__ = "calculation_logs"
    __table_args__ = (
        Index("idx_calc_log_calculation", "calculation_id"),
        Index("idx_calc_log_timestamp", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id = Column(Integer, ForeignKey("cser_calculations.id", ondelete="CASCADE"))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    event_type = Column(String(50), nullable=False)  # started, completed, failed, warning
    message = Column(Text)
    details = Column(JSONB)


class UserSession(Base):
    """User session tracking (for Streamlit sessions)."""

    __tablename__ = "user_sessions"
    __table_args__ = (
        Index("idx_session_id", "session_id"),
        Index("idx_session_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Session data
    user_agent = Column(String(500))
    ip_address = Column(String(50))

    # Current working data (stored in session)
    current_module_id = Column(Integer, ForeignKey("pv_modules.id", ondelete="SET NULL"))
    session_data = Column(JSONB)


class FileUpload(Base):
    """Track file uploads for validation and auditing."""

    __tablename__ = "file_uploads"
    __table_args__ = (
        Index("idx_upload_session", "session_id"),
        Index("idx_upload_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # File information
    original_filename = Column(String(500), nullable=False)
    file_type = Column(String(50))  # csv, xlsx, json
    file_size = Column(Integer)  # bytes
    file_hash = Column(String(64))  # SHA-256 hash

    # Upload purpose
    upload_type = Column(String(50))  # power_matrix, climate_data, module_specs

    # Validation results
    is_valid = Column(Boolean)
    validation_errors = Column(JSONB)
    validation_warnings = Column(JSONB)

    # Processing status
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)
    result_id = Column(Integer)  # ID of created record (module_id, matrix_id, etc.)


class ExportRecord(Base):
    """Track export operations."""

    __tablename__ = "export_records"
    __table_args__ = (
        Index("idx_export_session", "session_id"),
        Index("idx_export_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Export details
    export_type = Column(String(50), nullable=False)  # pdf, excel, json
    export_format = Column(String(50))  # full_report, summary, raw_data

    # Related data
    module_id = Column(Integer, ForeignKey("pv_modules.id", ondelete="SET NULL"))
    calculation_ids = Column(JSONB)  # List of included calculation IDs

    # File information
    filename = Column(String(500))
    file_size = Column(Integer)

    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)


# =============================================================================
# DATABASE MANAGER
# =============================================================================

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
                poolclass=QueuePool,
                pool_pre_ping=True,
                pool_size=int(os.getenv("DB_POOL_SIZE", 5)),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", 10)),
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", 30)),
                echo=os.getenv("DEBUG", "false").lower() == "true",
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

    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized

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

    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        stats = {}
        with self.get_session() as session:
            stats["pv_modules"] = session.query(PVModule).filter_by(is_active=True).count()
            stats["power_matrices"] = session.query(PowerMatrix).count()
            stats["climate_profiles"] = session.query(ClimateProfile).filter_by(is_active=True).count()
            stats["cser_calculations"] = session.query(CSERCalculation).count()
        return stats

    # =========================================================================
    # MODULE OPERATIONS
    # =========================================================================

    def create_module(self, module_data: Dict[str, Any]) -> int:
        """
        Create new PV module in database.

        Args:
            module_data: Dictionary with module specifications

        Returns:
            ID of created module
        """
        with self.get_session() as session:
            module = PVModule(**module_data)
            session.add(module)
            session.flush()
            logger.info(f"Created module: {module.manufacturer} {module.model_name} (ID: {module.id})")
            return module.id

    def get_module(self, module_id: int) -> Optional[Dict[str, Any]]:
        """Get module by ID."""
        with self.get_session() as session:
            module = session.query(PVModule).filter_by(id=module_id, is_active=True).first()
            if module:
                return module.to_dict()
            return None

    def update_module(self, module_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Update module data.

        Args:
            module_id: Module ID
            update_data: Dictionary with fields to update

        Returns:
            True if updated, False if not found
        """
        with self.get_session() as session:
            module = session.query(PVModule).filter_by(id=module_id, is_active=True).first()
            if not module:
                return False

            for key, value in update_data.items():
                if hasattr(module, key):
                    setattr(module, key, value)

            module.updated_at = datetime.utcnow()
            logger.info(f"Updated module ID: {module_id}")
            return True

    def delete_module(self, module_id: int, soft_delete: bool = True) -> bool:
        """
        Delete module.

        Args:
            module_id: Module ID
            soft_delete: If True, mark as inactive; if False, permanently delete

        Returns:
            True if deleted, False if not found
        """
        with self.get_session() as session:
            module = session.query(PVModule).filter_by(id=module_id).first()
            if not module:
                return False

            if soft_delete:
                module.is_active = False
                module.updated_at = datetime.utcnow()
            else:
                session.delete(module)

            logger.info(f"{'Soft-' if soft_delete else ''}Deleted module ID: {module_id}")
            return True

    def list_modules(
        self,
        manufacturer: Optional[str] = None,
        cell_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List modules with optional filtering.

        Args:
            manufacturer: Filter by manufacturer
            cell_type: Filter by cell type
            limit: Maximum number of results
            offset: Skip first N results

        Returns:
            List of module dictionaries
        """
        with self.get_session() as session:
            query = session.query(PVModule).filter_by(is_active=True)

            if manufacturer:
                query = query.filter(PVModule.manufacturer.ilike(f"%{manufacturer}%"))
            if cell_type:
                query = query.filter(PVModule.cell_type == cell_type)

            modules = query.order_by(PVModule.created_at.desc()).limit(limit).offset(offset).all()
            return [m.to_dict() for m in modules]

    def search_modules(self, search_term: str) -> List[Dict[str, Any]]:
        """Search modules by manufacturer, model name, or serial number."""
        with self.get_session() as session:
            modules = session.query(PVModule).filter(
                PVModule.is_active == True,
                (
                    PVModule.manufacturer.ilike(f"%{search_term}%") |
                    PVModule.model_name.ilike(f"%{search_term}%") |
                    PVModule.serial_number.ilike(f"%{search_term}%")
                )
            ).all()
            return [m.to_dict() for m in modules]

    # =========================================================================
    # POWER MATRIX OPERATIONS
    # =========================================================================

    def create_power_matrix(
        self,
        module_id: int,
        irradiance_levels: List[float],
        temperature_levels: List[float],
        power_values: List[List[float]],
        **kwargs,
    ) -> int:
        """Create power matrix for a module."""
        with self.get_session() as session:
            matrix = PowerMatrix(
                module_id=module_id,
                irradiance_levels=irradiance_levels,
                temperature_levels=temperature_levels,
                power_values=power_values,
                num_irradiance_levels=len(irradiance_levels),
                num_temperature_levels=len(temperature_levels),
                **kwargs,
            )
            session.add(matrix)
            session.flush()
            logger.info(f"Created power matrix for module {module_id} (ID: {matrix.id})")
            return matrix.id

    def get_power_matrix(self, matrix_id: int) -> Optional[Dict[str, Any]]:
        """Get power matrix by ID."""
        with self.get_session() as session:
            matrix = session.query(PowerMatrix).filter_by(id=matrix_id).first()
            if matrix:
                return {
                    "id": matrix.id,
                    "module_id": matrix.module_id,
                    "irradiance_levels": matrix.irradiance_levels,
                    "temperature_levels": matrix.temperature_levels,
                    "power_values": matrix.power_values,
                    "current_values": matrix.current_values,
                    "voltage_values": matrix.voltage_values,
                    "normalized_power": matrix.normalized_power,
                    "is_validated": matrix.is_validated,
                    "notes": matrix.notes,
                }
            return None

    def get_power_matrices_for_module(self, module_id: int) -> List[Dict[str, Any]]:
        """Get all power matrices for a module."""
        with self.get_session() as session:
            matrices = session.query(PowerMatrix).filter_by(module_id=module_id).all()
            return [
                {
                    "id": m.id,
                    "irradiance_levels": m.irradiance_levels,
                    "temperature_levels": m.temperature_levels,
                    "power_values": m.power_values,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in matrices
            ]

    # =========================================================================
    # CLIMATE PROFILE OPERATIONS
    # =========================================================================

    def create_climate_profile(self, profile_data: Dict[str, Any]) -> int:
        """Create climate profile."""
        with self.get_session() as session:
            profile = ClimateProfile(**profile_data)
            session.add(profile)
            session.flush()
            logger.info(f"Created climate profile: {profile.profile_name} (ID: {profile.id})")
            return profile.id

    def get_climate_profile(self, profile_id: int) -> Optional[Dict[str, Any]]:
        """Get climate profile by ID."""
        with self.get_session() as session:
            profile = session.query(ClimateProfile).filter_by(id=profile_id, is_active=True).first()
            if profile:
                return {
                    "id": profile.id,
                    "profile_name": profile.profile_name,
                    "profile_type": profile.profile_type,
                    "location": profile.location,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                    "annual_ghi": profile.annual_ghi,
                    "avg_temperature": profile.avg_temperature,
                    "ghi_data": profile.ghi_data,
                    "temperature_data": profile.temperature_data,
                    "is_standard": profile.is_standard,
                }
            return None

    def list_climate_profiles(self, profile_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List climate profiles with optional type filter."""
        with self.get_session() as session:
            query = session.query(ClimateProfile).filter_by(is_active=True)
            if profile_type:
                query = query.filter_by(profile_type=profile_type)

            profiles = query.order_by(ClimateProfile.profile_name).all()
            return [
                {
                    "id": p.id,
                    "profile_name": p.profile_name,
                    "profile_type": p.profile_type,
                    "location": p.location,
                    "annual_ghi": p.annual_ghi,
                    "is_standard": p.is_standard,
                }
                for p in profiles
            ]

    def get_standard_profiles(self) -> List[Dict[str, Any]]:
        """Get all standard IEC 61853-4 climate profiles."""
        return self.list_climate_profiles(profile_type="standard")

    # =========================================================================
    # CSER CALCULATION OPERATIONS
    # =========================================================================

    def create_calculation(
        self,
        module_id: int,
        climate_profile_id: int,
        results: Dict[str, Any],
    ) -> int:
        """Create CSER calculation record."""
        with self.get_session() as session:
            calc = CSERCalculation(
                module_id=module_id,
                climate_profile_id=climate_profile_id,
                cser_value=results.get("cser_value"),
                annual_energy_yield=results.get("annual_energy_yield"),
                annual_dc_energy=results.get("annual_dc_energy"),
                specific_yield=results.get("specific_yield"),
                performance_ratio=results.get("performance_ratio"),
                capacity_factor=results.get("capacity_factor"),
                avg_cell_temperature=results.get("avg_cell_temperature"),
                max_cell_temperature=results.get("max_cell_temperature"),
                operating_hours=results.get("operating_hours"),
                monthly_yields=results.get("monthly_yields"),
                monthly_cser=results.get("monthly_cser"),
                loss_breakdown=results.get("loss_breakdown"),
                temperature_loss=results.get("temperature_loss"),
                low_irradiance_loss=results.get("low_irradiance_loss"),
                spectral_loss=results.get("spectral_loss"),
                iam_loss=results.get("iam_loss"),
                soiling_loss=results.get("soiling_loss"),
                total_losses=results.get("total_losses"),
                calculation_method=results.get("calculation_method"),
                temperature_model=results.get("temperature_model"),
                parameters=results.get("parameters"),
                status="completed",
                completed_at=datetime.utcnow(),
            )
            session.add(calc)
            session.flush()
            logger.info(f"Created CSER calculation for module {module_id} (ID: {calc.id}, CSER: {calc.cser_value})")
            return calc.id

    def get_calculation(self, calculation_id: int) -> Optional[Dict[str, Any]]:
        """Get calculation by ID."""
        with self.get_session() as session:
            calc = session.query(CSERCalculation).filter_by(id=calculation_id).first()
            if calc:
                return {
                    "id": calc.id,
                    "module_id": calc.module_id,
                    "climate_profile_id": calc.climate_profile_id,
                    "cser_value": calc.cser_value,
                    "annual_energy_yield": calc.annual_energy_yield,
                    "performance_ratio": calc.performance_ratio,
                    "monthly_yields": calc.monthly_yields,
                    "loss_breakdown": calc.loss_breakdown,
                    "status": calc.status,
                    "created_at": calc.created_at.isoformat() if calc.created_at else None,
                }
            return None

    def get_calculations_for_module(self, module_id: int) -> List[Dict[str, Any]]:
        """Get all calculations for a module."""
        with self.get_session() as session:
            calcs = session.query(CSERCalculation).filter_by(module_id=module_id).all()
            return [
                {
                    "id": c.id,
                    "climate_profile_id": c.climate_profile_id,
                    "cser_value": c.cser_value,
                    "annual_energy_yield": c.annual_energy_yield,
                    "performance_ratio": c.performance_ratio,
                    "status": c.status,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in calcs
            ]

    def get_calculations_for_climate(self, climate_profile_id: int) -> List[Dict[str, Any]]:
        """Get all calculations for a climate profile."""
        with self.get_session() as session:
            calcs = session.query(CSERCalculation).filter_by(
                climate_profile_id=climate_profile_id
            ).all()
            return [
                {
                    "id": c.id,
                    "module_id": c.module_id,
                    "cser_value": c.cser_value,
                    "performance_ratio": c.performance_ratio,
                }
                for c in calcs
            ]

    def compare_modules_for_climate(
        self,
        module_ids: List[int],
        climate_profile_id: int,
    ) -> List[Dict[str, Any]]:
        """Compare multiple modules for a specific climate."""
        results = []
        with self.get_session() as session:
            for module_id in module_ids:
                calc = session.query(CSERCalculation).filter_by(
                    module_id=module_id,
                    climate_profile_id=climate_profile_id,
                ).order_by(CSERCalculation.created_at.desc()).first()

                if calc:
                    module = session.query(PVModule).filter_by(id=module_id).first()
                    results.append({
                        "module_id": module_id,
                        "manufacturer": module.manufacturer if module else None,
                        "model_name": module.model_name if module else None,
                        "pmax_stc": module.pmax_stc if module else None,
                        "cser_value": calc.cser_value,
                        "performance_ratio": calc.performance_ratio,
                    })
        return sorted(results, key=lambda x: x.get("cser_value", 0), reverse=True)

    # =========================================================================
    # FILE UPLOAD TRACKING
    # =========================================================================

    def log_file_upload(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        upload_type: str,
        is_valid: bool,
        validation_errors: Optional[List[str]] = None,
        validation_warnings: Optional[List[str]] = None,
    ) -> int:
        """Log file upload for auditing."""
        with self.get_session() as session:
            upload = FileUpload(
                session_id=session_id,
                original_filename=filename,
                file_type=file_type,
                file_size=file_size,
                upload_type=upload_type,
                is_valid=is_valid,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
            )
            session.add(upload)
            session.flush()
            return upload.id

    # =========================================================================
    # EXPORT TRACKING
    # =========================================================================

    def log_export(
        self,
        session_id: str,
        export_type: str,
        filename: str,
        module_id: Optional[int] = None,
        calculation_ids: Optional[List[int]] = None,
        file_size: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> int:
        """Log export operation."""
        with self.get_session() as session:
            record = ExportRecord(
                session_id=session_id,
                export_type=export_type,
                filename=filename,
                module_id=module_id,
                calculation_ids=calculation_ids,
                file_size=file_size,
                success=success,
                error_message=error_message,
            )
            session.add(record)
            session.flush()
            return record.id

    # =========================================================================
    # CALCULATION LOGGING
    # =========================================================================

    def log_calculation_event(
        self,
        calculation_id: int,
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log calculation event for auditing."""
        with self.get_session() as session:
            log = CalculationLog(
                calculation_id=calculation_id,
                event_type=event_type,
                message=message,
                details=details,
            )
            session.add(log)

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def create_or_update_session(
        self,
        session_id: str,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        session_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update user session."""
        with self.get_session() as session:
            user_session = session.query(UserSession).filter_by(session_id=session_id).first()
            if user_session:
                user_session.last_activity = datetime.utcnow()
                if session_data:
                    user_session.session_data = session_data
            else:
                user_session = UserSession(
                    session_id=session_id,
                    user_agent=user_agent,
                    ip_address=ip_address,
                    session_data=session_data,
                )
                session.add(user_session)

    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """Remove sessions older than specified hours."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self.get_session() as session:
            count = session.query(UserSession).filter(
                UserSession.last_activity < cutoff
            ).delete()
            logger.info(f"Cleaned up {count} old sessions")
            return count


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_db_connection(connection_string: Optional[str] = None) -> DatabaseManager:
    """
    Get database connection manager instance.

    Args:
        connection_string: Optional PostgreSQL connection string

    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(connection_string)


def init_database(connection_string: Optional[str] = None) -> DatabaseManager:
    """
    Initialize database with all tables.

    Args:
        connection_string: Optional PostgreSQL connection string

    Returns:
        Initialized DatabaseManager instance
    """
    db = DatabaseManager(connection_string)
    db.initialize()
    return db


# Event listener to calculate fill factor and efficiency before insert
@event.listens_for(PVModule, "before_insert")
def calculate_derived_fields(mapper, connection, target):
    """Calculate derived fields before inserting module."""
    if target.voc_stc and target.isc_stc and target.vmp_stc and target.imp_stc:
        voc_isc = target.voc_stc * target.isc_stc
        if voc_isc > 0:
            target.fill_factor = (target.vmp_stc * target.imp_stc) / voc_isc

    if target.pmax_stc and target.module_area:
        target.efficiency_stc = (target.pmax_stc / (target.module_area * 1000)) * 100
