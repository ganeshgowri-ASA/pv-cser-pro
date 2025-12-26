"""
SQLAlchemy ORM Models for PV-CSER Pro.

Defines all database models matching the Railway PostgreSQL schema
for IEC 61853 compliant PV module energy rating calculations.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class User(Base):
    """User account model for authentication and data ownership."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    organization: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    pv_modules: Mapped[List["PVModule"]] = relationship("PVModule", back_populates="owner")
    cser_calculations: Mapped[List["CSERCalculation"]] = relationship(
        "CSERCalculation", back_populates="user"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


class PVModule(Base):
    """
    PV Module specifications model.

    Stores module parameters per IEC 61853-1:
    - Nameplate data (Pmax, Voc, Isc at STC)
    - Temperature coefficients
    - Physical dimensions
    - NMOT parameters
    """

    __tablename__ = "pv_modules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Module identification
    module_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    manufacturer: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_number: Mapped[Optional[str]] = mapped_column(String(100))
    technology: Mapped[Optional[str]] = mapped_column(String(50))  # mono-Si, poly-Si, CdTe, etc.

    # STC ratings (1000 W/m2, 25C, AM1.5G)
    pmax_stc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # Watts
    voc_stc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)   # Volts
    isc_stc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)   # Amps
    vmpp_stc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # Volts
    impp_stc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # Amps
    fill_factor_stc: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))  # Ratio

    # Temperature coefficients
    alpha_isc: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)  # %/C or A/C
    beta_voc: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)   # %/C or V/C
    gamma_pmax: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False) # %/C or W/C
    temp_coeff_unit: Mapped[str] = mapped_column(
        String(10), default="percent", nullable=False
    )  # 'percent' or 'absolute'

    # Physical parameters
    cell_count: Mapped[Optional[int]] = mapped_column(Integer)
    cell_type: Mapped[Optional[str]] = mapped_column(String(50))
    module_area: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))  # m2
    aperture_area: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))  # m2

    # NMOT parameters (IEC 61853-2)
    nmot: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # Nominal Module Operating Temp
    nmot_irradiance: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2), default=800)  # W/m2
    nmot_ambient: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), default=20)  # C
    nmot_wind_speed: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), default=1)  # m/s

    # Additional thermal parameters
    u_const: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # W/m2K (const heat transfer)
    u_wind: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))   # W/m2K/(m/s) (wind-dependent)

    # Metadata
    owner_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    owner: Mapped[Optional["User"]] = relationship("User", back_populates="pv_modules")
    power_matrix: Mapped[List["PowerMatrix"]] = relationship(
        "PowerMatrix", back_populates="pv_module", cascade="all, delete-orphan"
    )
    spectral_response: Mapped[List["SpectralResponse"]] = relationship(
        "SpectralResponse", back_populates="pv_module", cascade="all, delete-orphan"
    )
    iam_data: Mapped[List["IAMData"]] = relationship(
        "IAMData", back_populates="pv_module", cascade="all, delete-orphan"
    )
    cser_calculations: Mapped[List["CSERCalculation"]] = relationship(
        "CSERCalculation", back_populates="pv_module"
    )

    __table_args__ = (
        Index("ix_pv_modules_manufacturer_name", "manufacturer", "module_name"),
        CheckConstraint("pmax_stc > 0", name="ck_pmax_positive"),
        CheckConstraint("voc_stc > 0", name="ck_voc_positive"),
        CheckConstraint("isc_stc > 0", name="ck_isc_positive"),
    )

    def __repr__(self) -> str:
        return f"<PVModule(id={self.id}, name='{self.module_name}', Pmax={self.pmax_stc}W)>"


class PowerMatrix(Base):
    """
    Power matrix data per IEC 61853-1.

    Stores measured IV parameters at various irradiance and temperature
    combinations (typically 6 irradiances x 4 temperatures = 24 points).
    """

    __tablename__ = "power_matrix"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    module_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Operating conditions
    irradiance: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)  # W/m2
    temperature: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)  # Cell temp C

    # Measured parameters
    isc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)   # Short-circuit current A
    voc: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)   # Open-circuit voltage V
    pmax: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # Max power W
    vmpp: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # MPP voltage V
    impp: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # MPP current A
    fill_factor: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))  # FF = Pmax/(Voc*Isc)

    # Quality indicators
    measurement_uncertainty: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))  # %
    measurement_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    pv_module: Mapped["PVModule"] = relationship("PVModule", back_populates="power_matrix")

    __table_args__ = (
        UniqueConstraint("module_id", "irradiance", "temperature", name="uq_power_matrix_point"),
        Index("ix_power_matrix_conditions", "module_id", "irradiance", "temperature"),
        CheckConstraint("irradiance >= 0", name="ck_pm_irradiance_positive"),
        CheckConstraint("pmax >= 0", name="ck_pm_pmax_positive"),
    )

    def __repr__(self) -> str:
        return f"<PowerMatrix(module_id={self.module_id}, G={self.irradiance}, T={self.temperature}, Pmax={self.pmax})>"


class SpectralResponse(Base):
    """
    Spectral response data per IEC 61853-2.

    Stores normalized spectral response (quantum efficiency or
    spectral responsivity) at discrete wavelengths.
    """

    __tablename__ = "spectral_response"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    module_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Wavelength and response
    wavelength: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)  # nm
    response: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)   # Normalized (0-1) or A/W
    response_unit: Mapped[str] = mapped_column(
        String(20), default="normalized", nullable=False
    )  # 'normalized', 'A/W', 'EQE'

    # Measurement metadata
    measurement_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    measurement_uncertainty: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    # Relationships
    pv_module: Mapped["PVModule"] = relationship("PVModule", back_populates="spectral_response")

    __table_args__ = (
        UniqueConstraint("module_id", "wavelength", name="uq_spectral_wavelength"),
        Index("ix_spectral_module_wavelength", "module_id", "wavelength"),
        CheckConstraint("wavelength > 0", name="ck_sr_wavelength_positive"),
    )

    def __repr__(self) -> str:
        return f"<SpectralResponse(module_id={self.module_id}, wavelength={self.wavelength}nm)>"


class IAMData(Base):
    """
    Incidence Angle Modifier (IAM) data per IEC 61853-2.

    Stores angular loss factors for different angles of incidence.
    """

    __tablename__ = "iam_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    module_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Angle and IAM value
    angle: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)  # Degrees (0-90)
    iam_value: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)  # Modifier (0-1)

    # For bi-directional IAM (azimuth-dependent)
    azimuth: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # Degrees (0-360)

    # Measurement metadata
    measurement_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    measurement_uncertainty: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    # Relationships
    pv_module: Mapped["PVModule"] = relationship("PVModule", back_populates="iam_data")

    __table_args__ = (
        UniqueConstraint("module_id", "angle", "azimuth", name="uq_iam_angle"),
        Index("ix_iam_module_angle", "module_id", "angle"),
        CheckConstraint("angle >= 0 AND angle <= 90", name="ck_iam_angle_range"),
        CheckConstraint("iam_value >= 0 AND iam_value <= 1", name="ck_iam_value_range"),
    )

    def __repr__(self) -> str:
        return f"<IAMData(module_id={self.module_id}, angle={self.angle}, IAM={self.iam_value})>"


class ClimateProfile(Base):
    """
    Climate profile metadata per IEC 61853-4.

    Defines standard and custom climate profiles used for CSER calculations.
    Standard profiles: Subtropical Arid, Subtropical Coastal, Temperate Coastal,
    High Elevation, Temperate Continental, Tropical Humid.
    """

    __tablename__ = "climate_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Profile identification
    profile_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    profile_code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)  # e.g., 'SA', 'SC', 'TC'
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Climate characteristics
    climate_type: Mapped[str] = mapped_column(String(50), nullable=False)  # Koeppen classification
    is_standard: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)  # IEC standard or custom

    # Location reference (for custom profiles)
    latitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(9, 6))
    longitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(9, 6))
    elevation: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))  # meters
    timezone: Mapped[Optional[str]] = mapped_column(String(50))

    # Annual summary statistics
    annual_ghi: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))  # kWh/m2/year
    annual_dni: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))  # kWh/m2/year
    annual_dhi: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))  # kWh/m2/year
    avg_temperature: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # C
    avg_wind_speed: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # m/s

    # Metadata
    source: Mapped[Optional[str]] = mapped_column(String(255))  # Data source
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    hourly_data: Mapped[List["HourlyClimateData"]] = relationship(
        "HourlyClimateData", back_populates="climate_profile", cascade="all, delete-orphan"
    )
    cser_calculations: Mapped[List["CSERCalculation"]] = relationship(
        "CSERCalculation", back_populates="climate_profile"
    )

    def __repr__(self) -> str:
        return f"<ClimateProfile(id={self.id}, name='{self.profile_name}', code='{self.profile_code}')>"


class HourlyClimateData(Base):
    """
    Hourly climate data for CSER calculations per IEC 61853-4.

    Stores 8760 hours (or representative hours) of climate data
    including irradiance, temperature, wind speed, etc.
    """

    __tablename__ = "hourly_climate_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    climate_profile_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("climate_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Time identification
    hour_of_year: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-8760
    month: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-12
    day: Mapped[int] = mapped_column(Integer, nullable=False)    # 1-31
    hour: Mapped[int] = mapped_column(Integer, nullable=False)   # 0-23

    # Irradiance components (W/m2)
    ghi: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)  # Global Horizontal
    dni: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))         # Direct Normal
    dhi: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))         # Diffuse Horizontal
    poa_global: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))  # Plane of Array Global
    poa_direct: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))  # POA Direct
    poa_diffuse: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2)) # POA Diffuse

    # Meteorological data
    ambient_temperature: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)  # C
    wind_speed: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # m/s
    relative_humidity: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # %
    pressure: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))  # Pa

    # Spectral data (for spectral mismatch calculations)
    airmass: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # Absolute air mass
    precipitable_water: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # cm
    aod: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))  # Aerosol optical depth

    # Sun position
    solar_zenith: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # degrees
    solar_azimuth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # degrees

    # Relationships
    climate_profile: Mapped["ClimateProfile"] = relationship(
        "ClimateProfile", back_populates="hourly_data"
    )

    __table_args__ = (
        UniqueConstraint("climate_profile_id", "hour_of_year", name="uq_hourly_data_hour"),
        Index("ix_hourly_climate_profile_hour", "climate_profile_id", "hour_of_year"),
        Index("ix_hourly_climate_month", "climate_profile_id", "month"),
        CheckConstraint("hour_of_year >= 1 AND hour_of_year <= 8760", name="ck_hcd_hour_range"),
        CheckConstraint("ghi >= 0", name="ck_hcd_ghi_positive"),
    )

    def __repr__(self) -> str:
        return f"<HourlyClimateData(profile_id={self.climate_profile_id}, hour={self.hour_of_year})>"


class CSERCalculation(Base):
    """
    CSER calculation results per IEC 61853-3/4.

    Stores the results of Climate Specific Energy Rating calculations
    for a given module and climate profile combination.
    """

    __tablename__ = "cser_calculations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # References
    module_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pv_modules.id", ondelete="CASCADE"), nullable=False, index=True
    )
    climate_profile_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("climate_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), index=True)

    # Calculation parameters
    calculation_name: Mapped[Optional[str]] = mapped_column(String(255))
    tilt_angle: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # degrees
    azimuth_angle: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))  # degrees
    albedo: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2), default=0.2)

    # CSER results
    cser_value: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)  # kWh/kWp
    annual_energy: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)  # kWh
    performance_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))  # PR (0-1)

    # Loss factors
    spectral_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # %
    angular_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))   # %
    thermal_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))   # %
    low_irradiance_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))  # %

    # Monthly breakdown (stored as JSON)
    monthly_energy: Mapped[Optional[dict]] = mapped_column(JSONB)  # {1: 123.4, 2: 145.6, ...}
    hourly_results: Mapped[Optional[dict]] = mapped_column(JSONB)  # Summary statistics per hour

    # Calculation metadata
    calculation_method: Mapped[Optional[str]] = mapped_column(String(50))  # 'IEC_61853', 'SIMPLIFIED'
    software_version: Mapped[Optional[str]] = mapped_column(String(50))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    calculated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    pv_module: Mapped["PVModule"] = relationship("PVModule", back_populates="cser_calculations")
    climate_profile: Mapped["ClimateProfile"] = relationship(
        "ClimateProfile", back_populates="cser_calculations"
    )
    user: Mapped[Optional["User"]] = relationship("User", back_populates="cser_calculations")

    __table_args__ = (
        Index("ix_cser_module_climate", "module_id", "climate_profile_id"),
        Index("ix_cser_calculated_at", "calculated_at"),
        CheckConstraint("cser_value >= 0", name="ck_cser_positive"),
        CheckConstraint("annual_energy >= 0", name="ck_annual_energy_positive"),
    )

    def __repr__(self) -> str:
        return f"<CSERCalculation(id={self.id}, module_id={self.module_id}, CSER={self.cser_value})>"
