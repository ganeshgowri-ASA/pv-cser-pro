"""
CRUD Operations for PV-CSER Pro Database.

Provides Create, Read, Update, Delete operations for all database models
with Pydantic validation for input data.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, and_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from .models import (
    ClimateProfile,
    CSERCalculation,
    HourlyClimateData,
    IAMData,
    PowerMatrix,
    PVModule,
    SpectralResponse,
    User,
)


# =============================================================================
# Pydantic Schemas for Input Validation
# =============================================================================


class PVModuleCreate(BaseModel):
    """Schema for creating a new PV module."""

    model_config = ConfigDict(from_attributes=True)

    module_name: str = Field(..., min_length=1, max_length=255)
    manufacturer: str = Field(..., min_length=1, max_length=255)
    model_number: Optional[str] = Field(None, max_length=100)
    technology: Optional[str] = Field(None, max_length=50)

    # STC ratings
    pmax_stc: Decimal = Field(..., gt=0, description="Maximum power at STC (W)")
    voc_stc: Decimal = Field(..., gt=0, description="Open-circuit voltage at STC (V)")
    isc_stc: Decimal = Field(..., gt=0, description="Short-circuit current at STC (A)")
    vmpp_stc: Decimal = Field(..., gt=0, description="MPP voltage at STC (V)")
    impp_stc: Decimal = Field(..., gt=0, description="MPP current at STC (A)")
    fill_factor_stc: Optional[Decimal] = Field(None, ge=0, le=1)

    # Temperature coefficients
    alpha_isc: Decimal = Field(..., description="Isc temperature coefficient (%/C or A/C)")
    beta_voc: Decimal = Field(..., description="Voc temperature coefficient (%/C or V/C)")
    gamma_pmax: Decimal = Field(..., description="Pmax temperature coefficient (%/C)")
    temp_coeff_unit: str = Field("percent", pattern="^(percent|absolute)$")

    # Physical parameters
    cell_count: Optional[int] = Field(None, gt=0)
    cell_type: Optional[str] = Field(None, max_length=50)
    module_area: Optional[Decimal] = Field(None, gt=0)
    aperture_area: Optional[Decimal] = Field(None, gt=0)

    # NMOT parameters
    nmot: Optional[Decimal] = Field(None, description="Nominal Module Operating Temp (C)")
    nmot_irradiance: Optional[Decimal] = Field(800, ge=0)
    nmot_ambient: Optional[Decimal] = Field(20)
    nmot_wind_speed: Optional[Decimal] = Field(1, ge=0)

    # Thermal parameters
    u_const: Optional[Decimal] = None
    u_wind: Optional[Decimal] = None

    owner_id: Optional[int] = None
    notes: Optional[str] = None


class PVModuleUpdate(BaseModel):
    """Schema for updating a PV module."""

    model_config = ConfigDict(from_attributes=True)

    module_name: Optional[str] = Field(None, min_length=1, max_length=255)
    manufacturer: Optional[str] = Field(None, min_length=1, max_length=255)
    model_number: Optional[str] = Field(None, max_length=100)
    technology: Optional[str] = Field(None, max_length=50)
    pmax_stc: Optional[Decimal] = Field(None, gt=0)
    voc_stc: Optional[Decimal] = Field(None, gt=0)
    isc_stc: Optional[Decimal] = Field(None, gt=0)
    vmpp_stc: Optional[Decimal] = Field(None, gt=0)
    impp_stc: Optional[Decimal] = Field(None, gt=0)
    fill_factor_stc: Optional[Decimal] = Field(None, ge=0, le=1)
    alpha_isc: Optional[Decimal] = None
    beta_voc: Optional[Decimal] = None
    gamma_pmax: Optional[Decimal] = None
    temp_coeff_unit: Optional[str] = Field(None, pattern="^(percent|absolute)$")
    cell_count: Optional[int] = Field(None, gt=0)
    cell_type: Optional[str] = Field(None, max_length=50)
    module_area: Optional[Decimal] = Field(None, gt=0)
    aperture_area: Optional[Decimal] = Field(None, gt=0)
    nmot: Optional[Decimal] = None
    nmot_irradiance: Optional[Decimal] = Field(None, ge=0)
    nmot_ambient: Optional[Decimal] = None
    nmot_wind_speed: Optional[Decimal] = Field(None, ge=0)
    u_const: Optional[Decimal] = None
    u_wind: Optional[Decimal] = None
    notes: Optional[str] = None


class PowerMatrixCreate(BaseModel):
    """Schema for creating a power matrix entry."""

    model_config = ConfigDict(from_attributes=True)

    irradiance: Decimal = Field(..., ge=0, description="Irradiance (W/m2)")
    temperature: Decimal = Field(..., description="Cell temperature (C)")
    isc: Decimal = Field(..., ge=0, description="Short-circuit current (A)")
    voc: Decimal = Field(..., ge=0, description="Open-circuit voltage (V)")
    pmax: Decimal = Field(..., ge=0, description="Maximum power (W)")
    vmpp: Decimal = Field(..., ge=0, description="MPP voltage (V)")
    impp: Decimal = Field(..., ge=0, description="MPP current (A)")
    fill_factor: Optional[Decimal] = Field(None, ge=0, le=1)
    measurement_uncertainty: Optional[Decimal] = Field(None, ge=0, le=100)
    measurement_date: Optional[datetime] = None


class SpectralResponseCreate(BaseModel):
    """Schema for creating spectral response entry."""

    model_config = ConfigDict(from_attributes=True)

    wavelength: Decimal = Field(..., gt=0, description="Wavelength (nm)")
    response: Decimal = Field(..., ge=0, description="Spectral response value")
    response_unit: str = Field("normalized", pattern="^(normalized|A/W|EQE)$")
    measurement_date: Optional[datetime] = None
    measurement_uncertainty: Optional[Decimal] = Field(None, ge=0)


class IAMDataCreate(BaseModel):
    """Schema for creating IAM data entry."""

    model_config = ConfigDict(from_attributes=True)

    angle: Decimal = Field(..., ge=0, le=90, description="Incidence angle (degrees)")
    iam_value: Decimal = Field(..., ge=0, le=1, description="IAM factor (0-1)")
    azimuth: Optional[Decimal] = Field(None, ge=0, le=360)
    measurement_date: Optional[datetime] = None
    measurement_uncertainty: Optional[Decimal] = Field(None, ge=0)


class ClimateProfileCreate(BaseModel):
    """Schema for creating a climate profile."""

    model_config = ConfigDict(from_attributes=True)

    profile_name: str = Field(..., min_length=1, max_length=100)
    profile_code: str = Field(..., min_length=1, max_length=20)
    description: Optional[str] = None
    climate_type: str = Field(..., min_length=1, max_length=50)
    is_standard: bool = True
    latitude: Optional[Decimal] = Field(None, ge=-90, le=90)
    longitude: Optional[Decimal] = Field(None, ge=-180, le=180)
    elevation: Optional[Decimal] = None
    timezone: Optional[str] = Field(None, max_length=50)
    annual_ghi: Optional[Decimal] = Field(None, ge=0)
    annual_dni: Optional[Decimal] = Field(None, ge=0)
    annual_dhi: Optional[Decimal] = Field(None, ge=0)
    avg_temperature: Optional[Decimal] = None
    avg_wind_speed: Optional[Decimal] = Field(None, ge=0)
    source: Optional[str] = Field(None, max_length=255)


class HourlyClimateDataCreate(BaseModel):
    """Schema for creating hourly climate data entry."""

    model_config = ConfigDict(from_attributes=True)

    hour_of_year: int = Field(..., ge=1, le=8760)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)
    ghi: Decimal = Field(..., ge=0, description="Global Horizontal Irradiance (W/m2)")
    dni: Optional[Decimal] = Field(None, ge=0)
    dhi: Optional[Decimal] = Field(None, ge=0)
    poa_global: Optional[Decimal] = Field(None, ge=0)
    poa_direct: Optional[Decimal] = Field(None, ge=0)
    poa_diffuse: Optional[Decimal] = Field(None, ge=0)
    ambient_temperature: Decimal = Field(..., description="Ambient temperature (C)")
    wind_speed: Optional[Decimal] = Field(None, ge=0)
    relative_humidity: Optional[Decimal] = Field(None, ge=0, le=100)
    pressure: Optional[Decimal] = Field(None, ge=0)
    airmass: Optional[Decimal] = Field(None, ge=0)
    precipitable_water: Optional[Decimal] = Field(None, ge=0)
    aod: Optional[Decimal] = Field(None, ge=0)
    solar_zenith: Optional[Decimal] = Field(None, ge=0, le=180)
    solar_azimuth: Optional[Decimal] = Field(None, ge=0, le=360)


class CSERCalculationCreate(BaseModel):
    """Schema for creating a CSER calculation result."""

    model_config = ConfigDict(from_attributes=True)

    module_id: int = Field(..., gt=0)
    climate_profile_id: int = Field(..., gt=0)
    user_id: Optional[int] = None
    calculation_name: Optional[str] = Field(None, max_length=255)
    tilt_angle: Optional[Decimal] = Field(None, ge=0, le=90)
    azimuth_angle: Optional[Decimal] = Field(None, ge=0, le=360)
    albedo: Optional[Decimal] = Field(0.2, ge=0, le=1)
    cser_value: Decimal = Field(..., ge=0, description="CSER value (kWh/kWp)")
    annual_energy: Decimal = Field(..., ge=0, description="Annual energy (kWh)")
    performance_ratio: Optional[Decimal] = Field(None, ge=0, le=1)
    spectral_loss: Optional[Decimal] = None
    angular_loss: Optional[Decimal] = None
    thermal_loss: Optional[Decimal] = None
    low_irradiance_loss: Optional[Decimal] = None
    monthly_energy: Optional[Dict[str, float]] = None
    hourly_results: Optional[Dict[str, float]] = None
    calculation_method: Optional[str] = Field(None, max_length=50)
    software_version: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = None


# =============================================================================
# PV Module CRUD Operations
# =============================================================================


def create_module(db: Session, module_data: PVModuleCreate) -> PVModule:
    """
    Create a new PV module.

    Args:
        db: Database session.
        module_data: Validated module data.

    Returns:
        Created PVModule instance.

    Raises:
        SQLAlchemyError: On database error.
    """
    db_module = PVModule(**module_data.model_dump())
    db.add(db_module)
    db.commit()
    db.refresh(db_module)
    return db_module


def get_module(db: Session, module_id: int) -> Optional[PVModule]:
    """
    Get a PV module by ID.

    Args:
        db: Database session.
        module_id: Module ID.

    Returns:
        PVModule instance or None if not found.
    """
    return db.get(PVModule, module_id)


def get_modules(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    manufacturer: Optional[str] = None,
    owner_id: Optional[int] = None,
) -> Sequence[PVModule]:
    """
    Get a list of PV modules with optional filtering.

    Args:
        db: Database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        manufacturer: Filter by manufacturer name.
        owner_id: Filter by owner ID.

    Returns:
        List of PVModule instances.
    """
    stmt = select(PVModule)

    if manufacturer:
        stmt = stmt.where(PVModule.manufacturer.ilike(f"%{manufacturer}%"))
    if owner_id:
        stmt = stmt.where(PVModule.owner_id == owner_id)

    stmt = stmt.offset(skip).limit(limit).order_by(PVModule.created_at.desc())

    return db.scalars(stmt).all()


def update_module(
    db: Session, module_id: int, module_data: PVModuleUpdate
) -> Optional[PVModule]:
    """
    Update a PV module.

    Args:
        db: Database session.
        module_id: Module ID to update.
        module_data: Updated module data.

    Returns:
        Updated PVModule instance or None if not found.
    """
    db_module = db.get(PVModule, module_id)
    if not db_module:
        return None

    update_data = module_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_module, field, value)

    db.commit()
    db.refresh(db_module)
    return db_module


def delete_module(db: Session, module_id: int) -> bool:
    """
    Delete a PV module.

    Args:
        db: Database session.
        module_id: Module ID to delete.

    Returns:
        True if deleted, False if not found.
    """
    db_module = db.get(PVModule, module_id)
    if not db_module:
        return False

    db.delete(db_module)
    db.commit()
    return True


# =============================================================================
# Power Matrix CRUD Operations
# =============================================================================


def create_power_matrix_bulk(
    db: Session, module_id: int, matrix_data: List[PowerMatrixCreate]
) -> List[PowerMatrix]:
    """
    Create multiple power matrix entries for a module.

    Args:
        db: Database session.
        module_id: Parent module ID.
        matrix_data: List of power matrix entries.

    Returns:
        List of created PowerMatrix instances.

    Raises:
        IntegrityError: If module_id doesn't exist or duplicate entries.
    """
    # Verify module exists
    module = db.get(PVModule, module_id)
    if not module:
        raise IntegrityError(
            f"Module with id {module_id} not found", params=None, orig=None
        )

    db_entries = []
    for entry in matrix_data:
        db_entry = PowerMatrix(module_id=module_id, **entry.model_dump())
        db.add(db_entry)
        db_entries.append(db_entry)

    db.commit()
    for entry in db_entries:
        db.refresh(entry)
    return db_entries


def get_power_matrix_for_module(
    db: Session,
    module_id: int,
    irradiance: Optional[Decimal] = None,
    temperature: Optional[Decimal] = None,
) -> Sequence[PowerMatrix]:
    """
    Get power matrix entries for a module.

    Args:
        db: Database session.
        module_id: Module ID.
        irradiance: Optional irradiance filter (W/m2).
        temperature: Optional temperature filter (C).

    Returns:
        List of PowerMatrix instances.
    """
    stmt = select(PowerMatrix).where(PowerMatrix.module_id == module_id)

    if irradiance is not None:
        stmt = stmt.where(PowerMatrix.irradiance == irradiance)
    if temperature is not None:
        stmt = stmt.where(PowerMatrix.temperature == temperature)

    stmt = stmt.order_by(PowerMatrix.irradiance, PowerMatrix.temperature)

    return db.scalars(stmt).all()


def delete_power_matrix_for_module(db: Session, module_id: int) -> int:
    """
    Delete all power matrix entries for a module.

    Args:
        db: Database session.
        module_id: Module ID.

    Returns:
        Number of deleted entries.
    """
    stmt = select(PowerMatrix).where(PowerMatrix.module_id == module_id)
    entries = db.scalars(stmt).all()
    count = len(entries)

    for entry in entries:
        db.delete(entry)

    db.commit()
    return count


# =============================================================================
# Spectral Response CRUD Operations
# =============================================================================


def create_spectral_response_bulk(
    db: Session, module_id: int, sr_data: List[SpectralResponseCreate]
) -> List[SpectralResponse]:
    """
    Create multiple spectral response entries for a module.

    Args:
        db: Database session.
        module_id: Parent module ID.
        sr_data: List of spectral response entries.

    Returns:
        List of created SpectralResponse instances.
    """
    module = db.get(PVModule, module_id)
    if not module:
        raise IntegrityError(
            f"Module with id {module_id} not found", params=None, orig=None
        )

    db_entries = []
    for entry in sr_data:
        db_entry = SpectralResponse(module_id=module_id, **entry.model_dump())
        db.add(db_entry)
        db_entries.append(db_entry)

    db.commit()
    for entry in db_entries:
        db.refresh(entry)
    return db_entries


def get_spectral_response_for_module(
    db: Session, module_id: int
) -> Sequence[SpectralResponse]:
    """
    Get spectral response data for a module.

    Args:
        db: Database session.
        module_id: Module ID.

    Returns:
        List of SpectralResponse instances ordered by wavelength.
    """
    stmt = (
        select(SpectralResponse)
        .where(SpectralResponse.module_id == module_id)
        .order_by(SpectralResponse.wavelength)
    )
    return db.scalars(stmt).all()


# =============================================================================
# IAM Data CRUD Operations
# =============================================================================


def create_iam_data_bulk(
    db: Session, module_id: int, iam_data: List[IAMDataCreate]
) -> List[IAMData]:
    """
    Create multiple IAM data entries for a module.

    Args:
        db: Database session.
        module_id: Parent module ID.
        iam_data: List of IAM data entries.

    Returns:
        List of created IAMData instances.
    """
    module = db.get(PVModule, module_id)
    if not module:
        raise IntegrityError(
            f"Module with id {module_id} not found", params=None, orig=None
        )

    db_entries = []
    for entry in iam_data:
        db_entry = IAMData(module_id=module_id, **entry.model_dump())
        db.add(db_entry)
        db_entries.append(db_entry)

    db.commit()
    for entry in db_entries:
        db.refresh(entry)
    return db_entries


def get_iam_data_for_module(db: Session, module_id: int) -> Sequence[IAMData]:
    """
    Get IAM data for a module.

    Args:
        db: Database session.
        module_id: Module ID.

    Returns:
        List of IAMData instances ordered by angle.
    """
    stmt = (
        select(IAMData)
        .where(IAMData.module_id == module_id)
        .order_by(IAMData.angle)
    )
    return db.scalars(stmt).all()


# =============================================================================
# Climate Profile CRUD Operations
# =============================================================================


def create_climate_profile(
    db: Session, profile_data: ClimateProfileCreate
) -> ClimateProfile:
    """
    Create a new climate profile.

    Args:
        db: Database session.
        profile_data: Validated profile data.

    Returns:
        Created ClimateProfile instance.
    """
    db_profile = ClimateProfile(**profile_data.model_dump())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile


def get_climate_profile(db: Session, profile_id: int) -> Optional[ClimateProfile]:
    """
    Get a climate profile by ID.

    Args:
        db: Database session.
        profile_id: Profile ID.

    Returns:
        ClimateProfile instance or None.
    """
    return db.get(ClimateProfile, profile_id)


def get_climate_profile_by_code(db: Session, profile_code: str) -> Optional[ClimateProfile]:
    """
    Get a climate profile by code.

    Args:
        db: Database session.
        profile_code: Profile code (e.g., 'SA', 'SC').

    Returns:
        ClimateProfile instance or None.
    """
    stmt = select(ClimateProfile).where(ClimateProfile.profile_code == profile_code)
    return db.scalars(stmt).first()


def get_climate_profiles(
    db: Session,
    is_standard: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
) -> Sequence[ClimateProfile]:
    """
    Get climate profiles with optional filtering.

    Args:
        db: Database session.
        is_standard: Filter by standard/custom profiles.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of ClimateProfile instances.
    """
    stmt = select(ClimateProfile)

    if is_standard is not None:
        stmt = stmt.where(ClimateProfile.is_standard == is_standard)

    stmt = stmt.offset(skip).limit(limit).order_by(ClimateProfile.profile_name)

    return db.scalars(stmt).all()


# =============================================================================
# Hourly Climate Data CRUD Operations
# =============================================================================


def create_hourly_climate_data_bulk(
    db: Session, climate_profile_id: int, hourly_data: List[HourlyClimateDataCreate]
) -> int:
    """
    Create hourly climate data in bulk for a profile.

    Args:
        db: Database session.
        climate_profile_id: Parent climate profile ID.
        hourly_data: List of hourly data entries.

    Returns:
        Number of entries created.
    """
    profile = db.get(ClimateProfile, climate_profile_id)
    if not profile:
        raise IntegrityError(
            f"Climate profile with id {climate_profile_id} not found",
            params=None,
            orig=None,
        )

    for entry in hourly_data:
        db_entry = HourlyClimateData(
            climate_profile_id=climate_profile_id, **entry.model_dump()
        )
        db.add(db_entry)

    db.commit()
    return len(hourly_data)


def get_hourly_climate_data(
    db: Session,
    climate_profile_id: int,
    month: Optional[int] = None,
    skip: int = 0,
    limit: int = 8760,
) -> Sequence[HourlyClimateData]:
    """
    Get hourly climate data for a profile.

    Args:
        db: Database session.
        climate_profile_id: Climate profile ID.
        month: Optional month filter (1-12).
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of HourlyClimateData instances.
    """
    stmt = select(HourlyClimateData).where(
        HourlyClimateData.climate_profile_id == climate_profile_id
    )

    if month:
        stmt = stmt.where(HourlyClimateData.month == month)

    stmt = stmt.offset(skip).limit(limit).order_by(HourlyClimateData.hour_of_year)

    return db.scalars(stmt).all()


# =============================================================================
# CSER Calculation CRUD Operations
# =============================================================================


def create_cser_calculation(
    db: Session, calc_data: CSERCalculationCreate
) -> CSERCalculation:
    """
    Create a new CSER calculation result.

    Args:
        db: Database session.
        calc_data: Validated calculation data.

    Returns:
        Created CSERCalculation instance.
    """
    db_calc = CSERCalculation(**calc_data.model_dump())
    db.add(db_calc)
    db.commit()
    db.refresh(db_calc)
    return db_calc


def get_cser_calculation(db: Session, calc_id: int) -> Optional[CSERCalculation]:
    """
    Get a CSER calculation by ID.

    Args:
        db: Database session.
        calc_id: Calculation ID.

    Returns:
        CSERCalculation instance or None.
    """
    return db.get(CSERCalculation, calc_id)


def get_cser_calculations_for_module(
    db: Session, module_id: int, skip: int = 0, limit: int = 100
) -> Sequence[CSERCalculation]:
    """
    Get CSER calculations for a specific module.

    Args:
        db: Database session.
        module_id: Module ID.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of CSERCalculation instances.
    """
    stmt = (
        select(CSERCalculation)
        .where(CSERCalculation.module_id == module_id)
        .offset(skip)
        .limit(limit)
        .order_by(CSERCalculation.calculated_at.desc())
    )
    return db.scalars(stmt).all()


def get_cser_calculations_by_user(
    db: Session, user_id: int, skip: int = 0, limit: int = 100
) -> Sequence[CSERCalculation]:
    """
    Get CSER calculations for a specific user.

    Args:
        db: Database session.
        user_id: User ID.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of CSERCalculation instances.
    """
    stmt = (
        select(CSERCalculation)
        .where(CSERCalculation.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .order_by(CSERCalculation.calculated_at.desc())
    )
    return db.scalars(stmt).all()


# =============================================================================
# User CRUD Operations
# =============================================================================


def create_user(
    db: Session,
    email: str,
    hashed_password: str,
    full_name: Optional[str] = None,
    organization: Optional[str] = None,
) -> User:
    """
    Create a new user.

    Args:
        db: Database session.
        email: User email.
        hashed_password: Hashed password.
        full_name: User's full name.
        organization: User's organization.

    Returns:
        Created User instance.
    """
    db_user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        organization=organization,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get a user by email.

    Args:
        db: Database session.
        email: User email.

    Returns:
        User instance or None.
    """
    stmt = select(User).where(User.email == email)
    return db.scalars(stmt).first()


def get_user(db: Session, user_id: int) -> Optional[User]:
    """
    Get a user by ID.

    Args:
        db: Database session.
        user_id: User ID.

    Returns:
        User instance or None.
    """
    return db.get(User, user_id)
