"""
PV-CSER Pro Database Module.

Provides SQLAlchemy ORM models, database connection management,
and CRUD operations for Railway PostgreSQL integration.

Usage:
    from src.database import (
        # Database connection
        get_db,
        get_session,
        init_db,
        check_connection,

        # Models
        PVModule,
        PowerMatrix,
        SpectralResponse,
        IAMData,
        ClimateProfile,
        HourlyClimateData,
        CSERCalculation,
        User,

        # CRUD operations
        create_module,
        get_module,
        update_module,
        delete_module,
        create_power_matrix_bulk,
        get_power_matrix_for_module,
        create_cser_calculation,
        get_climate_profiles,
    )

    # Initialize database
    init_db()

    # Use with context manager
    with get_session() as session:
        modules = session.query(PVModule).all()
"""

# Database connection and session management
from .database import (
    Database,
    DatabaseConfig,
    check_connection,
    get_database,
    get_db,
    get_session,
    init_db,
)

# ORM Models
from .models import (
    Base,
    ClimateProfile,
    CSERCalculation,
    HourlyClimateData,
    IAMData,
    PowerMatrix,
    PVModule,
    SpectralResponse,
    User,
)

# CRUD Operations
from .crud import (
    # PV Module CRUD
    create_module,
    get_module,
    get_modules,
    update_module,
    delete_module,

    # Power Matrix CRUD
    create_power_matrix_bulk,
    get_power_matrix_for_module,
    delete_power_matrix_for_module,

    # Spectral Response CRUD
    create_spectral_response_bulk,
    get_spectral_response_for_module,

    # IAM Data CRUD
    create_iam_data_bulk,
    get_iam_data_for_module,

    # Climate Profile CRUD
    create_climate_profile,
    get_climate_profile,
    get_climate_profile_by_code,
    get_climate_profiles,

    # Hourly Climate Data CRUD
    create_hourly_climate_data_bulk,
    get_hourly_climate_data,

    # CSER Calculation CRUD
    create_cser_calculation,
    get_cser_calculation,
    get_cser_calculations_for_module,
    get_cser_calculations_by_user,

    # User CRUD
    create_user,
    get_user,
    get_user_by_email,

    # Pydantic Schemas
    PVModuleCreate,
    PVModuleUpdate,
    PowerMatrixCreate,
    SpectralResponseCreate,
    IAMDataCreate,
    ClimateProfileCreate,
    HourlyClimateDataCreate,
    CSERCalculationCreate,
)

__all__ = [
    # Database
    "Database",
    "DatabaseConfig",
    "check_connection",
    "get_database",
    "get_db",
    "get_session",
    "init_db",

    # Models
    "Base",
    "ClimateProfile",
    "CSERCalculation",
    "HourlyClimateData",
    "IAMData",
    "PowerMatrix",
    "PVModule",
    "SpectralResponse",
    "User",

    # CRUD - PV Module
    "create_module",
    "get_module",
    "get_modules",
    "update_module",
    "delete_module",

    # CRUD - Power Matrix
    "create_power_matrix_bulk",
    "get_power_matrix_for_module",
    "delete_power_matrix_for_module",

    # CRUD - Spectral Response
    "create_spectral_response_bulk",
    "get_spectral_response_for_module",

    # CRUD - IAM Data
    "create_iam_data_bulk",
    "get_iam_data_for_module",

    # CRUD - Climate Profile
    "create_climate_profile",
    "get_climate_profile",
    "get_climate_profile_by_code",
    "get_climate_profiles",

    # CRUD - Hourly Climate Data
    "create_hourly_climate_data_bulk",
    "get_hourly_climate_data",

    # CRUD - CSER Calculation
    "create_cser_calculation",
    "get_cser_calculation",
    "get_cser_calculations_for_module",
    "get_cser_calculations_by_user",

    # CRUD - User
    "create_user",
    "get_user",
    "get_user_by_email",

    # Pydantic Schemas
    "PVModuleCreate",
    "PVModuleUpdate",
    "PowerMatrixCreate",
    "SpectralResponseCreate",
    "IAMDataCreate",
    "ClimateProfileCreate",
    "HourlyClimateDataCreate",
    "CSERCalculationCreate",
]

__version__ = "0.1.0"
