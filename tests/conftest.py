"""
Pytest Configuration and Fixtures for PV-CSER Pro Tests.

This module provides shared fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import Base, DatabaseManager


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Get test database URL."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "sqlite:///:memory:"  # Use SQLite for testing by default
    )


@pytest.fixture(scope="session")
def db_engine(test_db_url: str):
    """Create test database engine."""
    engine = create_engine(test_db_url, echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create a new database session for each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def db_manager(test_db_url: str) -> Generator[DatabaseManager, None, None]:
    """Create a DatabaseManager instance for testing."""
    # Use SQLite for testing
    db = DatabaseManager("sqlite:///:memory:")
    db.initialize()
    yield db


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_module_data() -> Dict[str, Any]:
    """Sample PV module data for testing."""
    return {
        "manufacturer": "Test Solar",
        "model_name": "TS-400M",
        "serial_number": "TS-2024-001",
        "pmax_stc": 400.0,
        "voc_stc": 48.5,
        "isc_stc": 10.5,
        "vmp_stc": 40.8,
        "imp_stc": 9.8,
        "temp_coeff_pmax": -0.35,
        "temp_coeff_voc": -0.28,
        "temp_coeff_isc": 0.05,
        "module_area": 1.92,
        "cell_type": "mono-Si",
        "num_cells": 72,
        "nmot": 43.0,
        "bifacial": False,
    }


@pytest.fixture
def sample_module_data_2() -> Dict[str, Any]:
    """Second sample PV module for comparison tests."""
    return {
        "manufacturer": "Solar Tech",
        "model_name": "ST-500M",
        "serial_number": "ST-2024-001",
        "pmax_stc": 500.0,
        "voc_stc": 52.0,
        "isc_stc": 12.5,
        "vmp_stc": 44.0,
        "imp_stc": 11.36,
        "temp_coeff_pmax": -0.32,
        "temp_coeff_voc": -0.26,
        "temp_coeff_isc": 0.04,
        "module_area": 2.35,
        "cell_type": "PERC",
        "num_cells": 144,
        "nmot": 42.0,
        "bifacial": True,
        "bifaciality_factor": 0.7,
    }


@pytest.fixture
def sample_power_matrix() -> Dict[str, Any]:
    """Sample power matrix data."""
    irradiance_levels = [100, 200, 400, 600, 800, 1000, 1100]
    temperature_levels = [15, 25, 50, 75]
    pmax_stc = 400.0
    gamma = -0.35 / 100

    power_values = []
    for T in temperature_levels:
        row = []
        for G in irradiance_levels:
            low_light_factor = 1.0 if G >= 200 else (0.95 + 0.05 * (G / 200))
            P = pmax_stc * (G / 1000) * (1 + gamma * (T - 25)) * low_light_factor
            row.append(round(P, 2))
        power_values.append(row)

    return {
        "irradiance_levels": irradiance_levels,
        "temperature_levels": temperature_levels,
        "power_values": power_values,
    }


@pytest.fixture
def sample_power_matrix_df(sample_power_matrix: Dict) -> pd.DataFrame:
    """Sample power matrix as DataFrame."""
    df = pd.DataFrame(
        sample_power_matrix["power_values"],
        index=sample_power_matrix["temperature_levels"],
        columns=sample_power_matrix["irradiance_levels"],
    )
    df.index.name = "Temperature (°C)"
    df.columns.name = "Irradiance (W/m²)"
    return df


@pytest.fixture
def sample_climate_profile() -> Dict[str, Any]:
    """Sample climate profile data."""
    return {
        "profile_name": "Test Tropical",
        "profile_code": "TEST_TROP",
        "profile_type": "standard",
        "location": "Test Location",
        "country": "Test Country",
        "latitude": 1.35,
        "longitude": 103.82,
        "annual_ghi": 1630.0,
        "avg_temperature": 27.5,
        "is_standard": True,
        "source": "Test",
        "description": "Test climate profile",
    }


@pytest.fixture
def sample_hourly_climate_data() -> Dict[str, np.ndarray]:
    """Generate sample hourly climate data (8760 hours)."""
    np.random.seed(42)
    hours = 8760

    # Generate realistic patterns
    hour_of_day = np.tile(np.arange(24), 365)[:hours]

    # GHI with day/night pattern
    ghi = np.zeros(hours)
    for i in range(hours):
        if 6 <= hour_of_day[i] <= 18:
            # Peak at noon
            solar_factor = np.sin((hour_of_day[i] - 6) * np.pi / 12)
            ghi[i] = 800 * solar_factor * (0.7 + 0.3 * np.random.random())
        else:
            ghi[i] = 0

    # Temperature with daily cycle
    base_temp = 25 + 5 * np.sin(2 * np.pi * np.arange(hours) / (24 * 365))
    daily_variation = 5 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
    temperature = base_temp + daily_variation + np.random.normal(0, 1, hours)

    # Wind speed
    wind_speed = 2 + np.abs(np.random.normal(3, 1.5, hours))

    return {
        "ghi": ghi,
        "temperature": temperature,
        "wind_speed": wind_speed,
    }


@pytest.fixture
def sample_cser_results() -> Dict[str, Any]:
    """Sample CSER calculation results."""
    return {
        "cser_value": 1580.5,
        "annual_energy_yield": 632.2,
        "annual_dc_energy": 645.8,
        "specific_yield": 1580.5,
        "performance_ratio": 82.5,
        "capacity_factor": 18.0,
        "avg_cell_temperature": 42.3,
        "max_cell_temperature": 68.5,
        "operating_hours": 4380,
        "monthly_yields": {
            "Jan": 45.2, "Feb": 48.5, "Mar": 58.3, "Apr": 62.1,
            "May": 68.5, "Jun": 65.2, "Jul": 64.8, "Aug": 62.5,
            "Sep": 55.3, "Oct": 48.2, "Nov": 42.1, "Dec": 41.5,
        },
        "monthly_cser": {
            "Jan": 113.0, "Feb": 121.25, "Mar": 145.75, "Apr": 155.25,
            "May": 171.25, "Jun": 163.0, "Jul": 162.0, "Aug": 156.25,
            "Sep": 138.25, "Oct": 120.5, "Nov": 105.25, "Dec": 103.75,
        },
        "loss_breakdown": {
            "temperature": 5.2,
            "low_irradiance": 2.1,
            "spectral": 1.0,
            "iam": 2.5,
            "soiling": 2.0,
            "mismatch": 2.0,
            "wiring": 1.5,
        },
        "temperature_loss": 5.2,
        "low_irradiance_loss": 2.1,
        "spectral_loss": 1.0,
        "total_losses": 16.3,
        "calculation_method": "IEC 61853-3",
        "temperature_model": "NMOT",
    }


# =============================================================================
# FILE HANDLING FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_power_matrix_df: pd.DataFrame) -> Path:
    """Create sample CSV file for testing."""
    filepath = temp_dir / "test_power_matrix.csv"
    sample_power_matrix_df.to_csv(filepath)
    return filepath


@pytest.fixture
def sample_excel_file(temp_dir: Path, sample_power_matrix_df: pd.DataFrame) -> Path:
    """Create sample Excel file for testing."""
    filepath = temp_dir / "test_power_matrix.xlsx"
    sample_power_matrix_df.to_excel(filepath)
    return filepath


@pytest.fixture
def sample_invalid_csv(temp_dir: Path) -> Path:
    """Create invalid CSV file for testing validation."""
    filepath = temp_dir / "invalid.csv"
    with open(filepath, "w") as f:
        f.write("invalid,data\n")
        f.write("not,numeric\n")
    return filepath


@pytest.fixture
def sample_json_module(temp_dir: Path, sample_module_data: Dict) -> Path:
    """Create sample JSON module file."""
    import json
    filepath = temp_dir / "module.json"
    with open(filepath, "w") as f:
        json.dump(sample_module_data, f)
    return filepath


# =============================================================================
# CALCULATION FIXTURES
# =============================================================================

@pytest.fixture
def stc_conditions() -> Dict[str, float]:
    """Standard Test Conditions."""
    return {
        "irradiance": 1000.0,  # W/m²
        "temperature": 25.0,   # °C
        "air_mass": 1.5,
    }


@pytest.fixture
def temperature_coefficients() -> Dict[str, float]:
    """Standard temperature coefficients for mono-Si."""
    return {
        "pmax": -0.35,  # %/°C
        "voc": -0.28,   # %/°C
        "isc": 0.05,    # %/°C
    }


# =============================================================================
# STREAMLIT FIXTURES
# =============================================================================

@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state."""
    class MockSessionState(dict):
        def __getattr__(self, name):
            return self.get(name)

        def __setattr__(self, name, value):
            self[name] = value

    return MockSessionState()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_almost_equal(actual: float, expected: float, tolerance: float = 0.01):
    """Assert two floats are approximately equal."""
    assert abs(actual - expected) < tolerance, f"Expected {expected}, got {actual}"


def assert_valid_power_matrix(power_values: List[List[float]], num_irr: int, num_temp: int):
    """Assert power matrix has valid structure."""
    assert len(power_values) == num_temp
    for row in power_values:
        assert len(row) == num_irr
        for val in row:
            assert val >= 0


def assert_valid_cser_result(result: Dict[str, Any]):
    """Assert CSER result has required fields."""
    required_fields = ["cser_value", "annual_energy_yield", "performance_ratio"]
    for field in required_fields:
        assert field in result
        assert result[field] is not None


# Make helpers available as fixtures
@pytest.fixture
def assert_utils():
    """Return assertion helper functions."""
    return {
        "almost_equal": assert_almost_equal,
        "valid_power_matrix": assert_valid_power_matrix,
        "valid_cser_result": assert_valid_cser_result,
    }
