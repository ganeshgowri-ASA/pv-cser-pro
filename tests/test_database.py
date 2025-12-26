"""
Database CRUD Operations Tests for PV-CSER Pro.

Tests cover:
- Database connectivity
- PV Module CRUD operations
- Power Matrix operations
- Climate Profile operations
- CSER Calculation operations
- Session and audit logging
"""

import pytest
from datetime import datetime
from typing import Dict, Any

# Import from conftest will be available via pytest


class TestDatabaseConnectivity:
    """Test database connection and initialization."""

    def test_database_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager is not None
        assert db_manager.is_initialized()

    def test_database_connection_check(self, db_manager):
        """Test database connection check."""
        assert db_manager.check_connection()

    def test_get_table_stats(self, db_manager):
        """Test getting table statistics."""
        stats = db_manager.get_table_stats()
        assert isinstance(stats, dict)
        assert "pv_modules" in stats
        assert "power_matrices" in stats
        assert "climate_profiles" in stats
        assert "cser_calculations" in stats


class TestPVModuleCRUD:
    """Test PV Module CRUD operations."""

    def test_create_module(self, db_manager, sample_module_data):
        """Test creating a new PV module."""
        module_id = db_manager.create_module(sample_module_data)
        assert module_id is not None
        assert isinstance(module_id, int)
        assert module_id > 0

    def test_get_module(self, db_manager, sample_module_data):
        """Test retrieving a PV module."""
        module_id = db_manager.create_module(sample_module_data)
        module = db_manager.get_module(module_id)

        assert module is not None
        assert module["id"] == module_id
        assert module["manufacturer"] == sample_module_data["manufacturer"]
        assert module["model_name"] == sample_module_data["model_name"]
        assert module["pmax_stc"] == sample_module_data["pmax_stc"]

    def test_get_nonexistent_module(self, db_manager):
        """Test retrieving a non-existent module."""
        module = db_manager.get_module(99999)
        assert module is None

    def test_update_module(self, db_manager, sample_module_data):
        """Test updating a PV module."""
        module_id = db_manager.create_module(sample_module_data)

        update_data = {"pmax_stc": 420.0, "nmot": 44.0}
        result = db_manager.update_module(module_id, update_data)
        assert result is True

        module = db_manager.get_module(module_id)
        assert module["pmax_stc"] == 420.0
        assert module["nmot"] == 44.0

    def test_update_nonexistent_module(self, db_manager):
        """Test updating a non-existent module."""
        result = db_manager.update_module(99999, {"pmax_stc": 400})
        assert result is False

    def test_delete_module_soft(self, db_manager, sample_module_data):
        """Test soft deleting a module."""
        module_id = db_manager.create_module(sample_module_data)
        result = db_manager.delete_module(module_id, soft_delete=True)
        assert result is True

        # Module should not be retrievable after soft delete
        module = db_manager.get_module(module_id)
        assert module is None

    def test_delete_nonexistent_module(self, db_manager):
        """Test deleting a non-existent module."""
        result = db_manager.delete_module(99999)
        assert result is False

    def test_list_modules(self, db_manager, sample_module_data, sample_module_data_2):
        """Test listing modules."""
        db_manager.create_module(sample_module_data)
        db_manager.create_module(sample_module_data_2)

        modules = db_manager.list_modules()
        assert len(modules) >= 2

    def test_list_modules_with_filter(self, db_manager, sample_module_data, sample_module_data_2):
        """Test listing modules with filters."""
        db_manager.create_module(sample_module_data)
        db_manager.create_module(sample_module_data_2)

        # Filter by manufacturer
        modules = db_manager.list_modules(manufacturer="Test Solar")
        assert len(modules) >= 1
        assert all("Test Solar" in m["manufacturer"] for m in modules)

        # Filter by cell type
        modules = db_manager.list_modules(cell_type="PERC")
        assert len(modules) >= 1

    def test_search_modules(self, db_manager, sample_module_data):
        """Test searching modules."""
        db_manager.create_module(sample_module_data)

        # Search by manufacturer
        results = db_manager.search_modules("Test")
        assert len(results) >= 1

        # Search by model name
        results = db_manager.search_modules("TS-400")
        assert len(results) >= 1

    def test_module_fill_factor_calculation(self, db_manager, sample_module_data):
        """Test that fill factor is calculated on insert."""
        module_id = db_manager.create_module(sample_module_data)
        module = db_manager.get_module(module_id)

        # Fill factor = (Vmp * Imp) / (Voc * Isc)
        expected_ff = (40.8 * 9.8) / (48.5 * 10.5)
        assert module["fill_factor"] is not None
        assert abs(module["fill_factor"] - expected_ff) < 0.01


class TestPowerMatrixOperations:
    """Test Power Matrix operations."""

    def test_create_power_matrix(self, db_manager, sample_module_data, sample_power_matrix):
        """Test creating a power matrix."""
        module_id = db_manager.create_module(sample_module_data)
        matrix_id = db_manager.create_power_matrix(
            module_id=module_id,
            **sample_power_matrix,
        )

        assert matrix_id is not None
        assert isinstance(matrix_id, int)

    def test_get_power_matrix(self, db_manager, sample_module_data, sample_power_matrix):
        """Test retrieving a power matrix."""
        module_id = db_manager.create_module(sample_module_data)
        matrix_id = db_manager.create_power_matrix(
            module_id=module_id,
            **sample_power_matrix,
        )

        matrix = db_manager.get_power_matrix(matrix_id)
        assert matrix is not None
        assert matrix["irradiance_levels"] == sample_power_matrix["irradiance_levels"]
        assert matrix["temperature_levels"] == sample_power_matrix["temperature_levels"]
        assert len(matrix["power_values"]) == len(sample_power_matrix["temperature_levels"])

    def test_get_power_matrices_for_module(self, db_manager, sample_module_data, sample_power_matrix):
        """Test getting all power matrices for a module."""
        module_id = db_manager.create_module(sample_module_data)

        # Create multiple matrices
        db_manager.create_power_matrix(module_id=module_id, **sample_power_matrix)
        db_manager.create_power_matrix(module_id=module_id, **sample_power_matrix, notes="Second matrix")

        matrices = db_manager.get_power_matrices_for_module(module_id)
        assert len(matrices) >= 2


class TestClimateProfileOperations:
    """Test Climate Profile operations."""

    def test_create_climate_profile(self, db_manager, sample_climate_profile):
        """Test creating a climate profile."""
        profile_id = db_manager.create_climate_profile(sample_climate_profile)
        assert profile_id is not None
        assert isinstance(profile_id, int)

    def test_get_climate_profile(self, db_manager, sample_climate_profile):
        """Test retrieving a climate profile."""
        profile_id = db_manager.create_climate_profile(sample_climate_profile)
        profile = db_manager.get_climate_profile(profile_id)

        assert profile is not None
        assert profile["profile_name"] == sample_climate_profile["profile_name"]
        assert profile["annual_ghi"] == sample_climate_profile["annual_ghi"]

    def test_list_climate_profiles(self, db_manager, sample_climate_profile):
        """Test listing climate profiles."""
        db_manager.create_climate_profile(sample_climate_profile)

        profiles = db_manager.list_climate_profiles()
        assert len(profiles) >= 1

    def test_list_profiles_by_type(self, db_manager, sample_climate_profile):
        """Test filtering profiles by type."""
        db_manager.create_climate_profile(sample_climate_profile)

        profiles = db_manager.list_climate_profiles(profile_type="standard")
        assert all(p["profile_type"] == "standard" for p in profiles)


class TestCSERCalculationOperations:
    """Test CSER Calculation operations."""

    def test_create_calculation(self, db_manager, sample_module_data, sample_climate_profile, sample_cser_results):
        """Test creating a CSER calculation."""
        module_id = db_manager.create_module(sample_module_data)
        profile_id = db_manager.create_climate_profile(sample_climate_profile)

        calc_id = db_manager.create_calculation(
            module_id=module_id,
            climate_profile_id=profile_id,
            results=sample_cser_results,
        )

        assert calc_id is not None
        assert isinstance(calc_id, int)

    def test_get_calculation(self, db_manager, sample_module_data, sample_climate_profile, sample_cser_results):
        """Test retrieving a calculation."""
        module_id = db_manager.create_module(sample_module_data)
        profile_id = db_manager.create_climate_profile(sample_climate_profile)
        calc_id = db_manager.create_calculation(
            module_id=module_id,
            climate_profile_id=profile_id,
            results=sample_cser_results,
        )

        calc = db_manager.get_calculation(calc_id)
        assert calc is not None
        assert calc["cser_value"] == sample_cser_results["cser_value"]
        assert calc["performance_ratio"] == sample_cser_results["performance_ratio"]

    def test_get_calculations_for_module(self, db_manager, sample_module_data, sample_climate_profile, sample_cser_results):
        """Test getting calculations for a module."""
        module_id = db_manager.create_module(sample_module_data)
        profile_id = db_manager.create_climate_profile(sample_climate_profile)
        db_manager.create_calculation(
            module_id=module_id,
            climate_profile_id=profile_id,
            results=sample_cser_results,
        )

        calcs = db_manager.get_calculations_for_module(module_id)
        assert len(calcs) >= 1

    def test_compare_modules_for_climate(self, db_manager, sample_module_data, sample_module_data_2, sample_climate_profile, sample_cser_results):
        """Test comparing modules for a climate."""
        module_id_1 = db_manager.create_module(sample_module_data)
        module_id_2 = db_manager.create_module(sample_module_data_2)
        profile_id = db_manager.create_climate_profile(sample_climate_profile)

        db_manager.create_calculation(
            module_id=module_id_1,
            climate_profile_id=profile_id,
            results=sample_cser_results,
        )

        # Create slightly different results for second module
        results_2 = sample_cser_results.copy()
        results_2["cser_value"] = 1620.0
        db_manager.create_calculation(
            module_id=module_id_2,
            climate_profile_id=profile_id,
            results=results_2,
        )

        comparison = db_manager.compare_modules_for_climate(
            [module_id_1, module_id_2],
            profile_id,
        )
        assert len(comparison) == 2
        # Should be sorted by CSER descending
        assert comparison[0]["cser_value"] >= comparison[1]["cser_value"]


class TestAuditOperations:
    """Test audit and logging operations."""

    def test_log_file_upload(self, db_manager):
        """Test logging file uploads."""
        upload_id = db_manager.log_file_upload(
            session_id="test-session-123",
            filename="test_matrix.csv",
            file_type="csv",
            file_size=1024,
            upload_type="power_matrix",
            is_valid=True,
        )
        assert upload_id is not None

    def test_log_file_upload_with_errors(self, db_manager):
        """Test logging file uploads with validation errors."""
        upload_id = db_manager.log_file_upload(
            session_id="test-session-123",
            filename="invalid.csv",
            file_type="csv",
            file_size=512,
            upload_type="power_matrix",
            is_valid=False,
            validation_errors=["Invalid format", "Missing columns"],
            validation_warnings=["Unusual values"],
        )
        assert upload_id is not None

    def test_log_export(self, db_manager):
        """Test logging exports."""
        export_id = db_manager.log_export(
            session_id="test-session-123",
            export_type="pdf",
            filename="report.pdf",
            file_size=50000,
        )
        assert export_id is not None

    def test_session_management(self, db_manager):
        """Test session creation and update."""
        session_id = "test-session-456"

        # Create session
        db_manager.create_or_update_session(
            session_id=session_id,
            user_agent="Mozilla/5.0",
            ip_address="127.0.0.1",
        )

        # Update session
        db_manager.create_or_update_session(
            session_id=session_id,
            session_data={"current_page": "calculations"},
        )


class TestDatabaseIntegrity:
    """Test database integrity and constraints."""

    def test_unique_module_constraint(self, db_manager, sample_module_data):
        """Test unique constraint on module identity."""
        db_manager.create_module(sample_module_data)

        # Attempting to create duplicate should raise an error
        with pytest.raises(Exception):
            db_manager.create_module(sample_module_data)

    def test_cascade_delete_power_matrix(self, db_manager, sample_module_data, sample_power_matrix):
        """Test cascading delete of power matrices when module is deleted."""
        module_id = db_manager.create_module(sample_module_data)
        matrix_id = db_manager.create_power_matrix(
            module_id=module_id,
            **sample_power_matrix,
        )

        # Hard delete module
        db_manager.delete_module(module_id, soft_delete=False)

        # Power matrix should also be deleted
        matrix = db_manager.get_power_matrix(matrix_id)
        assert matrix is None

    def test_foreign_key_constraint(self, db_manager, sample_power_matrix):
        """Test foreign key constraint on power matrix."""
        # Attempting to create power matrix with non-existent module should fail
        with pytest.raises(Exception):
            db_manager.create_power_matrix(
                module_id=99999,
                **sample_power_matrix,
            )
