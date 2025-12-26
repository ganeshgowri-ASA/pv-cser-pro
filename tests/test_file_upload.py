"""
File Upload and Validation Tests for PV-CSER Pro.

Tests cover:
- File format validation
- Power matrix data validation
- Module specification validation
- Climate data validation
- Error handling and messages
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any


class TestFileFormatValidation:
    """Test file format validation."""

    def test_valid_csv_detection(self, sample_csv_file):
        """Test valid CSV file detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()
        result = validator.validate_file_format(sample_csv_file)

        assert result.is_valid
        assert result.file_type == "csv"

    def test_valid_excel_detection(self, sample_excel_file):
        """Test valid Excel file detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()
        result = validator.validate_file_format(sample_excel_file)

        assert result.is_valid
        assert result.file_type in ["xlsx", "xls"]

    def test_invalid_extension(self, temp_dir):
        """Test invalid file extension."""
        from src.data_input.validation import DataValidator

        invalid_file = temp_dir / "test.txt"
        invalid_file.write_text("not valid")

        validator = DataValidator()
        result = validator.validate_file_format(invalid_file)

        assert not result.is_valid
        assert "extension" in result.errors[0].lower() or "format" in result.errors[0].lower()

    def test_file_size_limit(self, temp_dir):
        """Test file size validation."""
        from src.data_input.validation import DataValidator

        validator = DataValidator(max_size_mb=0.001)  # 1KB limit

        # Create file larger than limit
        large_file = temp_dir / "large.csv"
        large_file.write_text("x" * 2000)

        result = validator.validate_file_size(large_file)
        assert not result.is_valid

    def test_empty_file(self, temp_dir):
        """Test empty file validation."""
        from src.data_input.validation import DataValidator

        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("")

        validator = DataValidator()
        result = validator.validate_file_format(empty_file)

        assert not result.is_valid


class TestPowerMatrixValidation:
    """Test power matrix data validation."""

    def test_valid_power_matrix(self, sample_power_matrix):
        """Test valid power matrix validation."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()
        result = validator.validate_power_matrix(
            sample_power_matrix["power_values"],
            sample_power_matrix["irradiance_levels"],
            sample_power_matrix["temperature_levels"],
        )

        assert result.is_valid
        assert len(result.errors) == 0

    def test_dimension_mismatch(self, sample_power_matrix):
        """Test power matrix dimension mismatch."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        # Wrong number of rows
        wrong_data = sample_power_matrix["power_values"][:2]

        result = validator.validate_power_matrix(
            wrong_data,
            sample_power_matrix["irradiance_levels"],
            sample_power_matrix["temperature_levels"],
        )

        assert not result.is_valid
        assert any("dimension" in e.lower() for e in result.errors)

    def test_negative_power_values(self, sample_power_matrix):
        """Test negative power value detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        # Add negative values
        bad_data = [row.copy() for row in sample_power_matrix["power_values"]]
        bad_data[0][0] = -10

        result = validator.validate_power_matrix(
            bad_data,
            sample_power_matrix["irradiance_levels"],
            sample_power_matrix["temperature_levels"],
        )

        assert not result.is_valid or len(result.warnings) > 0

    def test_non_numeric_values(self, temp_dir):
        """Test non-numeric value detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        # Create matrix with non-numeric values
        bad_data = [[100, "abc", 200], [150, 300, 350]]

        result = validator.validate_power_matrix(
            bad_data,
            [100, 500, 1000],
            [25, 50],
        )

        assert not result.is_valid

    def test_unsorted_irradiance_levels(self, sample_power_matrix):
        """Test unsorted irradiance levels warning."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        unsorted_irr = [1000, 200, 600, 400, 100, 800, 1100]

        result = validator.validate_power_matrix(
            sample_power_matrix["power_values"],
            unsorted_irr,
            sample_power_matrix["temperature_levels"],
        )

        # Should warn but not necessarily fail
        assert len(result.warnings) > 0 or not result.is_valid

    def test_missing_stc_point(self, sample_power_matrix):
        """Test warning for missing STC point."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        # Remove 1000 W/m² and 25°C
        irr_no_stc = [100, 200, 400, 600, 800, 1100]
        temp_no_stc = [15, 50, 75]
        power_no_stc = [
            [row[i] for i in [0, 1, 2, 3, 4, 6]]
            for j, row in enumerate(sample_power_matrix["power_values"])
            if j != 1
        ]

        result = validator.validate_power_matrix(
            power_no_stc,
            irr_no_stc,
            temp_no_stc,
        )

        # Should warn about missing STC
        assert len(result.warnings) > 0

    def test_power_matrix_consistency(self, sample_power_matrix):
        """Test power matrix physical consistency."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        # Power should decrease with temperature at constant irradiance
        # Power should increase with irradiance at constant temperature

        result = validator.validate_power_matrix_consistency(
            sample_power_matrix["power_values"],
            sample_power_matrix["irradiance_levels"],
            sample_power_matrix["temperature_levels"],
        )

        assert result.is_valid


class TestModuleDataValidation:
    """Test module specification validation."""

    def test_valid_module_data(self, sample_module_data):
        """Test valid module data validation."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()
        result = validator.validate_module_specs(sample_module_data)

        assert result.is_valid

    def test_missing_required_fields(self):
        """Test missing required field detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        incomplete_data = {
            "manufacturer": "Test",
            # Missing model_name, pmax_stc
        }

        result = validator.validate_module_specs(incomplete_data)

        assert not result.is_valid
        assert any("required" in e.lower() for e in result.errors)

    def test_invalid_pmax_value(self, sample_module_data):
        """Test invalid Pmax value detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        bad_data = sample_module_data.copy()
        bad_data["pmax_stc"] = -100  # Negative power

        result = validator.validate_module_specs(bad_data)

        assert not result.is_valid

    def test_unusual_temperature_coefficient(self, sample_module_data):
        """Test unusual temperature coefficient warning."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        unusual_data = sample_module_data.copy()
        unusual_data["temp_coeff_pmax"] = -1.0  # Unusually high

        result = validator.validate_module_specs(unusual_data)

        # Should warn about unusual value
        assert len(result.warnings) > 0

    def test_positive_temp_coeff_pmax(self, sample_module_data):
        """Test positive temperature coefficient for Pmax."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        bad_data = sample_module_data.copy()
        bad_data["temp_coeff_pmax"] = 0.35  # Positive (should be negative)

        result = validator.validate_module_specs(bad_data)

        # Should warn or error
        assert len(result.warnings) > 0 or not result.is_valid

    def test_fill_factor_range(self, sample_module_data):
        """Test fill factor range validation."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        bad_data = sample_module_data.copy()
        # Set values that would give impossible fill factor
        bad_data["vmp_stc"] = 50  # Higher than Voc
        bad_data["imp_stc"] = 12  # Higher than Isc

        result = validator.validate_module_specs(bad_data)

        # Fill factor would be > 1, should fail
        assert not result.is_valid or len(result.warnings) > 0

    def test_module_area_vs_cell_count(self, sample_module_data):
        """Test module area vs cell count consistency."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        suspicious_data = sample_module_data.copy()
        suspicious_data["module_area"] = 0.5  # Too small for 72 cells
        suspicious_data["num_cells"] = 144

        result = validator.validate_module_specs(suspicious_data)

        # Should warn about inconsistency
        assert len(result.warnings) > 0


class TestClimateDataValidation:
    """Test climate data validation."""

    def test_valid_climate_data(self, sample_hourly_climate_data):
        """Test valid climate data validation."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()
        result = validator.validate_climate_data(sample_hourly_climate_data)

        assert result.is_valid

    def test_wrong_length(self, sample_hourly_climate_data):
        """Test wrong data length detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        bad_data = {
            "ghi": sample_hourly_climate_data["ghi"][:1000],  # Not 8760
            "temperature": sample_hourly_climate_data["temperature"][:1000],
            "wind_speed": sample_hourly_climate_data["wind_speed"][:1000],
        }

        result = validator.validate_climate_data(bad_data)

        assert not result.is_valid
        assert any("8760" in e or "hours" in e.lower() for e in result.errors)

    def test_negative_irradiance(self, sample_hourly_climate_data):
        """Test negative irradiance detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        bad_data = sample_hourly_climate_data.copy()
        bad_data["ghi"] = bad_data["ghi"].copy()
        bad_data["ghi"][100] = -50

        result = validator.validate_climate_data(bad_data)

        assert not result.is_valid or len(result.warnings) > 0

    def test_unrealistic_temperature(self, sample_hourly_climate_data):
        """Test unrealistic temperature detection."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        bad_data = sample_hourly_climate_data.copy()
        bad_data["temperature"] = bad_data["temperature"].copy()
        bad_data["temperature"][0] = 100  # Unrealistic

        result = validator.validate_climate_data(bad_data)

        assert len(result.warnings) > 0

    def test_missing_data_columns(self):
        """Test missing data columns."""
        from src.data_input.validation import DataValidator

        validator = DataValidator()

        incomplete_data = {
            "ghi": np.zeros(8760),
            # Missing temperature and wind_speed
        }

        result = validator.validate_climate_data(incomplete_data)

        assert not result.is_valid


class TestFileImport:
    """Test file import functionality."""

    def test_csv_import(self, sample_csv_file):
        """Test CSV file import."""
        from src.data_input.power_matrix import PowerMatrixHandler

        handler = PowerMatrixHandler()
        result = handler.load_from_csv(sample_csv_file)

        assert result is not None
        assert len(result.power_values) > 0

    def test_excel_import(self, sample_excel_file):
        """Test Excel file import."""
        from src.data_input.power_matrix import PowerMatrixHandler

        handler = PowerMatrixHandler()
        result = handler.load_from_excel(sample_excel_file)

        assert result is not None
        assert len(result.power_values) > 0

    def test_json_module_import(self, sample_json_module):
        """Test JSON module import."""
        from src.data_input.module_data import ModuleDataHandler

        handler = ModuleDataHandler()
        result = handler.load_from_json(sample_json_module)

        assert result is not None
        assert result.pmax_stc == 400.0

    def test_invalid_csv_handling(self, sample_invalid_csv):
        """Test invalid CSV handling."""
        from src.data_input.power_matrix import PowerMatrixHandler

        handler = PowerMatrixHandler()

        with pytest.raises(Exception):
            handler.load_from_csv(sample_invalid_csv)


class TestValidationResult:
    """Test validation result structure."""

    def test_validation_result_structure(self):
        """Test ValidationResult class structure."""
        from src.data_input.validation import ValidationResult

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue"],
            data_type="power_matrix",
        )

        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.data_type == "power_matrix"

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        from src.data_input.validation import ValidationResult

        result = ValidationResult(
            is_valid=False,
            errors=["Critical error 1", "Critical error 2"],
            warnings=[],
            data_type="module_specs",
        )

        assert not result.is_valid
        assert len(result.errors) == 2

    def test_validation_result_summary(self):
        """Test ValidationResult summary method."""
        from src.data_input.validation import ValidationResult

        result = ValidationResult(
            is_valid=False,
            errors=["Error 1"],
            warnings=["Warning 1", "Warning 2"],
            data_type="climate_data",
        )

        summary = result.get_summary()

        assert "1 error" in summary.lower()
        assert "2 warning" in summary.lower()
