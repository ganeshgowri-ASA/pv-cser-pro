"""
Data validation for PV-CSER Pro application.

Provides comprehensive validation functions for all input data types
according to IEC 61853 standards.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def __bool__(self) -> bool:
        """Return True if valid."""
        return self.is_valid


class DataValidator:
    """
    Comprehensive data validator for PV module data.

    Validates module specifications, power matrices, spectral data,
    IAM data, and climate profiles according to IEC 61853 standards.
    """

    # Validation ranges for module specifications
    VALIDATION_RANGES = {
        "pmax_stc": (10.0, 1000.0),       # W
        "voc_stc": (5.0, 120.0),          # V
        "isc_stc": (0.5, 30.0),           # A
        "vmp_stc": (4.0, 100.0),          # V
        "imp_stc": (0.5, 25.0),           # A
        "temp_coeff_pmax": (-1.0, 0.0),   # %/°C
        "temp_coeff_voc": (-1.0, 0.0),    # %/°C
        "temp_coeff_isc": (-0.1, 0.2),    # %/°C
        "module_area": (0.1, 5.0),        # m²
        "nmot": (30.0, 60.0),             # °C
        "noct": (30.0, 60.0),             # °C
        "ff_stc": (50.0, 90.0),           # %
    }

    def __init__(self):
        """Initialize validator."""
        pass

    def validate_module_specs(
        self,
        data: Dict[str, Any],
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate PV module specifications.

        Args:
            data: Dictionary with module specifications
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Required fields
        required = ["manufacturer", "model_name", "pmax_stc"]
        for field in required:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
            elif field == "pmax_stc" and data[field] <= 0:
                errors.append("Pmax at STC must be positive")

        # Validate numeric ranges
        for field, (min_val, max_val) in self.VALIDATION_RANGES.items():
            if field in data and data[field] is not None:
                value = data[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} must be numeric")
                elif value < min_val or value > max_val:
                    warnings.append(
                        f"{field} ({value}) is outside typical range "
                        f"[{min_val}, {max_val}]"
                    )

        # Cross-field validations
        if all(k in data and data[k] for k in ["voc_stc", "vmp_stc"]):
            if data["vmp_stc"] >= data["voc_stc"]:
                errors.append("Vmp must be less than Voc")

        if all(k in data and data[k] for k in ["isc_stc", "imp_stc"]):
            if data["imp_stc"] >= data["isc_stc"]:
                errors.append("Imp must be less than Isc")

        # Power consistency check
        if all(k in data and data[k] for k in ["pmax_stc", "vmp_stc", "imp_stc"]):
            calculated = data["vmp_stc"] * data["imp_stc"]
            deviation = abs(calculated - data["pmax_stc"]) / data["pmax_stc"]
            if deviation > 0.02:
                warnings.append(
                    f"Pmax ({data['pmax_stc']}W) differs from Vmp*Imp "
                    f"({calculated:.1f}W) by {deviation*100:.1f}%"
                )

        # Efficiency check
        if all(k in data and data[k] for k in ["pmax_stc", "module_area"]):
            efficiency = (data["pmax_stc"] / data["module_area"]) / 10
            if efficiency < 10 or efficiency > 25:
                warnings.append(f"Module efficiency ({efficiency:.1f}%) unusual")

        is_valid = len(errors) == 0 and (not strict or len(warnings) == 0)
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_power_matrix(
        self,
        irradiance: np.ndarray,
        temperature: np.ndarray,
        power: np.ndarray,
        pmax_stc: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate power matrix data.

        Args:
            irradiance: Array of irradiance values (W/m²)
            temperature: Array of temperature values (°C)
            power: 2D array of power values
            pmax_stc: Expected Pmax at STC for cross-validation

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Dimension check
        expected_shape = (len(irradiance), len(temperature))
        if power.shape != expected_shape:
            errors.append(
                f"Power matrix shape {power.shape} doesn't match "
                f"expected {expected_shape}"
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=[])

        # Check for negative values
        if np.any(power < 0):
            errors.append("Power matrix contains negative values")

        # Check for NaN/Inf
        nan_count = np.isnan(power).sum()
        inf_count = np.isinf(power).sum()
        if nan_count > 0:
            warnings.append(f"Power matrix contains {nan_count} NaN values")
        if inf_count > 0:
            errors.append(f"Power matrix contains {inf_count} infinite values")

        # Irradiance range check
        if irradiance.min() > 200:
            warnings.append(f"Min irradiance ({irradiance.min()}) > 200 W/m²")
        if irradiance.max() < 1000:
            warnings.append(f"Max irradiance ({irradiance.max()}) < 1000 W/m²")

        # Temperature range check
        if temperature.min() > 25:
            warnings.append(f"Min temperature ({temperature.min()}) > 25°C")
        if temperature.max() < 50:
            warnings.append(f"Max temperature ({temperature.max()}) < 50°C")

        # Check STC coverage
        has_stc_g = any(abs(g - 1000) < 1 for g in irradiance)
        has_stc_t = any(abs(t - 25) < 1 for t in temperature)
        if not has_stc_g:
            warnings.append("1000 W/m² irradiance level not present")
        if not has_stc_t:
            warnings.append("25°C temperature level not present")

        # Monotonicity checks
        for j, t in enumerate(temperature):
            col = power[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                diffs = np.diff(col[valid])
                # Allow small negative diffs due to measurement noise
                if np.any(diffs < -power[valid][:-1] * 0.02):
                    warnings.append(
                        f"Power not monotonic with irradiance at T={t}°C"
                    )

        # Temperature behavior check
        for i, g in enumerate(irradiance):
            row = power[i, :]
            valid = ~np.isnan(row)
            if valid.sum() > 1:
                diffs = np.diff(row[valid])
                # Power should generally decrease with temperature
                if np.sum(diffs > row[valid][:-1] * 0.01) > 1:
                    warnings.append(
                        f"Power increases with temperature at G={g} W/m²"
                    )

        # STC validation
        if pmax_stc is not None and has_stc_g and has_stc_t:
            idx_g = np.argmin(np.abs(irradiance - 1000))
            idx_t = np.argmin(np.abs(temperature - 25))
            measured = power[idx_g, idx_t]
            if not np.isnan(measured):
                deviation = abs(measured - pmax_stc) / pmax_stc * 100
                if deviation > 3:
                    warnings.append(
                        f"Matrix Pmax at STC ({measured:.1f}W) differs from "
                        f"specified ({pmax_stc:.1f}W) by {deviation:.1f}%"
                    )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_spectral_response(
        self,
        wavelength: np.ndarray,
        response: np.ndarray,
    ) -> ValidationResult:
        """
        Validate spectral response data.

        Args:
            wavelength: Array of wavelength values (nm)
            response: Array of spectral response values (A/W or normalized)

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        if len(wavelength) != len(response):
            errors.append("Wavelength and response arrays must have same length")
            return ValidationResult(is_valid=False, errors=errors, warnings=[])

        if len(wavelength) < 10:
            warnings.append("Spectral response has fewer than 10 data points")

        # Wavelength range (typical silicon: 300-1200 nm)
        if wavelength.min() > 350:
            warnings.append(f"Min wavelength ({wavelength.min()}nm) > 350nm")
        if wavelength.max() < 1100:
            warnings.append(f"Max wavelength ({wavelength.max()}nm) < 1100nm")

        # Check for negative values
        if np.any(response < 0):
            errors.append("Spectral response contains negative values")

        # Check ordering
        if not np.all(np.diff(wavelength) > 0):
            warnings.append("Wavelength values should be monotonically increasing")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_iam_data(
        self,
        angles: np.ndarray,
        iam: np.ndarray,
    ) -> ValidationResult:
        """
        Validate Incidence Angle Modifier (IAM) data.

        Args:
            angles: Array of incidence angles (degrees)
            iam: Array of IAM values (normalized to 1 at 0°)

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        if len(angles) != len(iam):
            errors.append("Angle and IAM arrays must have same length")
            return ValidationResult(is_valid=False, errors=errors, warnings=[])

        # Check angle range
        if angles.min() > 0:
            warnings.append("IAM data should include 0° angle")
        if angles.max() < 80:
            warnings.append("IAM data should extend to at least 80°")

        # Check normalization at 0°
        zero_idx = np.argmin(np.abs(angles))
        if abs(iam[zero_idx] - 1.0) > 0.02:
            warnings.append(f"IAM at 0° should be 1.0, got {iam[zero_idx]:.3f}")

        # Check monotonicity (IAM should decrease with angle)
        sorted_idx = np.argsort(angles)
        iam_sorted = iam[sorted_idx]
        if not np.all(np.diff(iam_sorted) <= 0.01):
            warnings.append("IAM should generally decrease with angle")

        # Check range
        if np.any(iam < 0) or np.any(iam > 1.1):
            errors.append("IAM values should be between 0 and 1")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_climate_profile(
        self,
        data: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate climate profile data.

        Args:
            data: Dictionary with climate profile data

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        required = ["profile_name"]
        for field in required:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")

        # Validate irradiance data if present
        if "irradiance_data" in data and data["irradiance_data"]:
            irr = np.array(data["irradiance_data"])
            if np.any(irr < 0):
                errors.append("Irradiance data contains negative values")
            if irr.max() > 1500:
                warnings.append(f"Max irradiance ({irr.max()}) exceeds 1500 W/m²")

        # Validate temperature data if present
        if "temperature_data" in data and data["temperature_data"]:
            temp = np.array(data["temperature_data"])
            if temp.min() < -50 or temp.max() > 60:
                warnings.append("Temperature values outside typical range [-50, 60]°C")

        # Validate coordinates
        if "latitude" in data and data["latitude"] is not None:
            if not -90 <= data["latitude"] <= 90:
                errors.append("Latitude must be between -90 and 90")

        if "longitude" in data and data["longitude"] is not None:
            if not -180 <= data["longitude"] <= 180:
                errors.append("Longitude must be between -180 and 180")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_uploaded_file(
        self,
        file_content: bytes,
        file_type: str,
        expected_format: str = "power_matrix",
    ) -> ValidationResult:
        """
        Validate an uploaded file.

        Args:
            file_content: Raw file content
            file_type: File extension ('.csv', '.xlsx')
            expected_format: Expected data format

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        try:
            from io import BytesIO

            if file_type in [".csv", "csv"]:
                df = pd.read_csv(BytesIO(file_content), index_col=0)
            elif file_type in [".xlsx", ".xls", "xlsx", "xls"]:
                df = pd.read_excel(BytesIO(file_content), index_col=0)
            else:
                errors.append(f"Unsupported file type: {file_type}")
                return ValidationResult(is_valid=False, errors=errors, warnings=[])

            if df.empty:
                errors.append("File contains no data")
            elif expected_format == "power_matrix":
                # Check that index and columns are numeric
                try:
                    df.index.astype(float)
                    df.columns.astype(float)
                except (ValueError, TypeError):
                    errors.append(
                        "Power matrix must have numeric row and column headers"
                    )

                # Check values are numeric
                if not df.apply(pd.to_numeric, errors='coerce').notna().all().all():
                    warnings.append("Some values could not be converted to numbers")

        except pd.errors.EmptyDataError:
            errors.append("File is empty")
        except pd.errors.ParserError as e:
            errors.append(f"Failed to parse file: {str(e)}")
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
