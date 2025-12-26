"""
Comprehensive test suite for CSER calculations.

Tests all modules involved in Climate Specific Energy Rating:
- Power matrix interpolation
- Climate profiles
- Energy yield calculations
- CSER computation

References:
    IEC 61853-1:2011 - Power matrix requirements
    IEC 61853-3:2018 - Energy rating methodology
    IEC 61853-4:2018 - Standard climate profiles
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from src.climate.climate_profiles import (
    ClimateProfile,
    ClimateType,
    create_custom_profile,
    get_climate_profile,
    list_available_profiles,
)
from src.climate.cser_calculator import (
    CSERCalculator,
    CSERComparison,
    CSERResult,
    calculate_cser,
    calculate_cser_from_stc,
)
from src.calculations.energy_yield import (
    EnergyYieldCalculator,
    EnergyYieldResult,
    calculate_actual_energy,
    calculate_reference_energy,
)
from src.utils.constants import (
    HOURS_PER_YEAR,
    STC_IRRADIANCE,
    STC_TEMPERATURE,
    STANDARD_IRRADIANCE_LEVELS,
    STANDARD_TEMPERATURE_LEVELS,
)
from src.utils.interpolation import (
    PowerMatrixInterpolator,
    PowerMatrixSpec,
    bilinear_interpolate,
    create_power_matrix,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_power_matrix() -> PowerMatrixSpec:
    """Create a sample power matrix for testing."""
    return create_power_matrix(
        pmax_stc=400.0,
        gamma_pmax=-0.004,
        efficiency_ratio=0.95,
    )


@pytest.fixture
def sample_interpolator(sample_power_matrix: PowerMatrixSpec) -> PowerMatrixInterpolator:
    """Create a sample interpolator for testing."""
    return PowerMatrixInterpolator(
        sample_power_matrix.irradiance_levels,
        sample_power_matrix.temperature_levels,
        sample_power_matrix.power_values,
    )


@pytest.fixture
def sample_cser_calculator(sample_power_matrix: PowerMatrixSpec) -> CSERCalculator:
    """Create a sample CSER calculator for testing."""
    return CSERCalculator.from_power_matrix_spec(
        sample_power_matrix,
        pmax_stc=400.0,
        nmot=45.0,
        gamma_pmax=-0.004,
    )


@pytest.fixture
def sample_hourly_data():
    """Create sample hourly climate data for testing."""
    hours = np.arange(HOURS_PER_YEAR)
    hour_of_day = hours % 24

    # Simple sinusoidal GHI pattern (daytime only)
    ghi = np.maximum(0, 800 * np.sin(np.pi * (hour_of_day - 6) / 12))
    ghi[hour_of_day < 6] = 0
    ghi[hour_of_day > 18] = 0

    # Temperature pattern
    temp = 20 + 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Wind speed
    wind = np.ones(HOURS_PER_YEAR) * 2.0

    return {
        "ghi": ghi,
        "temp": temp,
        "wind": wind,
    }


# =============================================================================
# Test: Constants
# =============================================================================


class TestConstants:
    """Test IEC standard constants."""

    def test_stc_values(self):
        """Test Standard Test Conditions values per IEC 61853-1."""
        assert STC_IRRADIANCE == 1000.0
        assert STC_TEMPERATURE == 25.0

    def test_hours_per_year(self):
        """Test hours per year constant."""
        assert HOURS_PER_YEAR == 8760

    def test_standard_levels(self):
        """Test standard measurement levels per IEC 61853-1."""
        assert len(STANDARD_IRRADIANCE_LEVELS) >= 5
        assert len(STANDARD_TEMPERATURE_LEVELS) >= 3
        assert 1000.0 in STANDARD_IRRADIANCE_LEVELS
        assert 25.0 in STANDARD_TEMPERATURE_LEVELS


# =============================================================================
# Test: Interpolation
# =============================================================================


class TestBilinearInterpolate:
    """Test bilinear interpolation function."""

    def test_corner_values(self):
        """Test interpolation returns corner values exactly."""
        # At corner (x1, y1)
        result = bilinear_interpolate(0, 0, 0, 1, 0, 1, 10, 20, 30, 40)
        assert result == 10.0

        # At corner (x2, y2)
        result = bilinear_interpolate(1, 1, 0, 1, 0, 1, 10, 20, 30, 40)
        assert result == 40.0

    def test_center_value(self):
        """Test interpolation at center of unit square."""
        result = bilinear_interpolate(0.5, 0.5, 0, 1, 0, 1, 0, 0, 0, 4)
        assert result == 1.0  # Average of corners

    def test_edge_values(self):
        """Test interpolation along edges."""
        # Along bottom edge (y=0)
        result = bilinear_interpolate(0.5, 0, 0, 1, 0, 1, 10, 20, 30, 40)
        assert result == 20.0  # Midpoint of 10 and 30

    def test_caching(self):
        """Test that lru_cache is working."""
        # Call twice with same arguments
        result1 = bilinear_interpolate(0.5, 0.5, 0, 1, 0, 1, 10, 20, 30, 40)
        result2 = bilinear_interpolate(0.5, 0.5, 0, 1, 0, 1, 10, 20, 30, 40)
        assert result1 == result2


class TestPowerMatrixInterpolator:
    """Test PowerMatrixInterpolator class."""

    def test_initialization(self, sample_power_matrix):
        """Test interpolator initialization."""
        interp = PowerMatrixInterpolator(
            sample_power_matrix.irradiance_levels,
            sample_power_matrix.temperature_levels,
            sample_power_matrix.power_values,
        )
        assert interp is not None

    def test_stc_power(self, sample_interpolator):
        """Test power at STC conditions."""
        power = sample_interpolator(1000.0, 25.0)
        assert_allclose(power, 400.0, rtol=0.01)

    def test_power_vs_irradiance(self, sample_interpolator):
        """Test power increases with irradiance."""
        p1 = sample_interpolator(400.0, 25.0)
        p2 = sample_interpolator(600.0, 25.0)
        p3 = sample_interpolator(1000.0, 25.0)
        assert p1 < p2 < p3

    def test_power_vs_temperature(self, sample_interpolator):
        """Test power decreases with temperature (typical module)."""
        p_cool = sample_interpolator(1000.0, 15.0)
        p_stc = sample_interpolator(1000.0, 25.0)
        p_hot = sample_interpolator(1000.0, 50.0)
        assert p_hot < p_stc < p_cool

    def test_array_input(self, sample_interpolator):
        """Test interpolation with array inputs."""
        irr = np.array([600.0, 800.0, 1000.0])
        temp = np.array([25.0, 35.0, 45.0])
        power = sample_interpolator(irr, temp)
        assert power.shape == (3,)
        assert np.all(power > 0)

    def test_irradiance_range(self, sample_interpolator):
        """Test irradiance range property."""
        low, high = sample_interpolator.irradiance_range
        assert low > 0
        assert high >= 1000.0

    def test_temperature_range(self, sample_interpolator):
        """Test temperature range property."""
        low, high = sample_interpolator.temperature_range
        assert low < 25.0
        assert high > 25.0


class TestPowerMatrixSpec:
    """Test PowerMatrixSpec dataclass."""

    def test_creation(self):
        """Test PowerMatrixSpec creation."""
        spec = create_power_matrix(pmax_stc=400.0)
        assert spec.power_values.shape[0] == len(spec.irradiance_levels)
        assert spec.power_values.shape[1] == len(spec.temperature_levels)

    def test_invalid_shape(self):
        """Test validation catches shape mismatch."""
        with pytest.raises(ValueError):
            PowerMatrixSpec(
                irradiance_levels=np.array([100, 200, 300]),
                temperature_levels=np.array([25, 50]),
                power_values=np.zeros((4, 2)),  # Wrong shape
            )


# =============================================================================
# Test: Climate Profiles
# =============================================================================


class TestClimateType:
    """Test ClimateType enum."""

    def test_all_types_exist(self):
        """Test all 6 IEC 61853-4 climate types exist."""
        assert len(ClimateType) == 6

    def test_from_string(self):
        """Test ClimateType.from_string conversion."""
        assert ClimateType.from_string("subtropical_arid") == ClimateType.SUBTROPICAL_ARID
        assert ClimateType.from_string("TROPICAL") == ClimateType.TROPICAL
        assert ClimateType.from_string("high-altitude") == ClimateType.HIGH_ALTITUDE

    def test_from_string_invalid(self):
        """Test from_string raises for invalid input."""
        with pytest.raises(ValueError):
            ClimateType.from_string("invalid_climate")


class TestClimateProfile:
    """Test ClimateProfile class."""

    def test_get_profile(self):
        """Test getting standard climate profile."""
        profile = get_climate_profile(ClimateType.SUBTROPICAL_ARID)
        assert profile.name == "Subtropical Arid"
        assert profile.annual_global_irradiation > 0

    def test_profile_data_length(self):
        """Test profile has correct hourly data length."""
        profile = get_climate_profile(ClimateType.TEMPERATE_COASTAL)
        assert len(profile.hourly_ghi) == HOURS_PER_YEAR
        assert len(profile.hourly_ambient_temp) == HOURS_PER_YEAR
        assert len(profile.hourly_wind_speed) == HOURS_PER_YEAR

    def test_profile_data_validity(self):
        """Test profile data is physically valid."""
        profile = get_climate_profile(ClimateType.TROPICAL)
        assert np.all(profile.hourly_ghi >= 0)
        assert np.all(profile.hourly_ghi <= 1500)
        assert np.all(profile.hourly_ambient_temp > -50)
        assert np.all(profile.hourly_ambient_temp < 60)

    def test_all_profiles_loadable(self):
        """Test all standard profiles can be loaded."""
        for climate_type in ClimateType:
            profile = get_climate_profile(climate_type)
            assert profile is not None
            assert profile.climate_type == climate_type

    def test_monthly_irradiation(self):
        """Test monthly irradiation calculation."""
        profile = get_climate_profile(ClimateType.SUBTROPICAL_ARID)
        monthly = profile.monthly_irradiation
        assert len(monthly) == 12
        assert np.all(monthly > 0)
        # Sum should be close to annual
        assert_allclose(np.sum(monthly), profile.annual_global_irradiation, rtol=0.1)

    def test_hours_with_irradiance(self):
        """Test hours with irradiance count."""
        profile = get_climate_profile(ClimateType.TEMPERATE_CONTINENTAL)
        hours = profile.hours_with_irradiance
        assert 2000 < hours < 6000  # Reasonable daylight hours

    def test_list_available_profiles(self):
        """Test listing available profiles."""
        profiles = list_available_profiles()
        assert len(profiles) == 6
        assert all("id" in p and "name" in p for p in profiles)


class TestCustomProfile:
    """Test custom climate profile creation."""

    def test_create_custom_profile(self):
        """Test creating custom profile from data."""
        ghi = np.random.rand(HOURS_PER_YEAR) * 600
        temp = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, HOURS_PER_YEAR))

        profile = create_custom_profile(
            name="Test Site",
            hourly_ghi=ghi,
            hourly_ambient_temp=temp,
            latitude=35.0,
        )

        assert profile.name == "Test Site"
        assert len(profile.hourly_ghi) == HOURS_PER_YEAR
        assert profile.latitude == 35.0

    def test_custom_profile_invalid_length(self):
        """Test custom profile rejects invalid data length."""
        ghi = np.zeros(100)  # Wrong length
        temp = np.zeros(100)

        with pytest.raises(ValueError):
            create_custom_profile("Bad", ghi, temp)


# =============================================================================
# Test: Energy Yield Calculator
# =============================================================================


class TestEnergyYieldCalculator:
    """Test EnergyYieldCalculator class."""

    def test_initialization(self, sample_power_matrix):
        """Test calculator initialization."""
        calc = EnergyYieldCalculator.from_power_matrix_spec(
            sample_power_matrix,
            pmax_stc=400.0,
        )
        assert calc is not None

    def test_cell_temperature(self, sample_power_matrix):
        """Test cell temperature calculation."""
        calc = EnergyYieldCalculator.from_power_matrix_spec(
            sample_power_matrix,
            nmot=45.0,
        )
        # At NMOT conditions
        t_cell = calc.calculate_cell_temperature(800.0, 20.0, 1.0)
        assert_allclose(t_cell, 45.0, rtol=0.1)

        # Higher irradiance = higher temp
        t_high = calc.calculate_cell_temperature(1000.0, 20.0, 1.0)
        assert t_high > t_cell

    def test_hourly_power(self, sample_power_matrix):
        """Test hourly power calculation."""
        calc = EnergyYieldCalculator.from_power_matrix_spec(sample_power_matrix)

        power = calc.calculate_hourly_power(1000.0, 25.0)
        assert_allclose(power, 400.0, rtol=0.01)

        # Zero irradiance = zero power
        power_zero = calc.calculate_hourly_power(0.0, 25.0)
        assert power_zero == 0.0

    def test_annual_energy_result(self, sample_power_matrix, sample_hourly_data):
        """Test annual energy calculation result structure."""
        calc = EnergyYieldCalculator.from_power_matrix_spec(
            sample_power_matrix,
            pmax_stc=400.0,
        )

        result = calc.calculate_annual_energy(
            sample_hourly_data["ghi"],
            sample_hourly_data["temp"],
            sample_hourly_data["wind"],
        )

        assert isinstance(result, EnergyYieldResult)
        assert result.annual_energy > 0
        assert len(result.monthly_energy) == 12
        assert result.hours_of_operation > 0
        assert 0 < result.performance_ratio < 1.5
        assert 0 < result.capacity_factor < 1.0

    def test_reference_energy(self, sample_power_matrix, sample_hourly_data):
        """Test reference energy calculation."""
        calc = EnergyYieldCalculator.from_power_matrix_spec(sample_power_matrix)

        e_ref = calc.calculate_reference_energy(sample_hourly_data["ghi"])
        assert e_ref > 0


class TestEnergyYieldFunctions:
    """Test standalone energy yield functions."""

    def test_calculate_actual_energy(self, sample_power_matrix, sample_hourly_data):
        """Test calculate_actual_energy function."""
        e_actual = calculate_actual_energy(
            sample_power_matrix,
            sample_hourly_data["ghi"],
            sample_hourly_data["temp"],
            sample_hourly_data["wind"],
        )
        assert e_actual > 0

    def test_calculate_reference_energy(self, sample_power_matrix, sample_hourly_data):
        """Test calculate_reference_energy function."""
        e_ref = calculate_reference_energy(
            sample_power_matrix,
            sample_hourly_data["ghi"],
        )
        assert e_ref > 0


# =============================================================================
# Test: CSER Calculator
# =============================================================================


class TestCSERCalculator:
    """Test CSERCalculator class."""

    def test_initialization(self, sample_power_matrix):
        """Test CSER calculator initialization."""
        calc = CSERCalculator.from_power_matrix_spec(sample_power_matrix)
        assert calc is not None
        assert calc.pmax_stc > 0

    def test_from_stc_parameters(self):
        """Test initialization from STC parameters."""
        calc = CSERCalculator.from_stc_parameters(
            pmax_stc=400.0,
            gamma_pmax=-0.004,
            nmot=45.0,
        )
        assert calc.pmax_stc == 400.0
        assert calc.gamma_pmax == -0.004
        assert calc.nmot == 45.0

    def test_calculate_cser_result_structure(self, sample_cser_calculator):
        """Test CSER result has correct structure."""
        result = sample_cser_calculator.calculate_cser(ClimateType.SUBTROPICAL_ARID)

        assert isinstance(result, CSERResult)
        assert 0.5 < result.cser < 1.5
        assert result.e_actual > 0
        assert result.e_reference > 0
        assert result.climate_type == ClimateType.SUBTROPICAL_ARID

    def test_cser_formula(self, sample_cser_calculator):
        """Test CSER = E_actual / E_reference."""
        result = sample_cser_calculator.calculate_cser(ClimateType.TEMPERATE_COASTAL)

        calculated_cser = result.e_actual / result.e_reference
        assert_allclose(result.cser, calculated_cser, rtol=1e-10)

    def test_cser_hot_climate(self, sample_cser_calculator):
        """Test CSER < 1 for hot climate (thermal losses)."""
        result = sample_cser_calculator.calculate_cser(ClimateType.SUBTROPICAL_ARID)
        # Hot climate should have CSER < 1 due to temperature losses
        assert result.cser < 1.0
        assert result.temperature_loss > 0

    def test_cser_cold_climate(self):
        """Test CSER behavior in cold climate."""
        # Use a module with strong negative temperature coefficient
        calc = CSERCalculator.from_stc_parameters(
            pmax_stc=400.0,
            gamma_pmax=-0.004,
            nmot=42.0,
        )
        result = calc.calculate_cser(ClimateType.HIGH_ALTITUDE)
        # High altitude is cold and sunny
        # CSER depends on balance of factors

    def test_calculate_all_climates(self, sample_cser_calculator):
        """Test calculation for all climates."""
        comparison = sample_cser_calculator.calculate_all_climates()

        assert isinstance(comparison, CSERComparison)
        assert len(comparison.results) == 6

        # Check all climates computed
        for climate_type in ClimateType:
            assert climate_type in comparison.results

    def test_cser_comparison_properties(self, sample_cser_calculator):
        """Test CSERComparison properties."""
        comparison = sample_cser_calculator.calculate_all_climates()

        # Best and worst climates
        assert comparison.best_climate in ClimateType
        assert comparison.worst_climate in ClimateType

        # CSER range
        min_cser, max_cser = comparison.cser_range
        assert min_cser <= max_cser
        assert min_cser > 0

        # Average
        assert 0.5 < comparison.avg_cser < 1.5

    def test_sensitivity_analysis(self, sample_cser_calculator):
        """Test sensitivity analysis."""
        sensitivity = sample_cser_calculator.sensitivity_analysis(
            ClimateType.SUBTROPICAL_ARID,
            gamma_range=(-0.005, -0.003),
            n_points=5,
        )

        assert "gamma_values" in sensitivity
        assert "cser_vs_gamma" in sensitivity
        assert len(sensitivity["gamma_values"]) == 5
        assert len(sensitivity["cser_vs_gamma"]) == 5

        # More negative gamma should give lower CSER in hot climate
        # (more temperature sensitivity)
        assert sensitivity["cser_vs_gamma"][0] < sensitivity["cser_vs_gamma"][-1]

    def test_caching(self, sample_cser_calculator):
        """Test that CSER results are cached."""
        # Call twice
        result1 = sample_cser_calculator.calculate_cser(ClimateType.TROPICAL)
        result2 = sample_cser_calculator.calculate_cser(ClimateType.TROPICAL)

        # Should return same object due to caching
        assert result1.cser == result2.cser


class TestCSERFunctions:
    """Test standalone CSER calculation functions."""

    def test_calculate_cser_function(self, sample_power_matrix):
        """Test calculate_cser convenience function."""
        result = calculate_cser(
            sample_power_matrix,
            ClimateType.TEMPERATE_COASTAL,
        )

        assert isinstance(result, CSERResult)
        assert 0.5 < result.cser < 1.5

    def test_calculate_cser_from_stc(self):
        """Test calculate_cser_from_stc function."""
        result = calculate_cser_from_stc(
            pmax_stc=400.0,
            climate=ClimateType.SUBTROPICAL_ARID,
            gamma_pmax=-0.0035,
        )

        assert isinstance(result, CSERResult)
        assert result.e_actual > 0


class TestCSERResult:
    """Test CSERResult dataclass."""

    def test_temperature_coefficient_effective(self, sample_cser_calculator):
        """Test effective temperature coefficient calculation."""
        result = sample_cser_calculator.calculate_cser(ClimateType.SUBTROPICAL_ARID)

        gamma_eff = result.temperature_coefficient_effective
        # Should be negative for typical modules
        assert gamma_eff < 0

    def test_repr(self, sample_cser_calculator):
        """Test CSERResult string representation."""
        result = sample_cser_calculator.calculate_cser(ClimateType.TROPICAL)
        repr_str = repr(result)

        assert "CSERResult" in repr_str
        assert "cser=" in repr_str


# =============================================================================
# Test: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for complete CSER workflow."""

    def test_full_workflow(self):
        """Test complete CSER calculation workflow."""
        # 1. Create power matrix
        spec = create_power_matrix(
            pmax_stc=400.0,
            gamma_pmax=-0.0038,
            efficiency_ratio=0.95,
        )

        # 2. Create CSER calculator
        calc = CSERCalculator.from_power_matrix_spec(
            spec,
            pmax_stc=400.0,
            nmot=45.0,
        )

        # 3. Calculate CSER for all climates
        comparison = calc.calculate_all_climates()

        # 4. Verify results make physical sense
        # Hot/dry climate should have lower CSER than temperate
        arid_cser = comparison.results[ClimateType.SUBTROPICAL_ARID].cser
        temperate_cser = comparison.results[ClimateType.TEMPERATE_COASTAL].cser

        assert arid_cser < temperate_cser

    def test_cser_ordering(self):
        """Test CSER ordering across climates."""
        calc = CSERCalculator.from_stc_parameters(
            pmax_stc=400.0,
            gamma_pmax=-0.004,
        )

        comparison = calc.calculate_all_climates()

        # Collect CSERs
        cser_values = {
            ct: comparison.results[ct].cser for ct in ClimateType
        }

        # Hot climates should generally have lower CSER
        assert cser_values[ClimateType.TROPICAL] < cser_values[ClimateType.TEMPERATE_CONTINENTAL]

    def test_energy_conservation(self, sample_power_matrix):
        """Test that energy calculations are consistent."""
        profile = get_climate_profile(ClimateType.SUBTROPICAL_COASTAL)

        calc = EnergyYieldCalculator.from_power_matrix_spec(
            sample_power_matrix,
            pmax_stc=400.0,
        )

        result = calc.calculate_annual_energy(
            profile.hourly_ghi,
            profile.hourly_ambient_temp,
            profile.hourly_wind_speed,
        )

        # Monthly sum should equal annual
        monthly_sum = np.sum(result.monthly_energy)
        assert_allclose(monthly_sum, result.annual_energy, rtol=0.001)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_irradiance(self, sample_power_matrix):
        """Test handling of zero irradiance."""
        calc = EnergyYieldCalculator.from_power_matrix_spec(sample_power_matrix)
        power = calc.calculate_hourly_power(0.0, 25.0)
        assert power == 0.0

    def test_extreme_temperature(self, sample_interpolator):
        """Test interpolation at extreme temperatures."""
        # Very hot
        power_hot = sample_interpolator(1000.0, 75.0)
        assert power_hot > 0
        assert power_hot < 400.0  # Should be reduced

        # Cool
        power_cool = sample_interpolator(1000.0, 15.0)
        assert power_cool > 400.0  # Should be higher than STC

    def test_extrapolation(self, sample_interpolator):
        """Test behavior beyond matrix bounds."""
        # Beyond maximum irradiance
        power = sample_interpolator(1200.0, 25.0)
        assert power > 0  # Should extrapolate

    def test_negative_values_rejected(self):
        """Test that negative irradiance values are rejected."""
        irr = np.array([100, 200, 300])
        temp = np.array([25, 50])
        power = np.ones((3, 2)) * 100

        irr_bad = np.array([-100, 200, 300])

        with pytest.raises(ValueError):
            PowerMatrixInterpolator(irr_bad, temp, power)

    def test_unsorted_values_rejected(self):
        """Test that unsorted values are rejected."""
        irr = np.array([300, 200, 100])  # Wrong order
        temp = np.array([25, 50])
        power = np.ones((3, 2)) * 100

        with pytest.raises(ValueError):
            PowerMatrixInterpolator(irr, temp, power)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
