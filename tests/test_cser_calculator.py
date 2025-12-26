"""
Unit Tests for CSER Calculator Module

Tests for IEC 61853-3 energy rating calculations including:
- Bilinear interpolation
- Module temperature modeling
- Power calculations
- CSER rating
- Climate profile loading

References:
    IEC 61853-3:2018 - Energy rating of PV modules
    IEC 61853-4:2018 - Standard reference climatic profiles
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.climate.interpolation import (
    InterpolationError,
    bilinear_interpolate,
    interpolate_power_matrix,
    validate_power_matrix,
    extrapolate_power,
)
from src.climate.climate_profiles import (
    ClimateProfile,
    CustomProfileBuilder,
    IEC_PROFILE_CODES,
    IEC_STANDARD_PROFILES,
    load_standard_profile,
    load_all_standard_profiles,
    validate_climate_data,
)
from src.climate.cser_calculator import (
    ModuleParameters,
    CSERResult,
    calculate_module_temperature,
    calculate_hourly_power,
    apply_spectral_correction,
    apply_iam_correction,
    calculate_annual_energy,
    calculate_cser_rating,
    calculate_performance_ratio,
    calculate_capacity_factor,
    calculate_specific_yield,
    run_cser_analysis,
    compare_climates,
    STC_IRRADIANCE,
    STC_TEMPERATURE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_power_matrix():
    """Create a sample power matrix for testing."""
    G_grid = np.array([200, 400, 600, 800, 1000, 1100], dtype=np.float64)
    T_grid = np.array([15, 25, 50, 75], dtype=np.float64)

    # Generate power matrix: P = P_stc * (G/1000) * (1 + gamma*(T-25))
    P_stc = 400  # Watts
    gamma = -0.004  # -0.4%/C

    P_matrix = np.zeros((len(T_grid), len(G_grid)), dtype=np.float64)
    for i, T in enumerate(T_grid):
        for j, G in enumerate(G_grid):
            P_matrix[i, j] = P_stc * (G / 1000) * (1 + gamma * (T - 25))

    return G_grid, T_grid, P_matrix, P_stc


@pytest.fixture
def sample_module(sample_power_matrix):
    """Create a sample module for testing."""
    G_grid, T_grid, P_matrix, P_stc = sample_power_matrix
    return ModuleParameters(
        P_stc=P_stc,
        noct=45.0,
        gamma_pmax=-0.4,
        G_grid=G_grid,
        T_grid=T_grid,
        P_matrix=P_matrix
    )


@pytest.fixture
def tropical_climate():
    """Load tropical climate profile for testing."""
    return load_standard_profile('tropical')


# =============================================================================
# Interpolation Tests
# =============================================================================

class TestBilinearInterpolation:
    """Tests for bilinear interpolation functions."""

    def test_interpolate_center_point(self, sample_power_matrix):
        """Test interpolation at center of grid cell."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        # Interpolate at center of first cell
        G = (G_grid[0] + G_grid[1]) / 2  # 300
        T = (T_grid[0] + T_grid[1]) / 2  # 20

        result = bilinear_interpolate(G, T, G_grid, T_grid, P_matrix)

        # Result should be between corner values
        assert P_matrix[0, 0] <= result <= P_matrix[1, 1]

    def test_interpolate_grid_point(self, sample_power_matrix):
        """Test interpolation at exact grid point."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        # Interpolate at exact grid point
        result = bilinear_interpolate(G_grid[2], T_grid[1], G_grid, T_grid, P_matrix)

        # Should match matrix value exactly
        np.testing.assert_almost_equal(result, P_matrix[1, 2], decimal=5)

    def test_interpolate_at_stc(self, sample_power_matrix):
        """Test interpolation at STC conditions (1000 W/m^2, 25 C)."""
        G_grid, T_grid, P_matrix, P_stc = sample_power_matrix

        result = bilinear_interpolate(1000, 25, G_grid, T_grid, P_matrix)

        # At STC, power should equal rated power
        np.testing.assert_almost_equal(result, P_stc, decimal=1)

    def test_extrapolation_within_limit(self, sample_power_matrix):
        """Test extrapolation within allowed limit."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        # Extrapolate slightly beyond grid (within 20% limit)
        G_max = G_grid[-1]
        G_test = G_max * 1.1  # 10% beyond

        result = bilinear_interpolate(G_test, 25, G_grid, T_grid, P_matrix, extrapolate=True)

        # Should return a positive value
        assert result > 0

    def test_extrapolation_beyond_limit_raises(self, sample_power_matrix):
        """Test that extrapolation beyond limit raises error."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        # Try to extrapolate 50% beyond grid
        G_test = G_grid[-1] * 1.5

        with pytest.raises(InterpolationError):
            bilinear_interpolate(G_test, 25, G_grid, T_grid, P_matrix, extrapolate=True)

    def test_extrapolation_disabled_raises(self, sample_power_matrix):
        """Test that out-of-bounds raises error when extrapolation disabled."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        # Try to interpolate beyond grid
        G_test = G_grid[-1] + 10

        with pytest.raises(InterpolationError):
            bilinear_interpolate(G_test, 25, G_grid, T_grid, P_matrix, extrapolate=False)

    def test_nan_input_raises(self, sample_power_matrix):
        """Test that NaN input raises error."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        with pytest.raises(ValueError):
            bilinear_interpolate(np.nan, 25, G_grid, T_grid, P_matrix)

    def test_interpolate_power_matrix_array(self, sample_power_matrix):
        """Test power matrix interpolation with array inputs."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        G_values = np.array([300, 500, 700, 900])
        T_values = np.array([20, 30, 40, 50])

        result = interpolate_power_matrix(G_values, T_values, G_grid, T_grid, P_matrix)

        assert len(result) == 4
        assert all(result > 0)
        # Power should increase with irradiance
        assert result[0] < result[-1]


class TestValidatePowerMatrix:
    """Tests for power matrix validation."""

    def test_valid_matrix(self, sample_power_matrix):
        """Test validation of a valid power matrix."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix

        is_valid, error = validate_power_matrix(G_grid, T_grid, P_matrix)

        assert is_valid
        assert error is None

    def test_invalid_shape(self):
        """Test validation catches shape mismatch."""
        G_grid = np.array([200, 400, 600])
        T_grid = np.array([15, 25])
        P_matrix = np.zeros((3, 3))  # Wrong shape

        is_valid, error = validate_power_matrix(G_grid, T_grid, P_matrix)

        assert not is_valid
        assert "shape" in error.lower()

    def test_negative_power(self, sample_power_matrix):
        """Test validation catches negative power values."""
        G_grid, T_grid, P_matrix, _ = sample_power_matrix
        P_matrix_neg = P_matrix.copy()
        P_matrix_neg[0, 0] = -10

        is_valid, error = validate_power_matrix(G_grid, T_grid, P_matrix_neg)

        assert not is_valid
        assert "negative" in error.lower()

    def test_insufficient_points(self):
        """Test validation catches insufficient grid points."""
        G_grid = np.array([200, 400])  # Only 2 points
        T_grid = np.array([15, 25, 50])
        P_matrix = np.zeros((3, 2))

        is_valid, error = validate_power_matrix(G_grid, T_grid, P_matrix)

        assert not is_valid
        assert "at least" in error.lower()


# =============================================================================
# Temperature Modeling Tests
# =============================================================================

class TestModuleTemperature:
    """Tests for module temperature calculations."""

    def test_noct_conditions(self):
        """Test module temperature at NOCT conditions."""
        # At NOCT conditions: G=800, T_amb=20, wind=1 m/s
        T_module = calculate_module_temperature(
            irradiance=800,
            ambient_temp=20,
            noct=45.0,
            efficiency=0.0  # Ignore efficiency for NOCT
        )

        # Should equal NOCT
        np.testing.assert_almost_equal(T_module, 45.0, decimal=1)

    def test_zero_irradiance(self):
        """Test module temperature with zero irradiance."""
        T_module = calculate_module_temperature(
            irradiance=0,
            ambient_temp=25,
            noct=45.0
        )

        # Module temp should equal ambient
        np.testing.assert_almost_equal(T_module, 25.0, decimal=5)

    def test_high_irradiance(self):
        """Test module temperature at high irradiance."""
        T_module = calculate_module_temperature(
            irradiance=1000,
            ambient_temp=35,
            noct=45.0,
            efficiency=0.20
        )

        # Module should be hotter than ambient
        assert T_module > 35
        # But not unreasonably hot
        assert T_module < 80

    def test_efficiency_effect(self):
        """Test that higher efficiency reduces module temperature."""
        T_low_eff = calculate_module_temperature(800, 25, noct=45, efficiency=0.10)
        T_high_eff = calculate_module_temperature(800, 25, noct=45, efficiency=0.22)

        # Higher efficiency should mean lower temperature
        assert T_high_eff < T_low_eff

    def test_array_input(self):
        """Test temperature calculation with array inputs."""
        G = np.array([0, 200, 500, 800, 1000])
        T_amb = np.array([20, 22, 25, 28, 30])

        T_module = calculate_module_temperature(G, T_amb, noct=45.0)

        assert len(T_module) == 5
        # Temperature should increase with irradiance
        assert T_module[0] < T_module[-1]


# =============================================================================
# Climate Profile Tests
# =============================================================================

class TestClimateProfiles:
    """Tests for climate profile loading and validation."""

    def test_load_tropical(self):
        """Test loading tropical climate profile."""
        profile = load_standard_profile('tropical')

        assert profile.code == 'TRO'
        assert profile.annual_irradiation == 2100
        assert profile.average_ambient_temperature == 28.0
        assert len(profile.hourly_irradiance) == 8760
        assert len(profile.hourly_temperature) == 8760

    def test_load_all_profiles(self):
        """Test loading all standard profiles."""
        profiles = load_all_standard_profiles()

        assert len(profiles) == 6
        assert 'tropical' in profiles
        assert 'desert' in profiles
        assert 'temperate' in profiles
        assert 'cold' in profiles
        assert 'marine' in profiles
        assert 'arctic' in profiles

    def test_load_by_code(self):
        """Test loading profile by short code."""
        profile = load_standard_profile('TRO')

        assert profile.code == 'TRO'
        assert profile.name == 'Tropical Hot-Humid'

    def test_invalid_profile_raises(self):
        """Test that invalid profile code raises error."""
        with pytest.raises(ValueError):
            load_standard_profile('INVALID')

    def test_profile_irradiation_matches(self, tropical_climate):
        """Test that hourly data approximately matches annual total."""
        calculated_annual = tropical_climate.get_annual_energy_potential()

        # Should be within 10% of stated annual irradiation
        np.testing.assert_allclose(
            calculated_annual,
            tropical_climate.annual_irradiation,
            rtol=0.1
        )

    def test_validate_climate_data_valid(self, tropical_climate):
        """Test validation of valid climate data."""
        is_valid, messages = validate_climate_data(
            tropical_climate.hourly_irradiance,
            tropical_climate.hourly_temperature
        )

        assert is_valid
        # May have warnings but no errors

    def test_validate_climate_data_wrong_length(self):
        """Test validation catches wrong array length."""
        G = np.zeros(1000)  # Wrong length
        T = np.zeros(8760)

        is_valid, messages = validate_climate_data(G, T)

        assert not is_valid
        assert any("8760" in msg for msg in messages)

    def test_validate_negative_irradiance(self):
        """Test validation catches negative irradiance."""
        G = np.random.uniform(0, 1000, 8760)
        G[100] = -50  # Negative value
        T = np.random.uniform(10, 40, 8760)

        is_valid, messages = validate_climate_data(G, T)

        assert not is_valid
        assert any("negative" in msg.lower() for msg in messages)

    def test_iec_standard_profiles_dict(self):
        """Test IEC_STANDARD_PROFILES dictionary."""
        assert len(IEC_STANDARD_PROFILES) == 6

        for name, data in IEC_STANDARD_PROFILES.items():
            assert 'code' in data
            assert 'name' in data
            assert 'annual_irradiation' in data
            assert 'avg_temp' in data


class TestCustomProfileBuilder:
    """Tests for custom profile builder."""

    def test_build_simple_profile(self):
        """Test building a simple custom profile."""
        builder = CustomProfileBuilder()
        profile = (builder
            .set_name("Test Location")
            .set_annual_irradiation(1500)
            .set_average_temperature(20)
            .build())

        assert profile.name == "Test Location"
        assert profile.annual_irradiation == 1500
        assert len(profile.hourly_irradiance) == 8760

    def test_build_with_hourly_data(self):
        """Test building profile with custom hourly data."""
        G = np.random.uniform(0, 1000, 8760)
        T = np.random.uniform(10, 35, 8760)

        builder = CustomProfileBuilder()
        profile = builder.set_hourly_data(G, T).build()

        np.testing.assert_array_equal(profile.hourly_irradiance, G)
        np.testing.assert_array_equal(profile.hourly_temperature, T)

    def test_build_from_monthly(self):
        """Test building profile from monthly averages."""
        monthly_G = [3000, 3500, 4500, 5500, 6000, 6500, 6500, 6000, 5000, 4000, 3500, 3000]
        monthly_T = [10, 12, 15, 18, 22, 26, 28, 27, 24, 19, 14, 11]

        builder = CustomProfileBuilder()
        profile = builder.from_monthly_averages(monthly_G, monthly_T).build()

        assert len(profile.hourly_irradiance) == 8760
        assert len(profile.hourly_temperature) == 8760

    def test_invalid_hourly_length_raises(self):
        """Test that invalid array length raises error."""
        builder = CustomProfileBuilder()

        with pytest.raises(ValueError):
            builder.set_hourly_data(np.zeros(100), np.zeros(100))


# =============================================================================
# CSER Calculation Tests
# =============================================================================

class TestCSERCalculations:
    """Tests for CSER calculation functions."""

    def test_cser_rating_unity(self):
        """Test CSER rating equals 1.0 when actual equals reference."""
        cser = calculate_cser_rating(1000, 1000)

        np.testing.assert_almost_equal(cser, 1.0, decimal=5)

    def test_cser_rating_above_unity(self):
        """Test CSER > 1 when actual exceeds reference."""
        cser = calculate_cser_rating(1200, 1000)

        assert cser > 1.0
        np.testing.assert_almost_equal(cser, 1.2, decimal=5)

    def test_cser_rating_below_unity(self):
        """Test CSER < 1 when actual is less than reference."""
        cser = calculate_cser_rating(800, 1000)

        assert cser < 1.0
        np.testing.assert_almost_equal(cser, 0.8, decimal=5)

    def test_cser_rating_zero_reference_raises(self):
        """Test that zero reference energy raises error."""
        with pytest.raises(ValueError):
            calculate_cser_rating(1000, 0)

    def test_performance_ratio_range(self):
        """Test performance ratio is in expected range."""
        pr = calculate_performance_ratio(
            annual_energy_kwh=600,
            P_stc=400,
            annual_irradiation_kwh_m2=2000
        )

        # PR should be between 0 and 1
        assert 0 < pr < 1
        # For typical values, PR should be around 0.7-0.9
        assert 0.5 < pr < 1.0

    def test_capacity_factor_range(self):
        """Test capacity factor is in expected range."""
        cf = calculate_capacity_factor(
            annual_energy_kwh=600,
            P_stc=400
        )

        # CF should be between 0 and 1
        assert 0 < cf < 1
        # For typical PV, CF is around 0.10-0.25
        assert 0.10 < cf < 0.30

    def test_specific_yield(self):
        """Test specific yield calculation."""
        sy = calculate_specific_yield(
            annual_energy_kwh=600,
            P_stc=400
        )

        # SY = E / P = 600 / 0.4 = 1500 kWh/kWp
        np.testing.assert_almost_equal(sy, 1500, decimal=1)

    def test_annual_energy_positive(self, tropical_climate, sample_module):
        """Test that annual energy is positive."""
        energy, power, temp = calculate_annual_energy(
            tropical_climate,
            sample_module
        )

        assert energy > 0
        assert len(power) == 8760
        assert len(temp) == 8760

    def test_annual_energy_reasonable_range(self, tropical_climate, sample_module):
        """Test that annual energy is in reasonable range."""
        energy, _, _ = calculate_annual_energy(tropical_climate, sample_module)

        # For 400W module in tropical climate (2100 kWh/m^2)
        # Energy should be roughly P_stc * Annual_irr / 1000 * PR
        # = 0.4 * 2100 * 0.8 = 672 kWh
        assert 400 < energy < 1000


class TestRunCSERAnalysis:
    """Tests for complete CSER analysis."""

    def test_run_analysis_returns_result(self, sample_module):
        """Test that run_cser_analysis returns CSERResult."""
        result = run_cser_analysis('tropical', sample_module)

        assert isinstance(result, CSERResult)
        assert result.cser > 0
        assert result.annual_energy_kwh > 0
        assert len(result.hourly_power) == 8760

    def test_compare_climates(self, sample_module):
        """Test comparing module across climates."""
        results = compare_climates(sample_module, ['tropical', 'temperate'])

        assert 'tropical' in results
        assert 'temperate' in results

        # Tropical should have higher energy due to more irradiation
        assert results['tropical'].annual_energy_kwh > results['temperate'].annual_energy_kwh

    def test_compare_all_climates(self, sample_module):
        """Test comparing module across all climates."""
        results = compare_climates(sample_module)

        assert len(results) == 6

        # Desert should have highest energy (highest irradiation)
        desert_energy = results['desert'].annual_energy_kwh
        arctic_energy = results['arctic'].annual_energy_kwh

        assert desert_energy > arctic_energy

    def test_cser_values_reasonable(self, sample_module):
        """Test that CSER values are in reasonable range."""
        results = compare_climates(sample_module)

        for climate, result in results.items():
            # CSER should typically be between 0.6 and 1.4
            assert 0.5 < result.cser < 1.5, f"{climate} CSER out of range: {result.cser}"

    def test_performance_ratio_values(self, sample_module):
        """Test performance ratio values for all climates."""
        results = compare_climates(sample_module)

        for climate, result in results.items():
            # PR should be between 0.6 and 1.0
            assert 0.5 < result.performance_ratio <= 1.0, \
                f"{climate} PR out of range: {result.performance_ratio}"


# =============================================================================
# Correction Factor Tests
# =============================================================================

class TestCorrectionFactors:
    """Tests for spectral and IAM correction factors."""

    def test_spectral_correction_am15(self):
        """Test spectral correction at AM1.5 (no correction)."""
        power = 400.0
        corrected = apply_spectral_correction(power, air_mass=1.5, precipitable_water=2.0)

        # At AM1.5 and PW=2.0, correction should be minimal
        np.testing.assert_allclose(corrected, power, rtol=0.02)

    def test_spectral_correction_high_am(self):
        """Test spectral correction at high air mass."""
        power = 400.0
        corrected = apply_spectral_correction(power, air_mass=3.0, precipitable_water=2.0)

        # Correction should reduce power for c-Si at high AM
        assert corrected != power

    def test_iam_normal_incidence(self):
        """Test IAM at normal incidence (0 degrees)."""
        power = 400.0
        corrected = apply_iam_correction(power, aoi=0.0)

        # At normal incidence, IAM should be ~1.0
        np.testing.assert_allclose(corrected, power, rtol=0.01)

    def test_iam_high_angle(self):
        """Test IAM at high angle of incidence."""
        power = 400.0
        corrected = apply_iam_correction(power, aoi=60.0)

        # At 60 degrees, IAM should reduce power significantly
        assert corrected < power
        # But not to zero
        assert corrected > 0

    def test_iam_90_degrees(self):
        """Test IAM at 90 degrees (grazing angle)."""
        power = 400.0
        corrected = apply_iam_correction(power, aoi=85.0, iam_model='physical')

        # At very high angles, almost no power
        assert corrected < power * 0.5

    def test_iam_array_input(self):
        """Test IAM with array input."""
        power = np.array([400.0, 400.0, 400.0])
        aoi = np.array([0.0, 30.0, 60.0])

        corrected = apply_iam_correction(power, aoi)

        assert len(corrected) == 3
        # Power should decrease with increasing AOI
        assert corrected[0] > corrected[1] > corrected[2]


# =============================================================================
# Module Parameters Tests
# =============================================================================

class TestModuleParameters:
    """Tests for ModuleParameters class."""

    def test_default_power_matrix_generation(self):
        """Test automatic power matrix generation."""
        module = ModuleParameters(P_stc=400)

        assert module.P_matrix is not None
        assert module.P_matrix.shape == (len(module.T_grid), len(module.G_grid))

    def test_power_at_stc(self):
        """Test that generated matrix gives P_stc at STC."""
        module = ModuleParameters(P_stc=400)

        # Find indices for STC (1000 W/m^2, 25 C)
        G_idx = np.argmin(np.abs(module.G_grid - 1000))
        T_idx = np.argmin(np.abs(module.T_grid - 25))

        # Power at STC should equal P_stc
        np.testing.assert_allclose(
            module.P_matrix[T_idx, G_idx],
            module.P_stc,
            rtol=0.01
        )

    def test_temperature_coefficient_effect(self):
        """Test that temperature coefficient affects power matrix."""
        module = ModuleParameters(P_stc=400, gamma_pmax=-0.4)

        # Power at high temperature should be lower
        T_low_idx = 0  # 15 C
        T_high_idx = -1  # 75 C (highest)
        G_idx = np.argmin(np.abs(module.G_grid - 1000))

        P_low_temp = module.P_matrix[T_low_idx, G_idx]
        P_high_temp = module.P_matrix[T_high_idx, G_idx]

        assert P_low_temp > P_high_temp


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full CSER workflow."""

    def test_full_workflow_tropical(self):
        """Test complete workflow for tropical climate."""
        # Define module
        module = ModuleParameters(
            P_stc=400,
            noct=45.0,
            gamma_pmax=-0.40,
            efficiency_stc=0.20
        )

        # Run analysis
        result = run_cser_analysis('tropical', module)

        # Verify all outputs
        assert result.cser > 0
        assert result.annual_energy_kwh > 0
        assert result.reference_energy_kwh > 0
        assert 0 < result.performance_ratio <= 1
        assert 0 < result.capacity_factor < 1
        assert result.specific_yield > 0
        assert len(result.hourly_power) == 8760
        assert len(result.hourly_module_temp) == 8760
        assert result.climate_profile == 'Tropical Hot-Humid'

    def test_iec_compliance_six_climates(self):
        """Test IEC compliance across all six standard climates."""
        module = ModuleParameters(P_stc=400)

        expected_climates = ['tropical', 'desert', 'temperate', 'cold', 'marine', 'arctic']
        results = compare_climates(module, expected_climates)

        assert len(results) == 6

        for climate in expected_climates:
            assert climate in results
            result = results[climate]

            # Basic sanity checks for IEC compliance
            assert result.annual_energy_kwh > 0
            assert 0.5 < result.cser < 1.5
            assert 0.5 < result.performance_ratio <= 1.0

    def test_different_module_technologies(self):
        """Test CSER for different module technologies."""
        # Mono-crystalline (high efficiency, moderate temp coeff)
        mono = ModuleParameters(P_stc=400, gamma_pmax=-0.35, efficiency_stc=0.21)

        # Thin-film (lower efficiency, better temp coeff)
        thin_film = ModuleParameters(P_stc=400, gamma_pmax=-0.20, efficiency_stc=0.14)

        # Run for hot climate
        result_mono = run_cser_analysis('desert', mono)
        result_tf = run_cser_analysis('desert', thin_film)

        # Both should produce positive energy
        assert result_mono.annual_energy_kwh > 0
        assert result_tf.annual_energy_kwh > 0

    def test_json_profiles_loadable(self):
        """Test that the JSON profiles file is valid and loadable."""
        profiles_path = Path(__file__).parent.parent / "data" / "climate_profiles" / "iec61853_4_standards.json"

        assert profiles_path.exists()

        with open(profiles_path) as f:
            data = json.load(f)

        assert 'profiles' in data
        assert len(data['profiles']) == 6
        assert 'metadata' in data


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_irradiance_power(self, sample_module):
        """Test power output at zero irradiance."""
        power = calculate_hourly_power(0, 25, sample_module)

        # Zero irradiance should give zero power
        np.testing.assert_almost_equal(power, 0, decimal=1)

    def test_very_cold_temperature(self, sample_module):
        """Test power at very cold temperature."""
        power = calculate_hourly_power(1000, -10, sample_module)

        # Cold temperature should increase power (for negative gamma)
        assert power > sample_module.P_stc

    def test_very_hot_temperature(self, sample_module):
        """Test power at very hot temperature."""
        power = calculate_hourly_power(1000, 70, sample_module)

        # Hot temperature should decrease power
        assert power < sample_module.P_stc
        # But should still be positive
        assert power > 0

    def test_annual_energy_with_corrections(self, tropical_climate, sample_module):
        """Test annual energy with and without corrections."""
        energy_with, _, _ = calculate_annual_energy(
            tropical_climate, sample_module,
            apply_spectral=True, apply_iam=True
        )

        energy_without, _, _ = calculate_annual_energy(
            tropical_climate, sample_module,
            apply_spectral=False, apply_iam=False
        )

        # Energy with corrections should generally be lower
        assert energy_with <= energy_without * 1.1  # Allow some tolerance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
