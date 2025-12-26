"""
IEC 61853 Calculation Tests for PV-CSER Pro.

Tests cover:
- IEC 61853-1 Power Rating calculations
- IEC 61853-3 Energy Rating calculations
- Temperature model calculations
- CSER calculations
- Energy yield calculations
"""

import pytest
import numpy as np
from typing import Dict, Any


class TestIEC61853Part1:
    """Test IEC 61853-1 Power Rating calculations."""

    def test_power_at_stc(self, sample_module_data):
        """Test power calculation at STC."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
        )

        # At STC (1000 W/m², 25°C), power should equal Pmax
        power = calculator.calculate_power_at_stc()
        assert abs(power - sample_module_data["pmax_stc"]) < 0.01

    def test_power_at_high_temperature(self, sample_module_data):
        """Test power reduction at high temperature."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
        )

        # At 50°C, power should be reduced
        # P = Pstc * (1 + gamma * (T - 25))
        # P = 400 * (1 + (-0.35/100) * (50 - 25))
        # P = 400 * (1 - 0.0875) = 400 * 0.9125 = 365 W
        power = calculator.calculate_power(irradiance=1000, temperature=50)
        expected = 400 * (1 + (-0.35 / 100) * (50 - 25))
        assert abs(power - expected) < 1.0

    def test_power_at_low_irradiance(self, sample_module_data):
        """Test power at low irradiance."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
        )

        # At 200 W/m² and 25°C
        power = calculator.calculate_power(irradiance=200, temperature=25)
        expected = 400 * (200 / 1000)  # 80 W (approximately)
        assert abs(power - expected) < 5.0

    def test_temperature_coefficient_extraction(self, sample_power_matrix):
        """Test temperature coefficient extraction from power matrix."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(pmax_stc=400)

        # Extract from power matrix at 1000 W/m²
        irr_index = sample_power_matrix["irradiance_levels"].index(1000)
        powers_at_1000 = [row[irr_index] for row in sample_power_matrix["power_values"]]
        temps = sample_power_matrix["temperature_levels"]

        gamma = calculator.calculate_temperature_coefficients(temps, powers_at_1000)
        assert gamma is not None
        assert -0.5 < gamma < 0  # Should be negative for silicon

    def test_efficiency_calculation(self, sample_module_data):
        """Test module efficiency calculation."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(
            pmax_stc=sample_module_data["pmax_stc"],
            module_area=sample_module_data["module_area"],
        )

        efficiency = calculator.calculate_efficiency()
        # Efficiency = Pmax / (Area * 1000) * 100
        expected = (400 / (1.92 * 1000)) * 100  # ~20.8%
        assert abs(efficiency - expected) < 0.1

    def test_nmot_calculation(self):
        """Test NMOT calculation."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(pmax_stc=400, noct=45)

        # NMOT should be calculated from NOCT
        nmot = calculator.calculate_nmot()
        assert 40 <= nmot <= 50  # Typical range

    def test_fill_factor_calculation(self, sample_module_data):
        """Test fill factor calculation."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(
            pmax_stc=sample_module_data["pmax_stc"],
            voc=sample_module_data["voc_stc"],
            isc=sample_module_data["isc_stc"],
            vmp=sample_module_data["vmp_stc"],
            imp=sample_module_data["imp_stc"],
        )

        ff = calculator.calculate_fill_factor()
        expected = (40.8 * 9.8) / (48.5 * 10.5)
        assert abs(ff - expected) < 0.01

    def test_low_light_efficiency(self, sample_power_matrix):
        """Test low-light efficiency analysis."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(pmax_stc=400)

        # Get power at different irradiance levels at 25°C
        temp_index = sample_power_matrix["temperature_levels"].index(25)
        powers = sample_power_matrix["power_values"][temp_index]
        irradiances = sample_power_matrix["irradiance_levels"]

        # Calculate relative efficiency at each irradiance
        p_stc = powers[irradiances.index(1000)]
        for i, g in enumerate(irradiances):
            rel_eff = (powers[i] / p_stc) / (g / 1000)
            # Low-light efficiency should be close to or slightly above 1
            assert 0.8 <= rel_eff <= 1.1


class TestIEC61853Part3:
    """Test IEC 61853-3 Energy Rating calculations."""

    def test_cell_temperature_calculation(self, sample_hourly_climate_data):
        """Test cell temperature calculation."""
        from src.calculations.iec_61853_3 import IEC61853Part3

        calculator = IEC61853Part3(
            pmax_stc=400,
            temp_coeff_pmax=-0.35,
            nmot=43,
        )

        ghi = sample_hourly_climate_data["ghi"]
        temp = sample_hourly_climate_data["temperature"]
        wind = sample_hourly_climate_data["wind_speed"]

        # Test single hour
        t_cell = calculator.calculate_cell_temperature(
            irradiance=800,
            ambient_temp=30,
            wind_speed=2,
        )

        # Cell temperature should be higher than ambient when irradiance > 0
        assert t_cell > 30

    def test_hourly_power_calculation(self, sample_hourly_climate_data):
        """Test hourly power calculation."""
        from src.calculations.iec_61853_3 import IEC61853Part3

        calculator = IEC61853Part3(
            pmax_stc=400,
            temp_coeff_pmax=-0.35,
            nmot=43,
        )

        power = calculator.calculate_hourly_power(
            irradiance=1000,
            cell_temperature=50,
        )

        # Power at 50°C should be less than Pmax
        assert power < 400
        assert power > 0

    def test_annual_energy_calculation(self, sample_hourly_climate_data):
        """Test annual energy calculation."""
        from src.calculations.iec_61853_3 import IEC61853Part3

        calculator = IEC61853Part3(
            pmax_stc=400,
            temp_coeff_pmax=-0.35,
            nmot=43,
        )

        ghi = sample_hourly_climate_data["ghi"]
        temp = sample_hourly_climate_data["temperature"]
        wind = sample_hourly_climate_data["wind_speed"]

        energy = calculator.calculate_annual_energy(ghi, temp, wind)

        # Energy should be positive
        assert energy > 0
        # Specific yield should be reasonable (typically 800-2000 kWh/kWp)
        specific_yield = energy / 0.4  # Convert to kWh/kWp
        assert 500 < specific_yield < 2500

    def test_monthly_breakdown(self, sample_hourly_climate_data):
        """Test monthly energy breakdown."""
        from src.calculations.iec_61853_3 import IEC61853Part3

        calculator = IEC61853Part3(
            pmax_stc=400,
            temp_coeff_pmax=-0.35,
            nmot=43,
        )

        ghi = sample_hourly_climate_data["ghi"]
        temp = sample_hourly_climate_data["temperature"]
        wind = sample_hourly_climate_data["wind_speed"]

        monthly = calculator.calculate_monthly_energy(ghi, temp, wind)

        assert len(monthly) == 12
        assert all(e >= 0 for e in monthly.values())

    def test_loss_breakdown(self, sample_hourly_climate_data):
        """Test loss breakdown calculation."""
        from src.calculations.iec_61853_3 import IEC61853Part3

        calculator = IEC61853Part3(
            pmax_stc=400,
            temp_coeff_pmax=-0.35,
            nmot=43,
        )

        ghi = sample_hourly_climate_data["ghi"]
        temp = sample_hourly_climate_data["temperature"]
        wind = sample_hourly_climate_data["wind_speed"]

        losses = calculator.calculate_losses(ghi, temp, wind)

        assert "temperature" in losses
        assert "low_irradiance" in losses
        assert losses["temperature"] >= 0


class TestTemperatureModels:
    """Test temperature model calculations."""

    def test_noct_model(self):
        """Test NOCT temperature model."""
        from src.calculations.temperature_models import TemperatureModel

        model = TemperatureModel(model_type="NOCT", noct=45)

        t_cell = model.calculate(
            irradiance=800,
            ambient_temp=25,
        )

        # T_cell = T_amb + (NOCT - 20) * G / 800
        expected = 25 + (45 - 20) * 800 / 800
        assert abs(t_cell - expected) < 1

    def test_nmot_model(self):
        """Test NMOT temperature model."""
        from src.calculations.temperature_models import TemperatureModel

        model = TemperatureModel(model_type="NMOT", nmot=43)

        t_cell = model.calculate(
            irradiance=1000,
            ambient_temp=20,
            wind_speed=1,
        )

        assert t_cell > 20  # Cell temp should be higher than ambient

    def test_faiman_model(self):
        """Test Faiman temperature model."""
        from src.calculations.temperature_models import TemperatureModel

        model = TemperatureModel(model_type="Faiman")

        t_cell = model.calculate(
            irradiance=1000,
            ambient_temp=25,
            wind_speed=2,
        )

        assert t_cell > 25

    def test_pvsyst_model(self):
        """Test PVsyst temperature model."""
        from src.calculations.temperature_models import TemperatureModel

        model = TemperatureModel(model_type="PVsyst")

        t_cell = model.calculate(
            irradiance=1000,
            ambient_temp=25,
            wind_speed=2,
        )

        assert t_cell > 25

    def test_wind_effect(self):
        """Test wind cooling effect on cell temperature."""
        from src.calculations.temperature_models import TemperatureModel

        model = TemperatureModel(model_type="NMOT", nmot=43)

        # Low wind
        t_cell_low_wind = model.calculate(
            irradiance=1000,
            ambient_temp=25,
            wind_speed=0.5,
        )

        # High wind
        t_cell_high_wind = model.calculate(
            irradiance=1000,
            ambient_temp=25,
            wind_speed=10,
        )

        # Higher wind should result in lower cell temperature
        assert t_cell_high_wind < t_cell_low_wind


class TestCSERCalculator:
    """Test CSER calculation functionality."""

    def test_cser_calculation(self, sample_hourly_climate_data, sample_module_data):
        """Test full CSER calculation."""
        from src.climate.cser_calculator import CSERCalculator

        calculator = CSERCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
        )

        result = calculator.calculate_cser(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        assert result is not None
        assert result.cser > 0
        assert result.annual_energy > 0
        assert result.performance_ratio > 0

    def test_cser_comparison(self, sample_hourly_climate_data, sample_module_data, sample_module_data_2):
        """Test CSER comparison between modules."""
        from src.climate.cser_calculator import CSERCalculator

        calc1 = CSERCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
        )

        calc2 = CSERCalculator(
            pmax_stc=sample_module_data_2["pmax_stc"],
            temp_coeff_pmax=sample_module_data_2["temp_coeff_pmax"],
            nmot=sample_module_data_2["nmot"],
        )

        result1 = calc1.calculate_cser(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        result2 = calc2.calculate_cser(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        # Both should produce valid results
        assert result1.cser > 0
        assert result2.cser > 0


class TestEnergyYieldCalculator:
    """Test energy yield calculations."""

    def test_dc_energy_calculation(self, sample_hourly_climate_data, sample_module_data):
        """Test DC energy calculation."""
        from src.calculations.energy_yield import EnergyYieldCalculator

        calculator = EnergyYieldCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
        )

        dc_energy = calculator.calculate_dc_energy(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        assert dc_energy > 0

    def test_ac_energy_with_inverter(self, sample_hourly_climate_data, sample_module_data):
        """Test AC energy calculation with inverter losses."""
        from src.calculations.energy_yield import EnergyYieldCalculator

        calculator = EnergyYieldCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
            inverter_efficiency=0.96,
        )

        ac_energy = calculator.calculate_ac_energy(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        dc_energy = calculator.calculate_dc_energy(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        # AC should be less than DC due to inverter losses
        assert ac_energy < dc_energy

    def test_system_losses(self, sample_hourly_climate_data, sample_module_data):
        """Test system loss application."""
        from src.calculations.energy_yield import EnergyYieldCalculator

        # Without losses
        calc_no_loss = EnergyYieldCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
            soiling_loss=0,
            mismatch_loss=0,
            wiring_loss=0,
        )

        # With losses
        calc_with_loss = EnergyYieldCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
            soiling_loss=0.02,
            mismatch_loss=0.02,
            wiring_loss=0.015,
        )

        energy_no_loss = calc_no_loss.calculate_dc_energy(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        energy_with_loss = calc_with_loss.calculate_dc_energy(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        # Energy with losses should be less
        assert energy_with_loss < energy_no_loss

    def test_capacity_factor(self, sample_hourly_climate_data, sample_module_data):
        """Test capacity factor calculation."""
        from src.calculations.energy_yield import EnergyYieldCalculator

        calculator = EnergyYieldCalculator(
            pmax_stc=sample_module_data["pmax_stc"],
            temp_coeff_pmax=sample_module_data["temp_coeff_pmax"],
            nmot=sample_module_data["nmot"],
        )

        cf = calculator.calculate_capacity_factor(
            ghi=sample_hourly_climate_data["ghi"],
            temperature=sample_hourly_climate_data["temperature"],
            wind_speed=sample_hourly_climate_data["wind_speed"],
        )

        # Capacity factor should be between 0 and 1
        assert 0 < cf < 1
        # Typical solar CF is 10-25%
        assert 0.05 < cf < 0.35


class TestPowerMatrixInterpolation:
    """Test power matrix interpolation."""

    def test_bilinear_interpolation(self, sample_power_matrix):
        """Test bilinear interpolation on power matrix."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(pmax_stc=400)

        # Interpolate at a point between grid points
        power = calculator.interpolate_power(
            irradiance=500,  # Between 400 and 600
            temperature=35,  # Between 25 and 50
            power_matrix=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        assert power > 0

    def test_extrapolation_warning(self, sample_power_matrix):
        """Test that extrapolation beyond matrix limits is handled."""
        from src.calculations.iec_61853_1 import IEC61853Part1

        calculator = IEC61853Part1(pmax_stc=400)

        # Try to extrapolate beyond matrix limits
        power = calculator.interpolate_power(
            irradiance=1200,  # Beyond 1100
            temperature=80,   # Beyond 75
            power_matrix=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        # Should still return a value (clamped or extrapolated)
        assert power >= 0
