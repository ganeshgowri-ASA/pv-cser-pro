"""
IEC 61853-1: Power rating at Standard Test Conditions.

Implements calculations for PV module power rating according to
IEC 61853-1 standard for irradiance and temperature performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, optimize


@dataclass
class PowerRatingResult:
    """Results from IEC 61853-1 power rating calculations."""

    # STC ratings
    pmax_stc: float           # W
    voc_stc: float            # V
    isc_stc: float            # A
    vmp_stc: float            # V
    imp_stc: float            # A
    ff_stc: float             # %

    # Temperature coefficients
    gamma_pmax: float         # %/°C
    gamma_voc: float          # %/°C or mV/°C
    gamma_isc: float          # %/°C or mA/°C

    # NMOT
    nmot: Optional[float] = None  # °C

    # Additional metrics
    efficiency_stc: Optional[float] = None  # %
    relative_efficiency: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pmax_stc": self.pmax_stc,
            "voc_stc": self.voc_stc,
            "isc_stc": self.isc_stc,
            "vmp_stc": self.vmp_stc,
            "imp_stc": self.imp_stc,
            "ff_stc": self.ff_stc,
            "gamma_pmax": self.gamma_pmax,
            "gamma_voc": self.gamma_voc,
            "gamma_isc": self.gamma_isc,
            "nmot": self.nmot,
            "efficiency_stc": self.efficiency_stc,
        }


class IEC61853Part1:
    """
    IEC 61853-1 Power Rating Calculator.

    Implements power rating calculations including:
    - Power matrix analysis
    - Temperature coefficient extraction
    - NMOT determination
    - Efficiency calculations
    """

    # Standard test conditions
    G_STC = 1000.0    # W/m²
    T_STC = 25.0      # °C
    AM_STC = 1.5      # Air mass

    # NMOT test conditions
    G_NMOT = 800.0    # W/m²
    T_AMB_NMOT = 20.0 # °C
    V_WIND_NMOT = 1.0 # m/s

    def __init__(
        self,
        irradiance_levels: np.ndarray,
        temperature_levels: np.ndarray,
        power_matrix: np.ndarray,
        module_area: Optional[float] = None,
    ):
        """
        Initialize IEC 61853-1 calculator.

        Args:
            irradiance_levels: Array of irradiance values (W/m²)
            temperature_levels: Array of temperature values (°C)
            power_matrix: 2D array of power values (W)
            module_area: Module area in m² (for efficiency calculation)
        """
        self.irradiance = np.array(irradiance_levels)
        self.temperature = np.array(temperature_levels)
        self.power_matrix = np.array(power_matrix)
        self.module_area = module_area

        # Create interpolation function
        self._interp_func = interpolate.RegularGridInterpolator(
            (self.irradiance, self.temperature),
            self.power_matrix,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

    def get_power(
        self,
        irradiance: Union[float, np.ndarray],
        temperature: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Get interpolated power at given conditions.

        Args:
            irradiance: Irradiance in W/m²
            temperature: Cell temperature in °C

        Returns:
            Power in W
        """
        scalar = np.isscalar(irradiance) and np.isscalar(temperature)
        irradiance = np.atleast_1d(irradiance)
        temperature = np.atleast_1d(temperature)

        points = np.column_stack([irradiance, temperature])
        result = self._interp_func(points)

        if scalar:
            return float(result[0])
        return result

    def calculate_power_at_stc(self) -> float:
        """Calculate power at STC conditions."""
        return self.get_power(self.G_STC, self.T_STC)

    def calculate_temperature_coefficients(
        self,
        irradiance: float = 1000.0,
    ) -> Dict[str, float]:
        """
        Calculate temperature coefficient from power matrix.

        Args:
            irradiance: Irradiance level for calculation

        Returns:
            Dictionary with temperature coefficients
        """
        # Find closest irradiance level
        idx = np.argmin(np.abs(self.irradiance - irradiance))
        power_vs_temp = self.power_matrix[idx, :]

        # Remove NaN values
        valid = ~np.isnan(power_vs_temp)
        if np.sum(valid) < 2:
            return {"gamma_pmax": np.nan, "gamma_pmax_rel": np.nan}

        temps_valid = self.temperature[valid]
        power_valid = power_vs_temp[valid]

        # Linear fit: P = P0 + gamma * (T - T_ref)
        coeffs = np.polyfit(temps_valid, power_valid, 1)
        gamma_pmax_abs = coeffs[0]  # W/°C

        # Get reference power at 25°C
        p_ref = self.get_power(irradiance, 25.0)

        # Relative coefficient (%/°C)
        gamma_pmax_rel = (gamma_pmax_abs / p_ref) * 100 if p_ref > 0 else np.nan

        return {
            "gamma_pmax": gamma_pmax_abs,      # W/°C
            "gamma_pmax_rel": gamma_pmax_rel,  # %/°C
        }

    def calculate_irradiance_response(
        self,
        temperature: float = 25.0,
    ) -> Dict[str, float]:
        """
        Calculate irradiance response characteristics.

        Args:
            temperature: Cell temperature for calculation

        Returns:
            Dictionary with irradiance response metrics
        """
        # Find closest temperature level
        idx = np.argmin(np.abs(self.temperature - temperature))
        power_vs_irr = self.power_matrix[:, idx]

        valid = ~np.isnan(power_vs_irr)
        if np.sum(valid) < 3:
            return {"linearity": np.nan}

        irr_valid = self.irradiance[valid]
        power_valid = power_vs_irr[valid]

        # Calculate linearity (ideal = linear with irradiance)
        p_stc = self.get_power(1000, temperature)
        ideal = p_stc * (irr_valid / 1000)

        # RMSE as percentage of STC power
        rmse = np.sqrt(np.mean((power_valid - ideal) ** 2)) / p_stc * 100

        # Low-light efficiency ratio
        p_200 = self.get_power(200, temperature)
        low_light_ratio = (p_200 / p_stc) / (200 / 1000) if p_stc > 0 else np.nan

        return {
            "linearity_rmse": rmse,
            "low_light_ratio": low_light_ratio,
        }

    def calculate_nmot(
        self,
        u_coeff: float = 25.0,
        v_coeff: float = 6.84,
    ) -> float:
        """
        Calculate Nominal Module Operating Temperature (NMOT).

        Uses thermal model: T_cell = T_amb + G * (NMOT - 20) / 800

        Args:
            u_coeff: Heat loss coefficient constant term
            v_coeff: Heat loss coefficient wind term

        Returns:
            NMOT in °C
        """
        # NMOT definition conditions
        g_nmot = 800.0   # W/m²
        t_amb = 20.0     # °C
        wind = 1.0       # m/s

        # Simple thermal model
        # NMOT = T_cell at (G=800, T_amb=20, wind=1)
        # T_cell = T_amb + G / (u_coeff + v_coeff * wind)

        heat_loss = u_coeff + v_coeff * wind
        delta_t = g_nmot / heat_loss
        nmot = t_amb + delta_t

        return nmot

    def calculate_efficiency(
        self,
        irradiance: float = 1000.0,
        temperature: float = 25.0,
    ) -> Optional[float]:
        """
        Calculate module efficiency.

        Args:
            irradiance: Irradiance in W/m²
            temperature: Cell temperature in °C

        Returns:
            Efficiency in % or None if area not available
        """
        if self.module_area is None or self.module_area <= 0:
            return None

        power = self.get_power(irradiance, temperature)
        incident_power = irradiance * self.module_area

        if incident_power <= 0:
            return None

        return (power / incident_power) * 100

    def calculate_relative_efficiency(self) -> Dict[str, float]:
        """
        Calculate relative efficiency at different conditions.

        Returns:
            Dictionary of relative efficiencies at different (G, T) points
        """
        p_stc = self.calculate_power_at_stc()
        if p_stc <= 0:
            return {}

        result = {}

        # Standard reference points
        reference_points = [
            (200, 25, "G200_T25"),
            (400, 25, "G400_T25"),
            (600, 25, "G600_T25"),
            (800, 25, "G800_T25"),
            (1000, 50, "G1000_T50"),
            (1000, 75, "G1000_T75"),
        ]

        for g, t, label in reference_points:
            p = self.get_power(g, t)
            if not np.isnan(p):
                # Relative efficiency = P/(P_stc * G/G_stc)
                ideal = p_stc * (g / 1000)
                result[label] = (p / ideal) * 100 if ideal > 0 else np.nan

        return result

    def generate_power_rating(self) -> PowerRatingResult:
        """
        Generate complete power rating according to IEC 61853-1.

        Returns:
            PowerRatingResult with all calculated values
        """
        # Calculate STC power
        pmax_stc = self.calculate_power_at_stc()

        # Calculate temperature coefficients
        temp_coeffs = self.calculate_temperature_coefficients(1000.0)
        gamma_pmax = temp_coeffs.get("gamma_pmax_rel", -0.35)

        # Calculate efficiency
        efficiency = self.calculate_efficiency(1000.0, 25.0)

        # Calculate relative efficiency
        rel_eff = self.calculate_relative_efficiency()

        # Calculate NMOT
        nmot = self.calculate_nmot()

        # Estimate other parameters from typical ratios
        # These would normally come from I-V curve data
        vmp_voc_ratio = 0.84
        imp_isc_ratio = 0.93

        voc_stc = (pmax_stc / (0.79 * 10)) if pmax_stc > 0 else 0  # Rough estimate
        isc_stc = pmax_stc / (voc_stc * 0.79) if voc_stc > 0 else 0
        vmp_stc = voc_stc * vmp_voc_ratio
        imp_stc = isc_stc * imp_isc_ratio

        ff_stc = (pmax_stc / (voc_stc * isc_stc) * 100) if (voc_stc * isc_stc) > 0 else 0

        return PowerRatingResult(
            pmax_stc=pmax_stc,
            voc_stc=voc_stc,
            isc_stc=isc_stc,
            vmp_stc=vmp_stc,
            imp_stc=imp_stc,
            ff_stc=ff_stc,
            gamma_pmax=gamma_pmax,
            gamma_voc=-0.30,  # Typical value
            gamma_isc=0.05,   # Typical value
            nmot=nmot,
            efficiency_stc=efficiency,
            relative_efficiency=rel_eff,
        )

    def create_power_model(
        self,
        method: str = "bilinear",
    ) -> callable:
        """
        Create a power model function based on the matrix data.

        Args:
            method: Modeling method ('bilinear', 'polynomial')

        Returns:
            Callable power model function(G, T) -> P
        """
        if method == "bilinear":
            # Bilinear model: P = P_stc * (G/G_stc) * (1 + gamma*(T-T_stc))
            p_stc = self.calculate_power_at_stc()
            coeffs = self.calculate_temperature_coefficients()
            gamma = coeffs.get("gamma_pmax_rel", -0.35) / 100

            def model(g: float, t: float) -> float:
                return p_stc * (g / 1000.0) * (1 + gamma * (t - 25.0))

            return model

        elif method == "polynomial":
            # Fit polynomial model to data
            g_mesh, t_mesh = np.meshgrid(self.irradiance, self.temperature, indexing='ij')
            g_flat = g_mesh.flatten()
            t_flat = t_mesh.flatten()
            p_flat = self.power_matrix.flatten()

            # Remove NaN values
            valid = ~np.isnan(p_flat)
            g_valid = g_flat[valid]
            t_valid = t_flat[valid]
            p_valid = p_flat[valid]

            # Normalize
            g_norm = g_valid / 1000.0
            t_norm = (t_valid - 25.0) / 50.0

            # Fit 2nd order polynomial
            # P = a0 + a1*G + a2*T + a3*G*T + a4*G² + a5*T²
            A = np.column_stack([
                np.ones_like(g_norm),
                g_norm,
                t_norm,
                g_norm * t_norm,
                g_norm ** 2,
                t_norm ** 2,
            ])

            coeffs, _, _, _ = np.linalg.lstsq(A, p_valid, rcond=None)

            def model(g: float, t: float) -> float:
                g_n = g / 1000.0
                t_n = (t - 25.0) / 50.0
                return (
                    coeffs[0] +
                    coeffs[1] * g_n +
                    coeffs[2] * t_n +
                    coeffs[3] * g_n * t_n +
                    coeffs[4] * g_n ** 2 +
                    coeffs[5] * t_n ** 2
                )

            return model

        else:
            raise ValueError(f"Unknown method: {method}")
