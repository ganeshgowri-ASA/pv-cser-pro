"""
Temperature models for PV module cell temperature calculation.

Implements various thermal models for estimating cell/module temperature
from ambient conditions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np


class ThermalModelType(Enum):
    """Available thermal model types."""

    NOCT = "noct"
    NMOT = "nmot"
    FAIMAN = "faiman"
    PVSYST = "pvsyst"
    SANDIA = "sandia"
    ROSS = "ross"


@dataclass
class ThermalModelParams:
    """Parameters for thermal models."""

    # NOCT/NMOT parameters
    noct: float = 45.0          # °C
    nmot: float = 45.0          # °C

    # Faiman model parameters
    u0: float = 25.0            # W/(m²·K)
    u1: float = 6.84            # W/(m²·K)/(m/s)

    # Sandia model parameters
    a: float = -3.47            # °C
    b: float = -0.0594          # °C·s/m

    # PVsyst model parameters
    alpha: float = 0.9          # Absorption coefficient
    uc: float = 29.0            # Constant loss factor (W/m²·K)
    uv: float = 0.0             # Wind loss factor (W/m²·K)/(m/s)


class TemperatureModel:
    """
    Cell temperature calculation models.

    Provides multiple thermal models for calculating PV cell temperature
    from ambient conditions.
    """

    # Reference conditions
    G_NOCT = 800.0      # W/m² - NOCT irradiance
    T_AMB_NOCT = 20.0   # °C - NOCT ambient temperature
    V_WIND_NOCT = 1.0   # m/s - NOCT wind speed

    def __init__(
        self,
        model_type: Union[ThermalModelType, str] = ThermalModelType.NMOT,
        params: Optional[ThermalModelParams] = None,
    ):
        """
        Initialize temperature model.

        Args:
            model_type: Type of thermal model to use
            params: Model parameters (uses defaults if None)
        """
        if isinstance(model_type, str):
            model_type = ThermalModelType(model_type.lower())

        self.model_type = model_type
        self.params = params or ThermalModelParams()

    def calculate(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
        wind_speed: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate cell temperature.

        Args:
            ambient_temp: Ambient temperature (°C)
            irradiance: Global irradiance on module plane (W/m²)
            wind_speed: Wind speed (m/s), defaults to 1 m/s

        Returns:
            Cell temperature (°C)
        """
        if wind_speed is None:
            if isinstance(irradiance, np.ndarray):
                wind_speed = np.ones_like(irradiance)
            else:
                wind_speed = 1.0

        if self.model_type == ThermalModelType.NOCT:
            return self._noct_model(ambient_temp, irradiance)

        elif self.model_type == ThermalModelType.NMOT:
            return self._nmot_model(ambient_temp, irradiance, wind_speed)

        elif self.model_type == ThermalModelType.FAIMAN:
            return self._faiman_model(ambient_temp, irradiance, wind_speed)

        elif self.model_type == ThermalModelType.PVSYST:
            return self._pvsyst_model(ambient_temp, irradiance, wind_speed)

        elif self.model_type == ThermalModelType.SANDIA:
            return self._sandia_model(ambient_temp, irradiance, wind_speed)

        elif self.model_type == ThermalModelType.ROSS:
            return self._ross_model(ambient_temp, irradiance)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _noct_model(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Simple NOCT-based model.

        T_cell = T_amb + (NOCT - 20) * G / 800
        """
        noct = self.params.noct
        delta_t = (noct - self.T_AMB_NOCT) * (irradiance / self.G_NOCT)
        return ambient_temp + delta_t

    def _nmot_model(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        NMOT model with wind correction.

        T_cell = T_amb + G * (NMOT - 20) / 800 * (9.5 / (5.7 + 3.8 * v))
        """
        nmot = self.params.nmot

        # Wind correction factor
        wind_factor = 9.5 / (5.7 + 3.8 * wind_speed)

        delta_t = irradiance * ((nmot - 20.0) / 800.0) * wind_factor

        return ambient_temp + delta_t

    def _faiman_model(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Faiman thermal model.

        T_cell = T_amb + G / (U0 + U1 * v)

        where U0 and U1 are heat transfer coefficients.
        """
        u0 = self.params.u0
        u1 = self.params.u1

        heat_loss = u0 + u1 * wind_speed

        # Prevent division by zero
        if isinstance(heat_loss, np.ndarray):
            heat_loss = np.maximum(heat_loss, 0.1)
        else:
            heat_loss = max(heat_loss, 0.1)

        delta_t = irradiance / heat_loss

        return ambient_temp + delta_t

    def _pvsyst_model(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        PVsyst thermal model.

        T_cell = T_amb + alpha * G / (Uc + Uv * v)
        """
        alpha = self.params.alpha
        uc = self.params.uc
        uv = self.params.uv

        heat_loss = uc + uv * wind_speed

        if isinstance(heat_loss, np.ndarray):
            heat_loss = np.maximum(heat_loss, 0.1)
        else:
            heat_loss = max(heat_loss, 0.1)

        delta_t = alpha * irradiance / heat_loss

        return ambient_temp + delta_t

    def _sandia_model(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Sandia back-of-module temperature model.

        T_back = G * exp(a + b * v) + T_amb
        T_cell = T_back + G/1000 * delta_T

        where delta_T is typically 3°C for glass/cell/glass modules.
        """
        a = self.params.a
        b = self.params.b
        delta_t_cell = 3.0  # Glass/cell/glass typical value

        t_back = irradiance * np.exp(a + b * wind_speed) + ambient_temp
        t_cell = t_back + (irradiance / 1000.0) * delta_t_cell

        return t_cell

    def _ross_model(
        self,
        ambient_temp: Union[float, np.ndarray],
        irradiance: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Ross model (simplified linear model).

        T_cell = T_amb + k * G

        where k is the Ross coefficient (typically 0.02-0.04 °C·m²/W)
        """
        # Calculate k from NOCT
        noct = self.params.noct
        k = (noct - 20.0) / 800.0

        return ambient_temp + k * irradiance

    @classmethod
    def from_noct(cls, noct: float) -> "TemperatureModel":
        """
        Create model from NOCT value.

        Args:
            noct: Nominal Operating Cell Temperature (°C)

        Returns:
            TemperatureModel instance
        """
        params = ThermalModelParams(noct=noct, nmot=noct)
        return cls(ThermalModelType.NMOT, params)

    @classmethod
    def from_faiman_coefficients(
        cls,
        u0: float,
        u1: float,
    ) -> "TemperatureModel":
        """
        Create Faiman model from coefficients.

        Args:
            u0: Constant heat transfer coefficient (W/m²·K)
            u1: Wind-dependent coefficient (W/m²·K)/(m/s)

        Returns:
            TemperatureModel instance
        """
        params = ThermalModelParams(u0=u0, u1=u1)
        return cls(ThermalModelType.FAIMAN, params)

    def estimate_faiman_coefficients_from_noct(
        self,
        noct: float,
    ) -> tuple:
        """
        Estimate Faiman coefficients from NOCT.

        Args:
            noct: NOCT value (°C)

        Returns:
            Tuple of (u0, u1)
        """
        # At NOCT conditions: T_cell = 20 + 800 / (u0 + u1)
        # Therefore: u0 + u1 = 800 / (NOCT - 20)

        delta_t = noct - 20.0
        total_u = 800.0 / delta_t if delta_t > 0 else 30.0

        # Typical ratio u0/u1 is about 4:1
        u1 = total_u / 5.0
        u0 = total_u - u1

        return u0, u1
