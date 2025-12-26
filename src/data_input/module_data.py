"""
PV Module data input and handling.

Provides classes and functions for inputting and managing
PV module specifications according to IEC 61853 standards.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModuleSpecification:
    """
    PV Module specification data class.

    Contains all relevant electrical and physical specifications
    for a PV module as defined in IEC 61853 standards.
    """

    # Identification
    manufacturer: str = ""
    model_name: str = ""
    serial_number: str = ""

    # Electrical specifications at STC (1000 W/m², 25°C, AM1.5G)
    pmax_stc: float = 0.0      # Maximum power (Wp)
    voc_stc: float = 0.0       # Open circuit voltage (V)
    isc_stc: float = 0.0       # Short circuit current (A)
    vmp_stc: float = 0.0       # Voltage at maximum power (V)
    imp_stc: float = 0.0       # Current at maximum power (A)
    ff_stc: float = 0.0        # Fill factor (%)

    # Temperature coefficients
    temp_coeff_pmax: float = -0.35    # %/°C (typical for c-Si)
    temp_coeff_voc: float = -0.30     # %/°C
    temp_coeff_isc: float = 0.05      # %/°C

    # Physical specifications
    module_area: float = 0.0          # m²
    cell_type: str = "mono-Si"        # Cell technology
    num_cells: int = 0                # Number of cells
    cell_area: float = 0.0            # Individual cell area (m²)

    # NMOT/NOCT specifications
    nmot: float = 45.0                # Nominal Module Operating Temperature (°C)
    noct: float = 45.0                # Nominal Operating Cell Temperature (°C)

    # Spectral response data (optional)
    spectral_response: Optional[Dict[str, List[float]]] = None

    # IAM data (optional)
    iam_data: Optional[Dict[str, List[float]]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived values after initialization."""
        if self.pmax_stc > 0 and self.voc_stc > 0 and self.isc_stc > 0:
            self.ff_stc = (self.pmax_stc / (self.voc_stc * self.isc_stc)) * 100

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleSpecification":
        """Create ModuleSpecification from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "serial_number": self.serial_number,
            "pmax_stc": self.pmax_stc,
            "voc_stc": self.voc_stc,
            "isc_stc": self.isc_stc,
            "vmp_stc": self.vmp_stc,
            "imp_stc": self.imp_stc,
            "ff_stc": self.ff_stc,
            "temp_coeff_pmax": self.temp_coeff_pmax,
            "temp_coeff_voc": self.temp_coeff_voc,
            "temp_coeff_isc": self.temp_coeff_isc,
            "module_area": self.module_area,
            "cell_type": self.cell_type,
            "num_cells": self.num_cells,
            "nmot": self.nmot,
            "noct": self.noct,
            "metadata": self.metadata,
        }

    def calculate_power_at_conditions(
        self,
        irradiance: float,
        temperature: float,
    ) -> float:
        """
        Calculate estimated power at given conditions using simple model.

        Args:
            irradiance: Irradiance in W/m²
            temperature: Cell temperature in °C

        Returns:
            Estimated power in W
        """
        # Linear irradiance scaling
        p_irr = self.pmax_stc * (irradiance / 1000.0)

        # Temperature correction
        temp_factor = 1 + (self.temp_coeff_pmax / 100) * (temperature - 25)

        return p_irr * temp_factor


class ModuleDataHandler:
    """
    Handler for PV module data operations.

    Provides methods for creating, validating, and managing
    module specification data.
    """

    # Standard cell types
    CELL_TYPES = [
        "mono-Si",
        "poly-Si",
        "PERC",
        "HJT",
        "TOPCon",
        "CdTe",
        "CIGS",
        "a-Si",
    ]

    # Typical temperature coefficients by cell type
    TYPICAL_TEMP_COEFFICIENTS = {
        "mono-Si": {"pmax": -0.35, "voc": -0.30, "isc": 0.05},
        "poly-Si": {"pmax": -0.40, "voc": -0.32, "isc": 0.05},
        "PERC": {"pmax": -0.34, "voc": -0.28, "isc": 0.05},
        "HJT": {"pmax": -0.26, "voc": -0.24, "isc": 0.04},
        "TOPCon": {"pmax": -0.30, "voc": -0.26, "isc": 0.04},
        "CdTe": {"pmax": -0.25, "voc": -0.22, "isc": 0.04},
        "CIGS": {"pmax": -0.36, "voc": -0.30, "isc": 0.01},
        "a-Si": {"pmax": -0.20, "voc": -0.22, "isc": 0.10},
    }

    def __init__(self):
        """Initialize handler."""
        self._modules: Dict[str, ModuleSpecification] = {}

    def create_module(self, data: Dict[str, Any]) -> ModuleSpecification:
        """
        Create a new module specification.

        Args:
            data: Dictionary with module data

        Returns:
            ModuleSpecification instance
        """
        module = ModuleSpecification.from_dict(data)

        # Store module
        key = f"{module.manufacturer}_{module.model_name}"
        self._modules[key] = module

        return module

    def get_typical_coefficients(self, cell_type: str) -> Dict[str, float]:
        """
        Get typical temperature coefficients for a cell type.

        Args:
            cell_type: Type of solar cell

        Returns:
            Dictionary with typical temperature coefficients
        """
        return self.TYPICAL_TEMP_COEFFICIENTS.get(
            cell_type,
            {"pmax": -0.35, "voc": -0.30, "isc": 0.05},
        )

    def validate_module(self, module: ModuleSpecification) -> List[str]:
        """
        Validate module specification data.

        Args:
            module: ModuleSpecification to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Required fields
        if not module.manufacturer:
            errors.append("Manufacturer is required")
        if not module.model_name:
            errors.append("Model name is required")
        if module.pmax_stc <= 0:
            errors.append("Pmax at STC must be positive")

        # Electrical validation
        if module.voc_stc > 0 and module.vmp_stc > 0:
            if module.vmp_stc >= module.voc_stc:
                errors.append("Vmp must be less than Voc")

        if module.isc_stc > 0 and module.imp_stc > 0:
            if module.imp_stc >= module.isc_stc:
                errors.append("Imp must be less than Isc")

        # Check power consistency
        if module.vmp_stc > 0 and module.imp_stc > 0:
            calculated_pmax = module.vmp_stc * module.imp_stc
            if abs(calculated_pmax - module.pmax_stc) / module.pmax_stc > 0.05:
                errors.append(
                    f"Pmax ({module.pmax_stc}W) differs from Vmp*Imp "
                    f"({calculated_pmax:.1f}W) by more than 5%"
                )

        # Fill factor check
        if module.ff_stc > 0:
            if module.ff_stc < 50 or module.ff_stc > 90:
                errors.append(
                    f"Fill factor ({module.ff_stc:.1f}%) is outside "
                    "typical range (50-90%)"
                )

        # Temperature coefficient checks
        if module.temp_coeff_pmax > 0:
            errors.append(
                "Temperature coefficient for Pmax should be negative"
            )

        # Area check
        if module.module_area > 0 and module.pmax_stc > 0:
            efficiency = (module.pmax_stc / module.module_area) / 10  # %
            if efficiency < 5 or efficiency > 30:
                errors.append(
                    f"Calculated efficiency ({efficiency:.1f}%) is outside "
                    "typical range (5-30%)"
                )

        return errors

    @staticmethod
    def create_sample_module() -> ModuleSpecification:
        """Create a sample module for demonstration."""
        return ModuleSpecification(
            manufacturer="Sample Solar",
            model_name="PV-400M",
            pmax_stc=400.0,
            voc_stc=48.5,
            isc_stc=10.5,
            vmp_stc=40.8,
            imp_stc=9.8,
            temp_coeff_pmax=-0.35,
            temp_coeff_voc=-0.28,
            temp_coeff_isc=0.05,
            module_area=1.92,
            cell_type="mono-Si",
            num_cells=72,
            nmot=43.0,
        )
