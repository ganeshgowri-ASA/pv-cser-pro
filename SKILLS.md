# PV-CSER Pro - Project Wisdom & Skills Document

## Overview

PV-CSER Pro is a Climate Specific Energy Rating (CSER) system compliant with IEC 61853 standards for photovoltaic module performance characterization.

---

## Core Principles

### 1. Design-First Approach

- **Think before coding**: Draft the solution architecture before implementation
- **Interface design**: Define function signatures and data contracts first
- **Separation of concerns**: Keep calculations, visualization, and UI logic separate
- **Documentation-driven**: Write docstrings before function bodies

### 2. Lean Six Sigma Methodology

- **Define**: Clearly specify inputs, outputs, and acceptance criteria
- **Measure**: Implement logging and metrics for all calculations
- **Analyze**: Validate results against IEC 61853 reference data
- **Improve**: Optimize performance using caching and vectorization
- **Control**: Maintain test coverage above 80%

### 3. IEC 61853 Compliance

- **Traceability**: Every calculation must reference specific IEC clause
- **Precision**: Use appropriate significant figures per standard
- **Validation**: Input ranges must match IEC test conditions
- **Uncertainty**: Consider measurement uncertainty in all calculations

---

## Common Mistakes to Avoid

### Calculation Errors

| Mistake | Correct Approach |
|---------|-----------------|
| Linear interpolation for Voc vs G | Use logarithmic: Voc = v1 + v2*ln(G) |
| Ignoring temperature coefficients below 25°C | Apply coefficients at all temperatures |
| Using power coefficients instead of current/voltage | Calculate α (Isc), β (Voc), γ (Pmax) separately |
| Extrapolating beyond measured range | Return NaN or raise warning |

### Data Handling

| Mistake | Correct Approach |
|---------|-----------------|
| Missing validation for negative irradiance | Validate G ≥ 0 W/m² |
| Not handling missing matrix cells | Use interpolation or mark as invalid |
| Hardcoding STC values | Read from measured data at G=1000, T=25°C |
| Ignoring spectral effects | Apply AM correction per IEC 61853-3 |

### UI/UX Issues

| Mistake | Correct Approach |
|---------|-----------------|
| No input validation feedback | Show real-time validation errors |
| Blocking UI during calculations | Use st.spinner and caching |
| No export capability | Provide CSV, Excel, and PDF exports |
| Missing units in displays | Always show units (W, W/m², °C) |

---

## IEC 61853 Quick Reference

### Part 1: Irradiance and Temperature Performance

**Scope**: Procedures for measuring PV module performance at various irradiance and temperature conditions.

**Key Test Conditions**:
| Condition | Abbreviation | Irradiance (W/m²) | Temperature (°C) |
|-----------|-------------|-------------------|------------------|
| Standard Test Conditions | STC | 1000 | 25 |
| Nominal Operating Cell Temperature | NOCT | 800 | 20 (ambient) |
| Low Irradiance Conditions | LIC | 200 | 25 |
| High Temperature Conditions | HTC | 1000 | 75 |
| Low Temperature Conditions | LTC | 500 | 15 |

**Power Matrix Requirements**:
- Minimum 22 measurement points (7 irradiances × 4 temperatures recommended)
- Irradiance levels: 100, 200, 400, 600, 800, 1000, 1100 W/m²
- Temperature levels: 15, 25, 50, 75 °C
- Each point requires Isc, Voc, Pmax, Vmpp, Impp, FF measurements

### Part 2: Spectral Responsivity

**Scope**: Measurement of spectral responsivity and calculation of spectral mismatch.

**Key Equations**:
```
SR(λ) = Isc(λ) / E(λ)  # Spectral responsivity
MMF = ∫SR(λ)·E_ref(λ)dλ / ∫SR(λ)·E_test(λ)dλ  # Mismatch factor
```

**Wavelength Range**: 300-1200 nm (crystalline Si), extended for thin-film

### Part 3: Energy Rating

**Scope**: Calculation of PV module energy rating for different climates.

**Climate Profiles** (per IEC 61853-4):
| Profile | Location Type | Annual GHI (kWh/m²) |
|---------|--------------|---------------------|
| Subtropical Coastal | Hot, humid | 1700-1900 |
| Subtropical Arid | Hot, dry | 2000-2200 |
| Temperate Coastal | Moderate, humid | 1000-1200 |
| Temperate Continental | Moderate, dry | 1200-1400 |
| High Elevation | Cold, high UV | 1800-2000 |
| Tropical | Hot, humid, stable | 1600-1800 |

**CSER Calculation**:
```
CSER = Σ(P(Gi, Ti) × Δt_i) / P_STC  # Climate Specific Energy Rating
```

### Part 4: Standard Reference Climatic Profiles

**Scope**: Definition of reference climatic profiles for energy rating calculations.

**Required Data** (hourly for typical meteorological year):
- Global horizontal irradiance (GHI)
- Diffuse horizontal irradiance (DHI)
- Ambient temperature
- Wind speed
- Air mass (derived)
- Angle of incidence

---

## Testing Checklist

### Unit Tests

- [ ] Temperature coefficient calculations (α, β, γ)
- [ ] Interpolation functions (linear, logarithmic, polynomial)
- [ ] Power matrix validation
- [ ] Climate profile loading
- [ ] CSER rating calculations
- [ ] Edge cases (zero irradiance, extreme temperatures)

### Integration Tests

- [ ] Full calculation pipeline from input to CSER
- [ ] Data export functionality
- [ ] Session state persistence
- [ ] Multi-page navigation

### Validation Tests

- [ ] Compare results against IEC 61853 reference modules
- [ ] Verify temperature coefficient ranges (typical values):
  - α (Isc): +0.03 to +0.06 %/°C
  - β (Voc): -0.25 to -0.35 %/°C
  - γ (Pmax): -0.35 to -0.50 %/°C
- [ ] Check power matrix interpolation accuracy (< 1% error)
- [ ] Validate CSER within expected range for climate profiles

### UI/UX Tests

- [ ] Input validation feedback
- [ ] Responsive layout
- [ ] Loading indicators
- [ ] Error handling and user messages
- [ ] Export file generation

---

## Code Standards

### Type Hints

```python
def calculate_power(
    irradiance: float,
    temperature: float,
    power_matrix: np.ndarray
) -> float:
    """Calculate power at given conditions."""
    ...
```

### Docstrings

```python
def interpolate_voc(irradiance: float, voc_ref: float, g_ref: float = 1000.0) -> float:
    """
    Interpolate open-circuit voltage using logarithmic relationship.

    Per IEC 61853-1 Clause 7.3, Voc varies logarithmically with irradiance:
    Voc(G) = Voc_ref + n*k*T/q * ln(G/G_ref)

    Args:
        irradiance: Target irradiance in W/m²
        voc_ref: Reference Voc at G_ref in Volts
        g_ref: Reference irradiance in W/m² (default: 1000)

    Returns:
        Interpolated Voc in Volts

    Raises:
        ValueError: If irradiance <= 0

    Reference:
        IEC 61853-1:2011 Clause 7.3
    """
    ...
```

### Caching

```python
@st.cache_data
def load_climate_profile(profile_name: str) -> pd.DataFrame:
    """Load and cache climate profile data."""
    ...
```

---

## Architecture

```
pv-cser-pro/
├── app.py                    # Main Streamlit application
├── SKILLS.md                 # This document
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── src/
    ├── __init__.py
    ├── utils/
    │   ├── __init__.py
    │   └── constants.py     # IEC 61853 constants
    ├── calculations/
    │   ├── __init__.py
    │   └── iec61853_1.py    # Core calculations
    └── visualizations/
        ├── __init__.py
        └── plots_3d.py      # 3D surface plots
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Run tests
pytest tests/ -v --cov=src
```

---

## References

1. IEC 61853-1:2011 - Irradiance and temperature performance measurements
2. IEC 61853-2:2016 - Spectral responsivity, incidence angle, and module operating temperature
3. IEC 61853-3:2018 - Energy rating of PV modules
4. IEC 61853-4:2018 - Standard reference climatic profiles

---

*Document Version: 1.0.0 | Last Updated: 2024*
