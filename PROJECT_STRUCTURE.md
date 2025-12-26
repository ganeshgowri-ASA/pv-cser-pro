# PV-CSER Pro - Project Structure

## Overview
Climate Specific Energy Rating (CSER) application for PV modules based on IEC 61853 standards.

## Repository Structure

```
pv-cser-pro/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
├── PROJECT_STRUCTURE.md           # This file
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_input/                # Phase 1: Data Input Module
│   │   ├── __init__.py
│   │   ├── module_data.py         # Module specifications input
│   │   ├── power_matrix.py        # Power matrix input
│   │   ├── temp_coefficients.py   # Temperature coefficients
│   │   ├── spectral_response.py   # Spectral response data
│   │   ├── iam.py                 # Incidence Angle Modifier
│   │   └── validation.py          # Data validation functions
│   │
│   ├── calculations/              # Phase 2: IEC Calculations
│   │   ├── __init__.py
│   │   ├── iec_61853_1.py         # Part 1: Power rating at STC
│   │   ├── iec_61853_2.py         # Part 2: Spectral responsivity
│   │   ├── iec_61853_3.py         # Part 3: Energy rating
│   │   ├── iec_61853_4.py         # Part 4: Climate profiles
│   │   ├── power_calculations.py  # Power calculations
│   │   ├── temperature_models.py  # Temperature modeling
│   │   └── energy_yield.py        # Energy yield calculations
│   │
│   ├── visualizations/            # Phase 3: Visualizations
│   │   ├── __init__.py
│   │   ├── plots_3d.py            # 3D surface plots
│   │   ├── charts.py              # 2D charts and graphs
│   │   ├── regression.py          # Regression analysis plots
│   │   └── interactive.py         # Interactive plotly charts
│   │
│   ├── climate/                   # Phase 4: Climate & CSER
│   │   ├── __init__.py
│   │   ├── climate_profiles.py    # Standard climate profiles
│   │   ├── custom_profile.py      # User-defined profiles
│   │   ├── cser_calculator.py     # CSER calculation engine
│   │   └── climate_database.py    # Climate data management
│   │
│   ├── exports/                   # Phase 5: Export System
│   │   ├── __init__.py
│   │   ├── pdf_export.py          # PDF report generation
│   │   ├── excel_export.py        # Excel export
│   │   ├── word_export.py         # Word document export
│   │   └── json_export.py         # JSON data export
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── database.py            # Database operations
│       ├── config.py              # Configuration management
│       └── helpers.py             # Helper functions
│
├── data/                          # Data files
│   ├── climate_profiles/          # Standard climate data
│   ├── sample_modules/            # Sample module data
│   └── schemas/                   # Data schemas
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_data_input.py
│   ├── test_calculations.py
│   ├── test_visualizations.py
│   ├── test_climate.py
│   └── test_exports.py
│
├── docs/                          # Documentation
│   ├── API.md                     # API documentation
│   ├── USER_GUIDE.md             # User guide
│   └── DEVELOPMENT.md            # Development guide
│
└── config/                        # Configuration files
    ├── app_config.yaml           # Application configuration
    └── database_config.yaml      # Database configuration
```

## Development Phases

### Phase 1: Data Input Module (Branch: feature/data-input)
- Module specifications form
- Power matrix input with validation
- Temperature coefficients
- Spectral response curves
- IAM data input
- NMOT analysis

### Phase 2: IEC Calculations Engine (Branch: feature/calculations)
- IEC 61853-1: Power rating calculations
- IEC 61853-2: Spectral responsivity
- IEC 61853-3: Energy rating methodology
- IEC 61853-4: Climate profile analysis
- Temperature modeling
- Energy yield predictions

### Phase 3: Visualization Components (Branch: feature/visualizations)
- 3D surface plots (Power vs Irradiance vs Temperature)
- Interactive charts with Plotly
- Regression analysis visualizations
- Spectral response curves
- IAM polar plots
- Temperature coefficient plots

### Phase 4: Climate Profiles & CSER (Branch: feature/climate-cser)
- Standard IEC 61853-4 climate profiles
- Custom climate profile builder
- CSER calculation engine
- Climate-specific energy predictions
- Comparative analysis tools

### Phase 5: Export & Reporting (Branch: feature/exports)
- PDF comprehensive reports
- Excel data exports
- Word document generation
- JSON data export
- Batch export capabilities

### Phase 6: Integration & Deployment (Branch: feature/integration)
- Railway database integration
- Streamlit deployment configuration
- Performance optimization
- Security enhancements
- Final QA testing

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **Database**: PostgreSQL (Railway)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Scientific**: NumPy, SciPy, Pandas, pvlib
- **Export**: ReportLab, python-docx, openpyxl

## Branch Strategy

- `main`: Production-ready code
- `feature/data-input`: Phase 1 development
- `feature/calculations`: Phase 2 development
- `feature/visualizations`: Phase 3 development
- `feature/climate-cser`: Phase 4 development
- `feature/exports`: Phase 5 development
- `feature/integration`: Phase 6 development

## Testing Strategy

- Unit tests for each module
- Integration tests for workflows
- QA testing session by session
- Branch-by-branch validation
- No code breaks during merges

## Deployment

1. Local development with Streamlit
2. Railway PostgreSQL database
3. Streamlit Cloud deployment
4. Continuous integration/deployment

## Key Principles

- **Modular**: Each phase is independent
- **Scalable**: Easy to add features
- **Tested**: Comprehensive QA
- **No breaks**: Preserve working features
- **Universal**: Accessible to all users
