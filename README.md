# PV-CSER Pro 🌞⚡

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

## Overview

**PV-CSER Pro** is a comprehensive Climate Specific Energy Rating (CSER) application for photovoltaic modules based on **IEC 61853** standards (Parts 1-4). This universal tool enables indoor/outdoor energy rating analysis with both standard and user-defined climate profiles.

### Key Features

-  **IEC 61853 Compliance**: Full implementation of all four parts
- 📊 **Interactive Visualizations**: 3D plots, regression analysis, spectral response curves
- 🌍 **Climate Profiles**: Standard IEC profiles + custom climate data
- 📈 **CSER Calculations**: Accurate climate-specific energy predictions
- 📁 **Multi-Format Export**: PDF, Excel, Word, JSON reports
- 🗄️ **Database Integration**: PostgreSQL backend for data persistence
- 🚀 **Web-Based**: Accessible through Streamlit interface

## Repository Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed architecture.

## Quick Start

### Prerequisites

```bash
Python 3.9 or higher
PostgreSQL (for production deployment)
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ganeshgowri-ASA/pv-cser-pro.git
cd pv-cser-pro
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Access the app**:
Open your browser to `http://localhost:8501`

## Development Workflow

### Branch Strategy

This project uses a **phased development approach** with feature branches:

- `main` - Production-ready code
- `feature/data-input` - Phase 1: Data input forms
- `feature/calculations` - Phase 2: IEC calculation engine
- `feature/visualizations` - Phase 3: Interactive plots
- `feature/climate-cser` - Phase 4: Climate & CSER
- `feature/exports` - Phase 5: Export system
- `feature/integration` - Phase 6: Final integration

### Development Phases

#### Phase 1: Data Input Module
- Module specifications
- Power matrix (irradiance × temperature)
- Temperature coefficients
- Spectral response
- Incidence Angle Modifier (IAM)
- NMOT analysis

#### Phase 2: IEC 61853 Calculations
- **Part 1**: Power rating at STC
- **Part 2**: Spectral responsivity
- **Part 3**: Energy rating methodology
- **Part 4**: Climate profile analysis
- Temperature modeling
- Energy yield predictions

#### Phase 3: Visualizations
- 3D surface plots (P = f(G, T))
- Interactive Plotly charts
- Regression analysis
- Spectral response curves
- IAM polar diagrams

#### Phase 4: Climate Profiles & CSER
- IEC 61853-4 standard profiles
- Custom climate profile builder
- CSER calculation engine
- Comparative analysis

#### Phase 5: Export & Reporting
- Comprehensive PDF reports
- Excel data exports
- Word documentation
- JSON data export

#### Phase 6: Deployment
- Railway PostgreSQL integration
- Streamlit Cloud deployment
- Performance optimization

## Technology Stack

### Core
- **Frontend**: Streamlit 1.31+
- **Backend**: Python 3.9+
- **Database**: PostgreSQL (Railway)

### Scientific Computing
- NumPy, SciPy, Pandas
- pvlib (PV system modeling)
- scikit-learn (regression analysis)

### Visualization
- Plotly (interactive 3D/2D plots)
- Matplotlib & Seaborn

### Export
- ReportLab (PDF)
- python-docx (Word)
- openpyxl (Excel)

## IEC 61853 Standards

This application implements:

- **IEC 61853-1**: Irradiance and temperature performance measurements
- **IEC 61853-2**: Spectral responsivity, incidence angle, module temperature
- **IEC 61853-3**: Energy rating of PV modules
- **IEC 61853-4**: Standard reference climatic profiles

## Usage

### 1. Module Data Input
Enter your PV module specifications and test data according to IEC 61853-1/2 procedures.

### 2. Run Calculations
The application automatically processes data through IEC 61853-3 energy rating calculations.

### 3. Select Climate Profile
Choose from standard IEC 61853-4 profiles or create custom climate conditions.

### 4. View Results
Interactive visualizations show module performance across operating conditions.

### 5. Export Reports
Generate comprehensive reports in PDF, Excel, or Word format.

## Contributing

Contributions are welcome! Please follow the branch strategy:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- ✅ **Modular code**: Keep phases independent
- ✅ **Comprehensive tests**: Unit + integration tests
- ✅ **No breaking changes**: Preserve existing functionality
- ✅ **Documentation**: Update docs with new features
- ✅ **Code quality**: Follow PEP 8 standards

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_calculations.py

# Run with coverage
pytest --cov=src tests/
```

## Deployment

### Streamlit Cloud

1. Connect your GitHub repository to Streamlit Cloud
2. Configure environment variables
3. Deploy!

### Railway (Database)

1. Create a PostgreSQL database on Railway
2. Copy connection string
3. Add to `.streamlit/secrets.toml`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IEC 61853 standards committee
- NREL for PV research and pvlib
- Open-source community

## Contact

Project Link: [https://github.com/ganeshgowri-ASA/pv-cser-pro](https://github.com/ganeshgowri-ASA/pv-cser-pro)

## Roadmap

- [ ] Phase 1: Data Input Module
- [ ] Phase 2: IEC Calculations
- [ ] Phase 3: Visualizations
- [ ] Phase 4: Climate & CSER
- [ ] Phase 5: Export System
- [ ] Phase 6: Production Deployment
- [ ] Advanced regression models
- [ ] Machine learning predictions
- [ ] Multi-language support
- [ ] Mobile-responsive design

---

**Made with ❤️ for the PV community** | **Universal • Free • Open Source**
