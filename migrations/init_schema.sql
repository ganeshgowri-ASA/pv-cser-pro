-- PV-CSER Pro Database Schema
-- PostgreSQL initialization script for Railway
-- Based on IEC 61853 standards for PV module energy rating

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Users Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    organization VARCHAR(255),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

-- =============================================================================
-- PV Modules Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS pv_modules (
    id SERIAL PRIMARY KEY,

    -- Module identification
    module_name VARCHAR(255) NOT NULL,
    manufacturer VARCHAR(255) NOT NULL,
    model_number VARCHAR(100),
    technology VARCHAR(50),

    -- STC ratings (1000 W/m2, 25C, AM1.5G)
    pmax_stc NUMERIC(10,4) NOT NULL CHECK (pmax_stc > 0),
    voc_stc NUMERIC(10,4) NOT NULL CHECK (voc_stc > 0),
    isc_stc NUMERIC(10,4) NOT NULL CHECK (isc_stc > 0),
    vmpp_stc NUMERIC(10,4) NOT NULL,
    impp_stc NUMERIC(10,4) NOT NULL,
    fill_factor_stc NUMERIC(6,4),

    -- Temperature coefficients
    alpha_isc NUMERIC(10,6) NOT NULL,
    beta_voc NUMERIC(10,6) NOT NULL,
    gamma_pmax NUMERIC(10,6) NOT NULL,
    temp_coeff_unit VARCHAR(10) NOT NULL DEFAULT 'percent',

    -- Physical parameters
    cell_count INTEGER,
    cell_type VARCHAR(50),
    module_area NUMERIC(10,4),
    aperture_area NUMERIC(10,4),

    -- NMOT parameters (IEC 61853-2)
    nmot NUMERIC(6,2),
    nmot_irradiance NUMERIC(8,2) DEFAULT 800,
    nmot_ambient NUMERIC(6,2) DEFAULT 20,
    nmot_wind_speed NUMERIC(6,2) DEFAULT 1,

    -- Thermal parameters
    u_const NUMERIC(8,4),
    u_wind NUMERIC(8,4),

    -- Metadata
    owner_id INTEGER REFERENCES users(id),
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_pv_modules_module_name ON pv_modules(module_name);
CREATE INDEX IF NOT EXISTS ix_pv_modules_manufacturer ON pv_modules(manufacturer);
CREATE INDEX IF NOT EXISTS ix_pv_modules_manufacturer_name ON pv_modules(manufacturer, module_name);

-- =============================================================================
-- Power Matrix Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS power_matrix (
    id SERIAL PRIMARY KEY,
    module_id INTEGER NOT NULL REFERENCES pv_modules(id) ON DELETE CASCADE,

    -- Operating conditions
    irradiance NUMERIC(8,2) NOT NULL CHECK (irradiance >= 0),
    temperature NUMERIC(6,2) NOT NULL,

    -- Measured parameters
    isc NUMERIC(10,4) NOT NULL,
    voc NUMERIC(10,4) NOT NULL,
    pmax NUMERIC(10,4) NOT NULL CHECK (pmax >= 0),
    vmpp NUMERIC(10,4) NOT NULL,
    impp NUMERIC(10,4) NOT NULL,
    fill_factor NUMERIC(6,4),

    -- Quality indicators
    measurement_uncertainty NUMERIC(6,4),
    measurement_date TIMESTAMP,

    CONSTRAINT uq_power_matrix_point UNIQUE (module_id, irradiance, temperature)
);

CREATE INDEX IF NOT EXISTS ix_power_matrix_module_id ON power_matrix(module_id);
CREATE INDEX IF NOT EXISTS ix_power_matrix_conditions ON power_matrix(module_id, irradiance, temperature);

-- =============================================================================
-- Spectral Response Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS spectral_response (
    id SERIAL PRIMARY KEY,
    module_id INTEGER NOT NULL REFERENCES pv_modules(id) ON DELETE CASCADE,

    -- Wavelength and response
    wavelength NUMERIC(8,2) NOT NULL CHECK (wavelength > 0),
    response NUMERIC(10,6) NOT NULL,
    response_unit VARCHAR(20) NOT NULL DEFAULT 'normalized',

    -- Measurement metadata
    measurement_date TIMESTAMP,
    measurement_uncertainty NUMERIC(6,4),

    CONSTRAINT uq_spectral_wavelength UNIQUE (module_id, wavelength)
);

CREATE INDEX IF NOT EXISTS ix_spectral_response_module_id ON spectral_response(module_id);
CREATE INDEX IF NOT EXISTS ix_spectral_module_wavelength ON spectral_response(module_id, wavelength);

-- =============================================================================
-- IAM Data Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS iam_data (
    id SERIAL PRIMARY KEY,
    module_id INTEGER NOT NULL REFERENCES pv_modules(id) ON DELETE CASCADE,

    -- Angle and IAM value
    angle NUMERIC(6,2) NOT NULL CHECK (angle >= 0 AND angle <= 90),
    iam_value NUMERIC(8,6) NOT NULL CHECK (iam_value >= 0 AND iam_value <= 1),
    azimuth NUMERIC(6,2),

    -- Measurement metadata
    measurement_date TIMESTAMP,
    measurement_uncertainty NUMERIC(6,4),

    CONSTRAINT uq_iam_angle UNIQUE (module_id, angle, azimuth)
);

CREATE INDEX IF NOT EXISTS ix_iam_data_module_id ON iam_data(module_id);
CREATE INDEX IF NOT EXISTS ix_iam_module_angle ON iam_data(module_id, angle);

-- =============================================================================
-- Climate Profiles Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS climate_profiles (
    id SERIAL PRIMARY KEY,

    -- Profile identification
    profile_name VARCHAR(100) NOT NULL UNIQUE,
    profile_code VARCHAR(20) NOT NULL UNIQUE,
    description TEXT,

    -- Climate characteristics
    climate_type VARCHAR(50) NOT NULL,
    is_standard BOOLEAN NOT NULL DEFAULT TRUE,

    -- Location reference
    latitude NUMERIC(9,6),
    longitude NUMERIC(9,6),
    elevation NUMERIC(8,2),
    timezone VARCHAR(50),

    -- Annual summary statistics
    annual_ghi NUMERIC(10,2),
    annual_dni NUMERIC(10,2),
    annual_dhi NUMERIC(10,2),
    avg_temperature NUMERIC(6,2),
    avg_wind_speed NUMERIC(6,2),

    -- Metadata
    source VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_climate_profiles_profile_name ON climate_profiles(profile_name);

-- =============================================================================
-- Hourly Climate Data Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS hourly_climate_data (
    id SERIAL PRIMARY KEY,
    climate_profile_id INTEGER NOT NULL REFERENCES climate_profiles(id) ON DELETE CASCADE,

    -- Time identification
    hour_of_year INTEGER NOT NULL CHECK (hour_of_year >= 1 AND hour_of_year <= 8760),
    month INTEGER NOT NULL CHECK (month >= 1 AND month <= 12),
    day INTEGER NOT NULL CHECK (day >= 1 AND day <= 31),
    hour INTEGER NOT NULL CHECK (hour >= 0 AND hour <= 23),

    -- Irradiance components (W/m2)
    ghi NUMERIC(8,2) NOT NULL CHECK (ghi >= 0),
    dni NUMERIC(8,2),
    dhi NUMERIC(8,2),
    poa_global NUMERIC(8,2),
    poa_direct NUMERIC(8,2),
    poa_diffuse NUMERIC(8,2),

    -- Meteorological data
    ambient_temperature NUMERIC(6,2) NOT NULL,
    wind_speed NUMERIC(6,2),
    relative_humidity NUMERIC(6,2),
    pressure NUMERIC(8,2),

    -- Spectral data
    airmass NUMERIC(8,4),
    precipitable_water NUMERIC(8,4),
    aod NUMERIC(8,6),

    -- Sun position
    solar_zenith NUMERIC(8,4),
    solar_azimuth NUMERIC(8,4),

    CONSTRAINT uq_hourly_data_hour UNIQUE (climate_profile_id, hour_of_year)
);

CREATE INDEX IF NOT EXISTS ix_hourly_climate_data_profile_id ON hourly_climate_data(climate_profile_id);
CREATE INDEX IF NOT EXISTS ix_hourly_climate_profile_hour ON hourly_climate_data(climate_profile_id, hour_of_year);
CREATE INDEX IF NOT EXISTS ix_hourly_climate_month ON hourly_climate_data(climate_profile_id, month);

-- =============================================================================
-- CSER Calculations Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS cser_calculations (
    id SERIAL PRIMARY KEY,

    -- References
    module_id INTEGER NOT NULL REFERENCES pv_modules(id) ON DELETE CASCADE,
    climate_profile_id INTEGER NOT NULL REFERENCES climate_profiles(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),

    -- Calculation parameters
    calculation_name VARCHAR(255),
    tilt_angle NUMERIC(6,2),
    azimuth_angle NUMERIC(6,2),
    albedo NUMERIC(4,2) DEFAULT 0.2,

    -- CSER results
    cser_value NUMERIC(12,4) NOT NULL CHECK (cser_value >= 0),
    annual_energy NUMERIC(12,4) NOT NULL CHECK (annual_energy >= 0),
    performance_ratio NUMERIC(6,4),

    -- Loss factors
    spectral_loss NUMERIC(8,4),
    angular_loss NUMERIC(8,4),
    thermal_loss NUMERIC(8,4),
    low_irradiance_loss NUMERIC(8,4),

    -- Monthly/hourly breakdown (JSONB)
    monthly_energy JSONB,
    hourly_results JSONB,

    -- Calculation metadata
    calculation_method VARCHAR(50),
    software_version VARCHAR(50),
    notes TEXT,
    calculated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_cser_calculations_module_id ON cser_calculations(module_id);
CREATE INDEX IF NOT EXISTS ix_cser_calculations_climate_profile_id ON cser_calculations(climate_profile_id);
CREATE INDEX IF NOT EXISTS ix_cser_calculations_user_id ON cser_calculations(user_id);
CREATE INDEX IF NOT EXISTS ix_cser_module_climate ON cser_calculations(module_id, climate_profile_id);
CREATE INDEX IF NOT EXISTS ix_cser_calculated_at ON cser_calculations(calculated_at);

-- =============================================================================
-- Seed Standard IEC 61853-4 Climate Profiles
-- =============================================================================
INSERT INTO climate_profiles (profile_name, profile_code, description, climate_type, is_standard, latitude, longitude, elevation, annual_ghi, avg_temperature, source)
VALUES
    ('Subtropical Arid', 'SA', 'Hot and dry climate with high irradiance (e.g., Phoenix, USA)', 'BWh', TRUE, 33.45, -112.07, 331, 2040, 23.9, 'IEC 61853-4'),
    ('Subtropical Coastal', 'SC', 'Warm coastal climate with moderate humidity (e.g., Miami, USA)', 'Am', TRUE, 25.76, -80.19, 2, 1770, 24.8, 'IEC 61853-4'),
    ('Temperate Coastal', 'TC', 'Mild maritime climate (e.g., Boulogne, France)', 'Cfb', TRUE, 50.73, 1.62, 73, 1100, 10.5, 'IEC 61853-4'),
    ('High Elevation', 'HE', 'High altitude climate (e.g., Vail, USA)', 'Dfc', TRUE, 39.64, -106.37, 2500, 1650, 3.3, 'IEC 61853-4'),
    ('Temperate Continental', 'TCO', 'Continental climate with warm summers (e.g., Munich, Germany)', 'Dfb', TRUE, 48.14, 11.58, 519, 1190, 9.1, 'IEC 61853-4'),
    ('Tropical Humid', 'TH', 'Hot and humid tropical climate (e.g., Singapore)', 'Af', TRUE, 1.35, 103.82, 15, 1580, 27.0, 'IEC 61853-4')
ON CONFLICT (profile_code) DO NOTHING;

-- =============================================================================
-- Updated_at Trigger Function
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_pv_modules_updated_at ON pv_modules;
CREATE TRIGGER update_pv_modules_updated_at
    BEFORE UPDATE ON pv_modules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_climate_profiles_updated_at ON climate_profiles;
CREATE TRIGGER update_climate_profiles_updated_at
    BEFORE UPDATE ON climate_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Grant permissions (adjust as needed for Railway)
-- =============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO railway;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO railway;

-- Verification
SELECT 'Schema initialization complete' AS status;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
