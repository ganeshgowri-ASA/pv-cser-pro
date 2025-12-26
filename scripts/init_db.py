#!/usr/bin/env python3
"""
Database Initialization Script for PV-CSER Pro.

This script initializes the PostgreSQL database with all required tables
and optionally seeds it with sample data.

Usage:
    python scripts/init_db.py [--seed] [--force]

Options:
    --seed    Load sample data after creating tables
    --force   Drop existing tables before creating new ones
    --check   Only check database connection
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(project_root / ".env")

from sqlalchemy import text
from src.utils.database import (
    Base,
    DatabaseManager,
    PVModule,
    PowerMatrix,
    ClimateProfile,
    CSERCalculation,
    init_database,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "db_init.log"),
    ],
)
logger = logging.getLogger(__name__)


def check_connection(db_url: str) -> bool:
    """Check database connection."""
    logger.info("Checking database connection...")
    try:
        db = DatabaseManager(db_url)
        db.initialize()
        result = db.check_connection()
        if result:
            logger.info("Database connection successful!")
            return True
        else:
            logger.error("Database connection failed!")
            return False
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return False


def drop_tables(db: DatabaseManager) -> None:
    """Drop all existing tables."""
    logger.warning("Dropping all existing tables...")
    try:
        Base.metadata.drop_all(bind=db.engine)
        logger.info("All tables dropped successfully.")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        raise


def create_tables(db: DatabaseManager) -> None:
    """Create all database tables."""
    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=db.engine)
        logger.info("All tables created successfully.")

        # List created tables
        with db.get_session() as session:
            result = session.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            logger.info(f"Created tables: {', '.join(tables)}")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def seed_sample_data(db: DatabaseManager) -> None:
    """Seed database with sample data."""
    logger.info("Seeding sample data...")

    try:
        # Create sample PV module
        module_data = {
            "manufacturer": "Sample Solar",
            "model_name": "PV-400M",
            "serial_number": "SS-2024-001",
            "pmax_stc": 400.0,
            "voc_stc": 48.5,
            "isc_stc": 10.5,
            "vmp_stc": 40.8,
            "imp_stc": 9.8,
            "temp_coeff_pmax": -0.35,
            "temp_coeff_voc": -0.28,
            "temp_coeff_isc": 0.05,
            "module_area": 1.92,
            "cell_type": "mono-Si",
            "num_cells": 72,
            "nmot": 43.0,
            "bifacial": False,
        }
        module_id = db.create_module(module_data)
        logger.info(f"Created sample module with ID: {module_id}")

        # Create sample power matrix
        irradiance_levels = [100, 200, 400, 600, 800, 1000, 1100]
        temperature_levels = [15, 25, 50, 75]

        # Generate power matrix values (simplified model)
        power_values = []
        pmax_stc = 400.0
        gamma = -0.35 / 100  # Convert to fraction

        for T in temperature_levels:
            row = []
            for G in irradiance_levels:
                # P = Pstc * (G/1000) * (1 + gamma * (T - 25))
                low_light_factor = 1.0 if G >= 200 else (0.95 + 0.05 * (G / 200))
                P = pmax_stc * (G / 1000) * (1 + gamma * (T - 25)) * low_light_factor
                row.append(round(P, 2))
            power_values.append(row)

        matrix_id = db.create_power_matrix(
            module_id=module_id,
            irradiance_levels=irradiance_levels,
            temperature_levels=temperature_levels,
            power_values=power_values,
            notes="Sample power matrix generated for testing",
        )
        logger.info(f"Created sample power matrix with ID: {matrix_id}")

        # Create standard climate profiles (IEC 61853-4)
        climate_profiles = [
            {
                "profile_name": "Tropical Humid",
                "profile_code": "TROP_HUMID",
                "profile_type": "standard",
                "location": "Singapore",
                "country": "Singapore",
                "latitude": 1.35,
                "longitude": 103.82,
                "annual_ghi": 1630,
                "avg_temperature": 27.5,
                "is_standard": True,
                "source": "IEC 61853-4",
                "description": "Tropical humid climate with high temperature and humidity",
            },
            {
                "profile_name": "Subtropical Arid",
                "profile_code": "SUBTROP_ARID",
                "profile_type": "standard",
                "location": "Phoenix, Arizona",
                "country": "USA",
                "latitude": 33.45,
                "longitude": -112.07,
                "annual_ghi": 2100,
                "avg_temperature": 23.0,
                "is_standard": True,
                "source": "IEC 61853-4",
                "description": "Hot desert climate with high irradiance",
            },
            {
                "profile_name": "Subtropical Coastal",
                "profile_code": "SUBTROP_COAST",
                "profile_type": "standard",
                "location": "Brisbane",
                "country": "Australia",
                "latitude": -27.47,
                "longitude": 153.03,
                "annual_ghi": 1850,
                "avg_temperature": 20.5,
                "is_standard": True,
                "source": "IEC 61853-4",
                "description": "Subtropical coastal climate",
            },
            {
                "profile_name": "Temperate Coastal",
                "profile_code": "TEMP_COAST",
                "profile_type": "standard",
                "location": "Tokyo",
                "country": "Japan",
                "latitude": 35.68,
                "longitude": 139.69,
                "annual_ghi": 1350,
                "avg_temperature": 15.8,
                "is_standard": True,
                "source": "IEC 61853-4",
                "description": "Temperate coastal climate with seasonal variation",
            },
            {
                "profile_name": "High Elevation",
                "profile_code": "HIGH_ELEV",
                "profile_type": "standard",
                "location": "Denver, Colorado",
                "country": "USA",
                "latitude": 39.74,
                "longitude": -104.99,
                "annual_ghi": 1750,
                "avg_temperature": 10.5,
                "elevation": 1609,
                "is_standard": True,
                "source": "IEC 61853-4",
                "description": "High elevation continental climate",
            },
            {
                "profile_name": "Temperate Continental",
                "profile_code": "TEMP_CONT",
                "profile_type": "standard",
                "location": "Berlin",
                "country": "Germany",
                "latitude": 52.52,
                "longitude": 13.40,
                "annual_ghi": 1050,
                "avg_temperature": 9.5,
                "is_standard": True,
                "source": "IEC 61853-4",
                "description": "Temperate continental climate",
            },
        ]

        for profile in climate_profiles:
            profile_id = db.create_climate_profile(profile)
            logger.info(f"Created climate profile '{profile['profile_name']}' with ID: {profile_id}")

        # Create sample CSER calculation
        calc_results = {
            "cser_value": 1580.5,
            "annual_energy_yield": 632.2,
            "annual_dc_energy": 645.8,
            "specific_yield": 1580.5,
            "performance_ratio": 82.5,
            "capacity_factor": 18.0,
            "avg_cell_temperature": 42.3,
            "max_cell_temperature": 68.5,
            "operating_hours": 4380,
            "monthly_yields": {
                "Jan": 45.2, "Feb": 48.5, "Mar": 58.3, "Apr": 62.1,
                "May": 68.5, "Jun": 65.2, "Jul": 64.8, "Aug": 62.5,
                "Sep": 55.3, "Oct": 48.2, "Nov": 42.1, "Dec": 41.5
            },
            "loss_breakdown": {
                "temperature": 5.2,
                "low_irradiance": 2.1,
                "spectral": 1.0,
                "iam": 2.5,
                "soiling": 2.0,
                "mismatch": 2.0,
                "wiring": 1.5,
            },
            "temperature_loss": 5.2,
            "low_irradiance_loss": 2.1,
            "total_losses": 16.3,
            "calculation_method": "IEC 61853-3",
            "temperature_model": "NMOT",
        }

        calc_id = db.create_calculation(
            module_id=module_id,
            climate_profile_id=1,  # First climate profile
            results=calc_results,
        )
        logger.info(f"Created sample CSER calculation with ID: {calc_id}")

        logger.info("Sample data seeded successfully!")

    except Exception as e:
        logger.error(f"Error seeding sample data: {e}")
        raise


def print_summary(db: DatabaseManager) -> None:
    """Print database summary."""
    logger.info("\n" + "=" * 50)
    logger.info("DATABASE SUMMARY")
    logger.info("=" * 50)

    stats = db.get_table_stats()
    for table, count in stats.items():
        logger.info(f"  {table}: {count} records")

    logger.info("=" * 50 + "\n")


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(
        description="Initialize PV-CSER Pro database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Load sample data after creating tables",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing tables before creating new ones",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check database connection",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://localhost/pv_cser_pro"),
        help="Database connection URL",
    )

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("PV-CSER Pro Database Initialization")
    logger.info("=" * 50)
    logger.info(f"Database URL: {args.db_url.split('@')[-1]}")  # Hide credentials

    # Check connection only
    if args.check:
        success = check_connection(args.db_url)
        sys.exit(0 if success else 1)

    try:
        # Initialize database manager
        db = DatabaseManager(args.db_url)
        db.initialize()

        # Drop tables if force flag is set
        if args.force:
            confirm = input("This will DROP all existing tables. Are you sure? (yes/no): ")
            if confirm.lower() == "yes":
                drop_tables(db)
            else:
                logger.info("Aborted.")
                sys.exit(0)

        # Create tables
        create_tables(db)

        # Seed sample data if requested
        if args.seed:
            seed_sample_data(db)

        # Print summary
        print_summary(db)

        logger.info("Database initialization completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
