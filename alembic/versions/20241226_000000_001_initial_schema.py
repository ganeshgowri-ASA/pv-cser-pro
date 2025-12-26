"""Initial database schema for PV-CSER Pro

Revision ID: 001
Revises:
Create Date: 2024-12-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create pv_modules table
    op.create_table(
        'pv_modules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('manufacturer', sa.String(length=255), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('serial_number', sa.String(length=100), nullable=True),
        sa.Column('pmax_stc', sa.Float(), nullable=False),
        sa.Column('voc_stc', sa.Float(), nullable=True),
        sa.Column('isc_stc', sa.Float(), nullable=True),
        sa.Column('vmp_stc', sa.Float(), nullable=True),
        sa.Column('imp_stc', sa.Float(), nullable=True),
        sa.Column('fill_factor', sa.Float(), nullable=True),
        sa.Column('temp_coeff_pmax', sa.Float(), nullable=True),
        sa.Column('temp_coeff_voc', sa.Float(), nullable=True),
        sa.Column('temp_coeff_isc', sa.Float(), nullable=True),
        sa.Column('temp_coeff_pmax_abs', sa.Float(), nullable=True),
        sa.Column('temp_coeff_voc_abs', sa.Float(), nullable=True),
        sa.Column('temp_coeff_isc_abs', sa.Float(), nullable=True),
        sa.Column('module_area', sa.Float(), nullable=True),
        sa.Column('cell_area', sa.Float(), nullable=True),
        sa.Column('cell_type', sa.String(length=50), nullable=True),
        sa.Column('num_cells', sa.Integer(), nullable=True),
        sa.Column('num_strings', sa.Integer(), nullable=True),
        sa.Column('length', sa.Float(), nullable=True),
        sa.Column('width', sa.Float(), nullable=True),
        sa.Column('thickness', sa.Float(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('nmot', sa.Float(), nullable=True),
        sa.Column('noct', sa.Float(), nullable=True),
        sa.Column('efficiency_stc', sa.Float(), nullable=True),
        sa.Column('iec_certification', sa.String(length=255), nullable=True),
        sa.Column('certification_date', sa.DateTime(), nullable=True),
        sa.Column('bifacial', sa.Boolean(), nullable=True, default=False),
        sa.Column('bifaciality_factor', sa.Float(), nullable=True),
        sa.Column('additional_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('manufacturer', 'model_name', 'serial_number', name='uq_module_identity')
    )
    op.create_index('idx_module_manufacturer', 'pv_modules', ['manufacturer'])
    op.create_index('idx_module_model', 'pv_modules', ['model_name'])
    op.create_index('idx_module_created', 'pv_modules', ['created_at'])

    # Create power_matrices table
    op.create_table(
        'power_matrices',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('module_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('num_irradiance_levels', sa.Integer(), nullable=False),
        sa.Column('num_temperature_levels', sa.Integer(), nullable=False),
        sa.Column('irradiance_levels', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('temperature_levels', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('power_values', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('current_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('voltage_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('voc_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('isc_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('normalized_power', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('measurement_date', sa.DateTime(), nullable=True),
        sa.Column('measurement_location', sa.String(length=255), nullable=True),
        sa.Column('measurement_uncertainty', sa.Float(), nullable=True),
        sa.Column('laboratory', sa.String(length=255), nullable=True),
        sa.Column('is_validated', sa.Boolean(), nullable=True, default=False),
        sa.Column('validation_notes', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['module_id'], ['pv_modules.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_power_matrix_module', 'power_matrices', ['module_id'])
    op.create_index('idx_power_matrix_created', 'power_matrices', ['created_at'])

    # Create spectral_responses table
    op.create_table(
        'spectral_responses',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('module_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('wavelengths', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('response_values', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('is_normalized', sa.Boolean(), nullable=True, default=False),
        sa.Column('reference_spectrum', sa.String(length=100), nullable=True),
        sa.Column('measurement_date', sa.DateTime(), nullable=True),
        sa.Column('measurement_location', sa.String(length=255), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['module_id'], ['pv_modules.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_spectral_module', 'spectral_responses', ['module_id'])

    # Create iam_data table
    op.create_table(
        'iam_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('module_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('angles', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('iam_values', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=True),
        sa.Column('model_parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('measurement_date', sa.DateTime(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['module_id'], ['pv_modules.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_iam_module', 'iam_data', ['module_id'])

    # Create climate_profiles table
    op.create_table(
        'climate_profiles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('profile_name', sa.String(length=255), nullable=False),
        sa.Column('profile_code', sa.String(length=50), nullable=True),
        sa.Column('profile_type', sa.String(length=50), nullable=True, default='standard'),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('country', sa.String(length=100), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('elevation', sa.Float(), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('annual_ghi', sa.Float(), nullable=True),
        sa.Column('annual_dni', sa.Float(), nullable=True),
        sa.Column('annual_dhi', sa.Float(), nullable=True),
        sa.Column('avg_temperature', sa.Float(), nullable=True),
        sa.Column('avg_wind_speed', sa.Float(), nullable=True),
        sa.Column('ghi_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('dni_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('dhi_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('temperature_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('wind_speed_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('relative_humidity_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('spectral_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('average_am', sa.Float(), nullable=True),
        sa.Column('is_standard', sa.Boolean(), nullable=True, default=False),
        sa.Column('source', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('profile_name', 'location', name='uq_climate_profile')
    )
    op.create_index('idx_climate_profile_type', 'climate_profiles', ['profile_type'])
    op.create_index('idx_climate_location', 'climate_profiles', ['location'])

    # Create cser_calculations table
    op.create_table(
        'cser_calculations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('module_id', sa.Integer(), nullable=False),
        sa.Column('climate_profile_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True, default='completed'),
        sa.Column('cser_value', sa.Float(), nullable=True),
        sa.Column('annual_energy_yield', sa.Float(), nullable=True),
        sa.Column('annual_dc_energy', sa.Float(), nullable=True),
        sa.Column('specific_yield', sa.Float(), nullable=True),
        sa.Column('performance_ratio', sa.Float(), nullable=True),
        sa.Column('capacity_factor', sa.Float(), nullable=True),
        sa.Column('avg_cell_temperature', sa.Float(), nullable=True),
        sa.Column('max_cell_temperature', sa.Float(), nullable=True),
        sa.Column('operating_hours', sa.Integer(), nullable=True),
        sa.Column('monthly_yields', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('monthly_cser', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('hourly_yields', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('loss_breakdown', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('temperature_loss', sa.Float(), nullable=True),
        sa.Column('low_irradiance_loss', sa.Float(), nullable=True),
        sa.Column('spectral_loss', sa.Float(), nullable=True),
        sa.Column('iam_loss', sa.Float(), nullable=True),
        sa.Column('soiling_loss', sa.Float(), nullable=True),
        sa.Column('total_losses', sa.Float(), nullable=True),
        sa.Column('calculation_method', sa.String(length=100), nullable=True),
        sa.Column('temperature_model', sa.String(length=50), nullable=True),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['climate_profile_id'], ['climate_profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['module_id'], ['pv_modules.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_cser_module', 'cser_calculations', ['module_id'])
    op.create_index('idx_cser_climate', 'cser_calculations', ['climate_profile_id'])
    op.create_index('idx_cser_status', 'cser_calculations', ['status'])
    op.create_index('idx_cser_created', 'cser_calculations', ['created_at'])

    # Create calculation_logs table
    op.create_table(
        'calculation_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('calculation_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['calculation_id'], ['cser_calculations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_calc_log_calculation', 'calculation_logs', ['calculation_id'])
    op.create_index('idx_calc_log_timestamp', 'calculation_logs', ['timestamp'])

    # Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('ip_address', sa.String(length=50), nullable=True),
        sa.Column('current_module_id', sa.Integer(), nullable=True),
        sa.Column('session_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['current_module_id'], ['pv_modules.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index('idx_session_id', 'user_sessions', ['session_id'])
    op.create_index('idx_session_created', 'user_sessions', ['created_at'])

    # Create file_uploads table
    op.create_table(
        'file_uploads',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('original_filename', sa.String(length=500), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('file_hash', sa.String(length=64), nullable=True),
        sa.Column('upload_type', sa.String(length=50), nullable=True),
        sa.Column('is_valid', sa.Boolean(), nullable=True),
        sa.Column('validation_errors', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('validation_warnings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('processed', sa.Boolean(), nullable=True, default=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('result_id', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_upload_session', 'file_uploads', ['session_id'])
    op.create_index('idx_upload_created', 'file_uploads', ['created_at'])

    # Create export_records table
    op.create_table(
        'export_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('export_type', sa.String(length=50), nullable=False),
        sa.Column('export_format', sa.String(length=50), nullable=True),
        sa.Column('module_id', sa.Integer(), nullable=True),
        sa.Column('calculation_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('filename', sa.String(length=500), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True, default=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['module_id'], ['pv_modules.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_export_session', 'export_records', ['session_id'])
    op.create_index('idx_export_created', 'export_records', ['created_at'])


def downgrade() -> None:
    op.drop_table('export_records')
    op.drop_table('file_uploads')
    op.drop_table('user_sessions')
    op.drop_table('calculation_logs')
    op.drop_table('cser_calculations')
    op.drop_table('climate_profiles')
    op.drop_table('iam_data')
    op.drop_table('spectral_responses')
    op.drop_table('power_matrices')
    op.drop_table('pv_modules')
