"""
Excel export functionality for PV-CSER Pro.

Generates comprehensive Excel reports with multiple worksheets.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ExcelExporter:
    """
    Excel report exporter for CSER analysis.

    Creates Excel workbooks with multiple sheets containing
    module data, power matrices, and calculation results.
    """

    def __init__(self):
        """Initialize Excel exporter."""
        pass

    def export_complete_report(
        self,
        module_data: Dict[str, Any],
        power_matrix_data: Optional[Dict] = None,
        cser_results: Optional[Dict[str, Any]] = None,
        climate_comparison: Optional[Dict[str, Dict]] = None,
    ) -> BytesIO:
        """
        Export complete analysis to Excel workbook.

        Args:
            module_data: Module specification data
            power_matrix_data: Power matrix data (optional)
            cser_results: CSER calculation results (optional)
            climate_comparison: Comparison across climates (optional)

        Returns:
            BytesIO containing Excel workbook
        """
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#2C5282',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
            })

            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'font_color': '#1E3A5F',
            })

            number_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1,
            })

            # 1. Summary sheet
            self._write_summary_sheet(
                writer, module_data, cser_results, header_format, title_format
            )

            # 2. Module specifications sheet
            self._write_module_sheet(writer, module_data, header_format)

            # 3. Power matrix sheet
            if power_matrix_data:
                self._write_power_matrix_sheet(
                    writer, power_matrix_data, header_format, number_format
                )

            # 4. CSER results sheet
            if cser_results:
                self._write_cser_sheet(writer, cser_results, header_format)

            # 5. Climate comparison sheet
            if climate_comparison:
                self._write_comparison_sheet(
                    writer, climate_comparison, header_format
                )

        buffer.seek(0)
        return buffer

    def _write_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        module_data: Dict[str, Any],
        cser_results: Optional[Dict],
        header_format,
        title_format,
    ) -> None:
        """Write summary sheet."""
        worksheet = writer.book.add_worksheet('Summary')

        # Title
        worksheet.write(0, 0, 'PV-CSER Pro Analysis Summary', title_format)
        worksheet.write(1, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Module info
        row = 3
        worksheet.write(row, 0, 'Module Information', title_format)
        row += 1
        worksheet.write(row, 0, 'Manufacturer:', header_format)
        worksheet.write(row, 1, module_data.get('manufacturer', 'N/A'))
        row += 1
        worksheet.write(row, 0, 'Model:', header_format)
        worksheet.write(row, 1, module_data.get('model_name', 'N/A'))
        row += 1
        worksheet.write(row, 0, 'Pmax (STC):', header_format)
        worksheet.write(row, 1, f"{module_data.get('pmax_stc', 0):.1f} W")

        # CSER results
        if cser_results:
            row += 2
            worksheet.write(row, 0, 'Key Results', title_format)
            row += 1
            worksheet.write(row, 0, 'CSER:', header_format)
            worksheet.write(row, 1, f"{cser_results.get('cser', 0):.0f} kWh/kWp")
            row += 1
            worksheet.write(row, 0, 'Annual Energy:', header_format)
            worksheet.write(row, 1, f"{cser_results.get('annual_energy', 0):.1f} kWh")
            row += 1
            worksheet.write(row, 0, 'Performance Ratio:', header_format)
            worksheet.write(row, 1, f"{cser_results.get('performance_ratio', 0):.1f}%")
            row += 1
            worksheet.write(row, 0, 'Climate Profile:', header_format)
            worksheet.write(row, 1, cser_results.get('climate_profile', 'N/A'))

        # Adjust column widths
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 1, 25)

    def _write_module_sheet(
        self,
        writer: pd.ExcelWriter,
        module_data: Dict[str, Any],
        header_format,
    ) -> None:
        """Write module specifications sheet."""
        specs_df = pd.DataFrame([
            {'Parameter': 'Manufacturer', 'Value': module_data.get('manufacturer', 'N/A'), 'Unit': ''},
            {'Parameter': 'Model Name', 'Value': module_data.get('model_name', 'N/A'), 'Unit': ''},
            {'Parameter': 'Pmax (STC)', 'Value': module_data.get('pmax_stc', 0), 'Unit': 'W'},
            {'Parameter': 'Voc (STC)', 'Value': module_data.get('voc_stc', 0), 'Unit': 'V'},
            {'Parameter': 'Isc (STC)', 'Value': module_data.get('isc_stc', 0), 'Unit': 'A'},
            {'Parameter': 'Vmp (STC)', 'Value': module_data.get('vmp_stc', 0), 'Unit': 'V'},
            {'Parameter': 'Imp (STC)', 'Value': module_data.get('imp_stc', 0), 'Unit': 'A'},
            {'Parameter': 'Temp Coeff (Pmax)', 'Value': module_data.get('temp_coeff_pmax', -0.35), 'Unit': '%/°C'},
            {'Parameter': 'Temp Coeff (Voc)', 'Value': module_data.get('temp_coeff_voc', -0.30), 'Unit': '%/°C'},
            {'Parameter': 'Temp Coeff (Isc)', 'Value': module_data.get('temp_coeff_isc', 0.05), 'Unit': '%/°C'},
            {'Parameter': 'NMOT', 'Value': module_data.get('nmot', 45), 'Unit': '°C'},
            {'Parameter': 'Module Area', 'Value': module_data.get('module_area', 0), 'Unit': 'm²'},
            {'Parameter': 'Cell Type', 'Value': module_data.get('cell_type', 'N/A'), 'Unit': ''},
            {'Parameter': 'Number of Cells', 'Value': module_data.get('num_cells', 0), 'Unit': ''},
        ])

        specs_df.to_excel(writer, sheet_name='Module Specifications', index=False)

        # Format header
        worksheet = writer.sheets['Module Specifications']
        for col_num, value in enumerate(specs_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 1, 15)
        worksheet.set_column(2, 2, 10)

    def _write_power_matrix_sheet(
        self,
        writer: pd.ExcelWriter,
        power_matrix_data: Dict,
        header_format,
        number_format,
    ) -> None:
        """Write power matrix sheet."""
        irradiance = power_matrix_data.get('irradiance', [])
        temperature = power_matrix_data.get('temperature', [])
        power_matrix = power_matrix_data.get('power_matrix', [[]])

        # Create DataFrame
        df = pd.DataFrame(
            power_matrix,
            index=[f"{g} W/m²" for g in irradiance],
            columns=[f"{t}°C" for t in temperature],
        )
        df.index.name = 'Irradiance \\ Temperature'

        df.to_excel(writer, sheet_name='Power Matrix')

        # Format
        worksheet = writer.sheets['Power Matrix']
        worksheet.set_column(0, 0, 18)
        for i in range(len(temperature)):
            worksheet.set_column(i+1, i+1, 12)

    def _write_cser_sheet(
        self,
        writer: pd.ExcelWriter,
        cser_results: Dict[str, Any],
        header_format,
    ) -> None:
        """Write CSER results sheet."""
        # Main results
        results_df = pd.DataFrame([
            {'Metric': 'CSER', 'Value': cser_results.get('cser', 0), 'Unit': 'kWh/kWp'},
            {'Metric': 'Annual Energy', 'Value': cser_results.get('annual_energy', 0), 'Unit': 'kWh'},
            {'Metric': 'Performance Ratio', 'Value': cser_results.get('performance_ratio', 0), 'Unit': '%'},
            {'Metric': 'Temperature Loss', 'Value': cser_results.get('temperature_loss', 0), 'Unit': '%'},
            {'Metric': 'Low Irradiance Loss', 'Value': cser_results.get('low_irradiance_loss', 0), 'Unit': '%'},
            {'Metric': 'Climate Profile', 'Value': cser_results.get('climate_profile', 'N/A'), 'Unit': ''},
            {'Metric': 'Annual Irradiation', 'Value': cser_results.get('annual_irradiation', 0), 'Unit': 'kWh/m²'},
            {'Metric': 'Average Temperature', 'Value': cser_results.get('avg_temperature', 0), 'Unit': '°C'},
        ])

        results_df.to_excel(writer, sheet_name='CSER Results', index=False, startrow=0)

        # Monthly breakdown
        monthly_energy = cser_results.get('monthly_energy', [])
        monthly_cser = cser_results.get('monthly_cser', [])

        if monthly_energy:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            monthly_df = pd.DataFrame({
                'Month': months,
                'Energy (kWh)': monthly_energy,
                'Yield (kWh/kWp)': monthly_cser if monthly_cser else [0]*12,
            })

            # Add totals
            monthly_df = pd.concat([
                monthly_df,
                pd.DataFrame([{
                    'Month': 'Total',
                    'Energy (kWh)': sum(monthly_energy),
                    'Yield (kWh/kWp)': cser_results.get('cser', 0),
                }])
            ], ignore_index=True)

            monthly_df.to_excel(
                writer,
                sheet_name='CSER Results',
                index=False,
                startrow=len(results_df) + 3,
            )

        # Format
        worksheet = writer.sheets['CSER Results']
        for col_num, value in enumerate(results_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 1, 15)
        worksheet.set_column(2, 2, 12)

    def _write_comparison_sheet(
        self,
        writer: pd.ExcelWriter,
        climate_comparison: Dict[str, Dict],
        header_format,
    ) -> None:
        """Write climate comparison sheet."""
        comparison_data = []

        for profile_name, data in climate_comparison.items():
            comparison_data.append({
                'Climate Profile': profile_name,
                'CSER (kWh/kWp)': data.get('cser', 0),
                'Annual Energy (kWh)': data.get('annual_energy', 0),
                'Performance Ratio (%)': data.get('performance_ratio', 0),
                'Temperature Loss (%)': data.get('temperature_loss', 0),
                'Annual GHI (kWh/m²)': data.get('annual_irradiation', 0),
                'Avg Temp (°C)': data.get('avg_temperature', 0),
            })

        df = pd.DataFrame(comparison_data)

        # Sort by CSER
        df = df.sort_values('CSER (kWh/kWp)', ascending=False)

        df.to_excel(writer, sheet_name='Climate Comparison', index=False)

        # Format
        worksheet = writer.sheets['Climate Comparison']
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        worksheet.set_column(0, 0, 22)
        for i in range(1, len(df.columns)):
            worksheet.set_column(i, i, 18)

    def export_power_matrix(
        self,
        irradiance: np.ndarray,
        temperature: np.ndarray,
        power_matrix: np.ndarray,
        module_name: str = "Module",
    ) -> BytesIO:
        """
        Export just the power matrix to Excel.

        Args:
            irradiance: Irradiance array
            temperature: Temperature array
            power_matrix: 2D power matrix
            module_name: Module name for filename

        Returns:
            BytesIO containing Excel workbook
        """
        buffer = BytesIO()

        df = pd.DataFrame(
            power_matrix,
            index=irradiance,
            columns=temperature,
        )
        df.index.name = 'Irradiance (W/m²)'
        df.columns.name = 'Temperature (°C)'

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Power Matrix')

            worksheet = writer.sheets['Power Matrix']
            worksheet.set_column(0, 0, 18)

        buffer.seek(0)
        return buffer

    def export_monthly_data(
        self,
        monthly_energy: List[float],
        monthly_irradiation: List[float],
        module_pmax: float,
    ) -> BytesIO:
        """
        Export monthly energy data to Excel.

        Args:
            monthly_energy: Monthly energy values (kWh)
            monthly_irradiation: Monthly irradiation values (kWh/m²)
            module_pmax: Module power at STC (W)

        Returns:
            BytesIO containing Excel workbook
        """
        buffer = BytesIO()

        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']

        pmax_kw = module_pmax / 1000

        df = pd.DataFrame({
            'Month': months,
            'Energy (kWh)': monthly_energy,
            'Yield (kWh/kWp)': [e/pmax_kw for e in monthly_energy],
            'Irradiation (kWh/m²)': monthly_irradiation,
        })

        # Add totals
        totals = pd.DataFrame([{
            'Month': 'Annual Total',
            'Energy (kWh)': sum(monthly_energy),
            'Yield (kWh/kWp)': sum(monthly_energy) / pmax_kw,
            'Irradiation (kWh/m²)': sum(monthly_irradiation),
        }])

        df = pd.concat([df, totals], ignore_index=True)

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Monthly Data', index=False)

        buffer.seek(0)
        return buffer
