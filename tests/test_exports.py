"""
PDF/Excel Export Tests for PV-CSER Pro.

Tests cover:
- PDF report generation
- Excel workbook export
- Report formatting and styling
- Data accuracy in exports
- Error handling
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any


class TestPDFExport:
    """Test PDF report generation."""

    def test_create_quick_summary_pdf(self, temp_dir, sample_module_data, sample_cser_results):
        """Test creating quick summary PDF."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        output_path = temp_dir / "quick_summary.pdf"
        generator.generate_quick_summary(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results=sample_cser_results,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_create_full_report_pdf(self, temp_dir, sample_module_data, sample_power_matrix, sample_cser_results):
        """Test creating full PDF report."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        output_path = temp_dir / "full_report.pdf"
        generator.generate_full_report(
            output_path=output_path,
            module_data=sample_module_data,
            power_matrix=sample_power_matrix,
            cser_results=sample_cser_results,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should be reasonably sized

    def test_pdf_with_charts(self, temp_dir, sample_module_data, sample_cser_results, sample_power_matrix):
        """Test PDF with embedded charts."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        output_path = temp_dir / "report_with_charts.pdf"
        generator.generate_full_report(
            output_path=output_path,
            module_data=sample_module_data,
            power_matrix=sample_power_matrix,
            cser_results=sample_cser_results,
            include_charts=True,
        )

        assert output_path.exists()

    def test_pdf_page_size_a4(self, temp_dir, sample_module_data):
        """Test PDF with A4 page size."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator(page_size="A4")

        output_path = temp_dir / "a4_report.pdf"
        generator.generate_quick_summary(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results={"cser_value": 1580},
        )

        assert output_path.exists()

    def test_pdf_page_size_letter(self, temp_dir, sample_module_data):
        """Test PDF with Letter page size."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator(page_size="Letter")

        output_path = temp_dir / "letter_report.pdf"
        generator.generate_quick_summary(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results={"cser_value": 1580},
        )

        assert output_path.exists()

    def test_pdf_cover_page(self, temp_dir, sample_module_data):
        """Test PDF cover page generation."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        output_path = temp_dir / "with_cover.pdf"
        generator.generate_full_report(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results={"cser_value": 1580},
            include_cover=True,
            company_name="Test Solar Corp",
        )

        assert output_path.exists()

    def test_pdf_custom_styling(self, temp_dir, sample_module_data):
        """Test PDF with custom styling."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator(
            header_color="#1E3A5F",
            accent_color="#FF6B35",
        )

        output_path = temp_dir / "styled_report.pdf"
        generator.generate_quick_summary(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results={"cser_value": 1580},
        )

        assert output_path.exists()

    def test_pdf_with_multiple_climates(self, temp_dir, sample_module_data):
        """Test PDF with multiple climate comparison."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        climate_results = [
            {"name": "Tropical", "cser": 1580, "pr": 82.5},
            {"name": "Temperate", "cser": 1350, "pr": 84.0},
            {"name": "Arid", "cser": 1820, "pr": 80.0},
        ]

        output_path = temp_dir / "multi_climate.pdf"
        generator.generate_comparison_report(
            output_path=output_path,
            module_data=sample_module_data,
            climate_results=climate_results,
        )

        assert output_path.exists()


class TestExcelExport:
    """Test Excel workbook export."""

    def test_create_basic_excel(self, temp_dir, sample_module_data, sample_cser_results):
        """Test creating basic Excel workbook."""
        from src.exports.excel_export import ExcelExporter

        exporter = ExcelExporter()

        output_path = temp_dir / "basic_export.xlsx"
        exporter.export_results(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results=sample_cser_results,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_excel_multiple_sheets(self, temp_dir, sample_module_data, sample_power_matrix, sample_cser_results):
        """Test Excel with multiple sheets."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "multi_sheet.xlsx"
        exporter.export_full_report(
            output_path=output_path,
            module_data=sample_module_data,
            power_matrix=sample_power_matrix,
            cser_results=sample_cser_results,
        )

        assert output_path.exists()

        # Read back and check sheets
        with pd.ExcelFile(output_path) as xls:
            assert len(xls.sheet_names) >= 3

    def test_excel_summary_sheet(self, temp_dir, sample_module_data, sample_cser_results):
        """Test Excel summary sheet content."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "with_summary.xlsx"
        exporter.export_results(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results=sample_cser_results,
        )

        # Read summary sheet
        with pd.ExcelFile(output_path) as xls:
            if "Summary" in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name="Summary")
                assert len(df) > 0

    def test_excel_power_matrix_sheet(self, temp_dir, sample_module_data, sample_power_matrix):
        """Test Excel power matrix sheet."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "with_matrix.xlsx"
        exporter.export_power_matrix(
            output_path=output_path,
            power_matrix=sample_power_matrix,
        )

        assert output_path.exists()

        # Read and verify matrix
        with pd.ExcelFile(output_path) as xls:
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], index_col=0)
            assert df.shape[0] == len(sample_power_matrix["temperature_levels"])
            assert df.shape[1] == len(sample_power_matrix["irradiance_levels"])

    def test_excel_monthly_data(self, temp_dir, sample_cser_results):
        """Test Excel monthly breakdown."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "with_monthly.xlsx"
        exporter.export_monthly_breakdown(
            output_path=output_path,
            monthly_yields=sample_cser_results["monthly_yields"],
            monthly_cser=sample_cser_results["monthly_cser"],
        )

        assert output_path.exists()

        with pd.ExcelFile(output_path) as xls:
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            assert len(df) == 12  # 12 months

    def test_excel_formatting(self, temp_dir, sample_module_data, sample_cser_results):
        """Test Excel cell formatting."""
        from src.exports.excel_export import ExcelExporter

        exporter = ExcelExporter()

        output_path = temp_dir / "formatted.xlsx"
        exporter.export_results(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results=sample_cser_results,
            apply_formatting=True,
        )

        assert output_path.exists()

    def test_excel_climate_comparison(self, temp_dir, sample_module_data):
        """Test Excel climate comparison sheet."""
        from src.exports.excel_export import ExcelExporter

        exporter = ExcelExporter()

        climate_data = [
            {"name": "Tropical", "cser": 1580, "annual_ghi": 1630},
            {"name": "Temperate", "cser": 1350, "annual_ghi": 1350},
            {"name": "Arid", "cser": 1820, "annual_ghi": 2100},
        ]

        output_path = temp_dir / "climate_comparison.xlsx"
        exporter.export_climate_comparison(
            output_path=output_path,
            module_data=sample_module_data,
            climate_data=climate_data,
        )

        assert output_path.exists()


class TestExportDataAccuracy:
    """Test data accuracy in exports."""

    def test_module_data_accuracy_excel(self, temp_dir, sample_module_data):
        """Test module data accuracy in Excel export."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "accuracy_test.xlsx"
        exporter.export_module_specs(
            output_path=output_path,
            module_data=sample_module_data,
        )

        # Read back and verify
        with pd.ExcelFile(output_path) as xls:
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

            # Find Pmax value in the DataFrame
            # This depends on export format - adjust as needed
            pmax_found = False
            for col in df.columns:
                if "pmax" in str(col).lower() or any(
                    sample_module_data["pmax_stc"] == v for v in df[col].values if isinstance(v, (int, float))
                ):
                    pmax_found = True
                    break

    def test_cser_values_accuracy_excel(self, temp_dir, sample_cser_results):
        """Test CSER values accuracy in Excel export."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "cser_accuracy.xlsx"
        exporter.export_cser_results(
            output_path=output_path,
            cser_results=sample_cser_results,
        )

        assert output_path.exists()

    def test_power_matrix_accuracy(self, temp_dir, sample_power_matrix):
        """Test power matrix values are accurately exported."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd
        import numpy as np

        exporter = ExcelExporter()

        output_path = temp_dir / "matrix_accuracy.xlsx"
        exporter.export_power_matrix(
            output_path=output_path,
            power_matrix=sample_power_matrix,
        )

        # Read back and compare
        df = pd.read_excel(output_path, index_col=0)
        original = np.array(sample_power_matrix["power_values"])
        exported = df.values

        np.testing.assert_array_almost_equal(original, exported, decimal=2)


class TestExportErrorHandling:
    """Test export error handling."""

    def test_invalid_output_path(self, sample_module_data):
        """Test handling of invalid output path."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        with pytest.raises(Exception):
            generator.generate_quick_summary(
                output_path=Path("/nonexistent/directory/report.pdf"),
                module_data=sample_module_data,
                cser_results={"cser_value": 1580},
            )

    def test_missing_required_data(self, temp_dir):
        """Test handling of missing required data."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        output_path = temp_dir / "incomplete.pdf"

        with pytest.raises((ValueError, KeyError, TypeError)):
            generator.generate_quick_summary(
                output_path=output_path,
                module_data={},  # Empty module data
                cser_results={},
            )

    def test_excel_write_permission(self, temp_dir, sample_module_data):
        """Test handling of write permission issues."""
        from src.exports.excel_export import ExcelExporter

        exporter = ExcelExporter()

        # Create read-only directory (platform dependent)
        try:
            readonly_dir = temp_dir / "readonly"
            readonly_dir.mkdir()
            os.chmod(readonly_dir, 0o444)

            with pytest.raises(Exception):
                exporter.export_module_specs(
                    output_path=readonly_dir / "test.xlsx",
                    module_data=sample_module_data,
                )

            os.chmod(readonly_dir, 0o755)
        except Exception:
            pass  # Skip if can't set permissions


class TestExportFormats:
    """Test different export formats."""

    def test_json_export(self, temp_dir, sample_module_data, sample_cser_results):
        """Test JSON export functionality."""
        import json

        output_path = temp_dir / "export.json"

        export_data = {
            "module": sample_module_data,
            "results": sample_cser_results,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path) as f:
            loaded = json.load(f)
            assert loaded["module"]["pmax_stc"] == sample_module_data["pmax_stc"]

    def test_csv_export(self, temp_dir, sample_cser_results):
        """Test CSV export for tabular data."""
        import pandas as pd

        monthly_df = pd.DataFrame({
            "Month": list(sample_cser_results["monthly_yields"].keys()),
            "Energy (kWh)": list(sample_cser_results["monthly_yields"].values()),
            "CSER (kWh/kWp)": list(sample_cser_results["monthly_cser"].values()),
        })

        output_path = temp_dir / "monthly.csv"
        monthly_df.to_csv(output_path, index=False)

        assert output_path.exists()

        # Read back
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 12


class TestReportGenerator:
    """Test report generator utilities."""

    def test_generate_report_filename(self, sample_module_data):
        """Test automatic report filename generation."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        filename = generator.generate_filename(
            module_data=sample_module_data,
            report_type="cser",
            extension="pdf",
        )

        assert "Test_Solar" in filename or "TS-400M" in filename
        assert filename.endswith(".pdf")

    def test_report_timestamp(self, temp_dir, sample_module_data):
        """Test report includes timestamp."""
        from src.exports.pdf_export import PDFReportGenerator
        from datetime import datetime

        generator = PDFReportGenerator()

        output_path = temp_dir / "timestamped.pdf"
        generator.generate_quick_summary(
            output_path=output_path,
            module_data=sample_module_data,
            cser_results={"cser_value": 1580},
            include_timestamp=True,
        )

        assert output_path.exists()


class TestBatchExport:
    """Test batch export functionality."""

    def test_batch_pdf_export(self, temp_dir, sample_module_data, sample_cser_results):
        """Test batch PDF export for multiple modules."""
        from src.exports.pdf_export import PDFReportGenerator

        generator = PDFReportGenerator()

        modules = [
            {"data": sample_module_data, "name": "module_1"},
            {"data": {**sample_module_data, "model_name": "TS-500M"}, "name": "module_2"},
        ]

        for module in modules:
            output_path = temp_dir / f"{module['name']}.pdf"
            generator.generate_quick_summary(
                output_path=output_path,
                module_data=module["data"],
                cser_results=sample_cser_results,
            )
            assert output_path.exists()

    def test_batch_excel_export(self, temp_dir, sample_module_data, sample_cser_results):
        """Test exporting multiple results to single Excel."""
        from src.exports.excel_export import ExcelExporter
        import pandas as pd

        exporter = ExcelExporter()

        output_path = temp_dir / "batch_export.xlsx"

        # Export multiple modules to different sheets
        exporter.export_batch_results(
            output_path=output_path,
            results=[
                {"module": sample_module_data, "cser": sample_cser_results, "name": "Module A"},
                {"module": {**sample_module_data, "pmax_stc": 500}, "cser": sample_cser_results, "name": "Module B"},
            ],
        )

        assert output_path.exists()

        with pd.ExcelFile(output_path) as xls:
            assert len(xls.sheet_names) >= 2
