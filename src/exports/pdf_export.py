"""
PDF report generation for PV-CSER Pro.

Generates comprehensive PDF reports for CSER analysis results.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFReportGenerator:
    """
    PDF report generator for CSER analysis.

    Creates professional PDF reports with module specifications,
    calculation results, and visualizations.
    """

    # Color scheme
    HEADER_COLOR = colors.HexColor("#1E3A5F")
    ACCENT_COLOR = colors.HexColor("#FF6B35")
    TABLE_HEADER_BG = colors.HexColor("#2C5282")
    TABLE_ALT_BG = colors.HexColor("#F7FAFC")

    def __init__(
        self,
        title: str = "PV-CSER Pro Analysis Report",
        page_size: tuple = A4,
    ):
        """
        Initialize PDF report generator.

        Args:
            title: Report title
            page_size: Page size (A4 or letter)
        """
        self.title = title
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Set up custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.HEADER_COLOR,
            spaceAfter=30,
            alignment=1,  # Center
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.HEADER_COLOR,
            spaceBefore=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=self.ACCENT_COLOR,
            borderPadding=5,
        ))

        # Normal text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
        ))

        # Table header
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.white,
            alignment=1,
        ))

    def generate_report(
        self,
        module_data: Dict[str, Any],
        cser_results: Dict[str, Any],
        power_matrix_data: Optional[Dict] = None,
        charts: Optional[List[BytesIO]] = None,
    ) -> BytesIO:
        """
        Generate complete PDF report.

        Args:
            module_data: Module specification data
            cser_results: CSER calculation results
            power_matrix_data: Power matrix data (optional)
            charts: List of chart images as BytesIO (optional)

        Returns:
            BytesIO containing PDF data
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=1*cm,
            leftMargin=1*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm,
        )

        # Build content
        story = []

        # Title page
        story.extend(self._create_title_page(module_data))

        # Module specifications section
        story.extend(self._create_module_section(module_data))

        # CSER results section
        story.extend(self._create_cser_section(cser_results))

        # Power matrix section
        if power_matrix_data:
            story.extend(self._create_power_matrix_section(power_matrix_data))

        # Charts section
        if charts:
            story.extend(self._create_charts_section(charts))

        # Summary section
        story.extend(self._create_summary_section(module_data, cser_results))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    def _create_title_page(self, module_data: Dict[str, Any]) -> List:
        """Create title page content."""
        content = []

        # Title
        content.append(Spacer(1, 2*inch))
        content.append(Paragraph(self.title, self.styles['CustomTitle']))
        content.append(Spacer(1, 0.5*inch))

        # Subtitle with module info
        module_name = f"{module_data.get('manufacturer', 'N/A')} {module_data.get('model_name', 'N/A')}"
        content.append(Paragraph(
            f"<b>Module:</b> {module_name}",
            self.styles['Heading3']
        ))
        content.append(Spacer(1, 0.3*inch))

        # Date
        content.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles['Normal']
        ))

        content.append(PageBreak())
        return content

    def _create_module_section(self, module_data: Dict[str, Any]) -> List:
        """Create module specifications section."""
        content = []

        content.append(Paragraph("Module Specifications", self.styles['SectionHeader']))

        # Create specifications table
        specs = [
            ["Parameter", "Value", "Unit"],
            ["Manufacturer", str(module_data.get('manufacturer', 'N/A')), ""],
            ["Model", str(module_data.get('model_name', 'N/A')), ""],
            ["Pmax (STC)", f"{module_data.get('pmax_stc', 0):.1f}", "W"],
            ["Voc (STC)", f"{module_data.get('voc_stc', 0):.2f}", "V"],
            ["Isc (STC)", f"{module_data.get('isc_stc', 0):.2f}", "A"],
            ["Vmp (STC)", f"{module_data.get('vmp_stc', 0):.2f}", "V"],
            ["Imp (STC)", f"{module_data.get('imp_stc', 0):.2f}", "A"],
            ["Temp. Coeff. (Pmax)", f"{module_data.get('temp_coeff_pmax', -0.35):.2f}", "%/°C"],
            ["NMOT", f"{module_data.get('nmot', 45):.0f}", "°C"],
            ["Module Area", f"{module_data.get('module_area', 0):.2f}", "m²"],
            ["Cell Type", str(module_data.get('cell_type', 'N/A')), ""],
        ]

        table = Table(specs, colWidths=[2.5*inch, 2*inch, 1*inch])
        table.setStyle(self._get_table_style())
        content.append(table)
        content.append(Spacer(1, 0.3*inch))

        return content

    def _create_cser_section(self, cser_results: Dict[str, Any]) -> List:
        """Create CSER results section."""
        content = []

        content.append(Paragraph("CSER Calculation Results", self.styles['SectionHeader']))

        # Main results
        results_data = [
            ["Metric", "Value"],
            ["CSER", f"{cser_results.get('cser', 0):.0f} kWh/kWp"],
            ["Annual Energy", f"{cser_results.get('annual_energy', 0):.1f} kWh"],
            ["Performance Ratio", f"{cser_results.get('performance_ratio', 0):.1f}%"],
            ["Temperature Loss", f"{cser_results.get('temperature_loss', 0):.1f}%"],
            ["Climate Profile", str(cser_results.get('climate_profile', 'N/A'))],
            ["Annual Irradiation", f"{cser_results.get('annual_irradiation', 0):.0f} kWh/m²"],
        ]

        table = Table(results_data, colWidths=[3*inch, 2.5*inch])
        table.setStyle(self._get_table_style())
        content.append(table)
        content.append(Spacer(1, 0.3*inch))

        # Monthly breakdown if available
        monthly = cser_results.get('monthly_energy', [])
        if monthly:
            content.append(Paragraph("Monthly Energy Yield", self.styles['Heading3']))

            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            monthly_data = [["Month", "Energy (kWh)", "Yield (kWh/kWp)"]]
            pmax_kw = cser_results.get('module_pmax', 1000) / 1000

            for i, (month, energy) in enumerate(zip(months, monthly)):
                monthly_data.append([
                    month,
                    f"{energy:.1f}",
                    f"{energy/pmax_kw:.1f}" if pmax_kw > 0 else "N/A"
                ])

            table = Table(monthly_data, colWidths=[1.5*inch, 2*inch, 2*inch])
            table.setStyle(self._get_table_style())
            content.append(table)

        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_power_matrix_section(self, power_matrix_data: Dict) -> List:
        """Create power matrix section."""
        content = []

        content.append(Paragraph("Power Matrix Data", self.styles['SectionHeader']))

        irr = power_matrix_data.get('irradiance', [])
        temp = power_matrix_data.get('temperature', [])
        power = power_matrix_data.get('power_matrix', [[]])

        if len(irr) > 0 and len(temp) > 0:
            # Create header row
            header = ["G \\ T"] + [f"{t}°C" for t in temp]

            # Create data rows
            table_data = [header]
            for i, g in enumerate(irr):
                row = [f"{g} W/m²"]
                for j in range(len(temp)):
                    if i < len(power) and j < len(power[i]):
                        row.append(f"{power[i][j]:.1f}")
                    else:
                        row.append("N/A")
                table_data.append(row)

            table = Table(table_data)
            table.setStyle(self._get_table_style(header_rows=1))
            content.append(table)

        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_charts_section(self, charts: List[BytesIO]) -> List:
        """Create charts section."""
        content = []

        content.append(PageBreak())
        content.append(Paragraph("Visualization Charts", self.styles['SectionHeader']))

        for i, chart_buffer in enumerate(charts):
            try:
                chart_buffer.seek(0)
                img = Image(chart_buffer, width=6*inch, height=4*inch)
                content.append(img)
                content.append(Spacer(1, 0.5*inch))
            except Exception:
                content.append(Paragraph(f"Chart {i+1} could not be loaded", self.styles['Normal']))

        return content

    def _create_summary_section(
        self,
        module_data: Dict[str, Any],
        cser_results: Dict[str, Any],
    ) -> List:
        """Create summary section."""
        content = []

        content.append(Paragraph("Summary", self.styles['SectionHeader']))

        summary_text = f"""
        This report presents the Climate Specific Energy Rating (CSER) analysis for the
        {module_data.get('manufacturer', 'N/A')} {module_data.get('model_name', 'N/A')}
        PV module. The analysis was performed according to IEC 61853 standards.

        Key findings:
        - The module achieved a CSER of {cser_results.get('cser', 0):.0f} kWh/kWp under the
          {cser_results.get('climate_profile', 'selected')} climate profile.
        - Annual energy production is estimated at {cser_results.get('annual_energy', 0):.1f} kWh.
        - The overall performance ratio is {cser_results.get('performance_ratio', 0):.1f}%.
        - Temperature-related losses account for approximately {cser_results.get('temperature_loss', 0):.1f}%
          of total losses.

        This analysis helps in comparing module performance across different climate zones
        and supports informed decision-making for PV system design.
        """

        content.append(Paragraph(summary_text, self.styles['CustomBody']))

        # Disclaimer
        content.append(Spacer(1, 0.5*inch))
        disclaimer = """
        <i>Disclaimer: This report is generated based on the provided input data and standard
        calculation methodologies. Actual field performance may vary due to installation conditions,
        shading, soiling, and other environmental factors.</i>
        """
        content.append(Paragraph(disclaimer, self.styles['Normal']))

        return content

    def _get_table_style(self, header_rows: int = 1) -> TableStyle:
        """Get standard table style."""
        style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, header_rows-1), self.TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, header_rows-1), colors.white),
            ('FONTNAME', (0, 0), (-1, header_rows-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, header_rows-1), 10),
            ('ALIGN', (0, 0), (-1, header_rows-1), 'CENTER'),

            # Body styling
            ('FONTNAME', (0, header_rows), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, header_rows), (-1, -1), 9),
            ('ALIGN', (1, header_rows), (-1, -1), 'CENTER'),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, self.HEADER_COLOR),

            # Alternating rows
            ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [colors.white, self.TABLE_ALT_BG]),

            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
        return style

    def generate_quick_summary(
        self,
        module_name: str,
        cser_value: float,
        climate_profile: str,
    ) -> BytesIO:
        """
        Generate a quick one-page summary.

        Args:
            module_name: Module name
            cser_value: CSER value
            climate_profile: Climate profile name

        Returns:
            BytesIO containing PDF data
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=self.page_size)

        story = []

        story.append(Paragraph("PV-CSER Pro Quick Summary", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))

        summary_data = [
            ["Module", module_name],
            ["Climate Profile", climate_profile],
            ["CSER Value", f"{cser_value:.0f} kWh/kWp"],
            ["Generated", datetime.now().strftime('%Y-%m-%d')],
        ]

        table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
        table.setStyle(self._get_table_style(header_rows=0))
        story.append(table)

        doc.build(story)
        buffer.seek(0)
        return buffer
