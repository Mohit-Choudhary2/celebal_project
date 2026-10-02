"""Executive Report Generator.
Generates standalone, self-contained HTML and Markdown reports
with embedded interactive Plotly charts, executive summaries, and data profiles.
"""

from typing import Dict, Any, List
import plotly.io as pio


class ReportGenerator:
    """Exports polished data analysis reports in HTML and Markdown formats."""

    @staticmethod
    def generate_html_report(pipeline_results: Dict[str, Any], title: str = "Executive Data Analysis Report") -> str:
        """Builds a responsive, modern HTML report with embedded interactive Plotly charts."""
        profile = pipeline_results.get("profile", {})
        insights = pipeline_results.get("insights", {})
        charts = pipeline_results.get("charts", [])
        cleaning_log = pipeline_results.get("cleaning_log", [])

        # Generate HTML divs for each chart
        chart_divs = []
        for c in charts:
            fig = c.get("figure")
            if fig:
                chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
                chart_divs.append(f"""
                <div class="chart-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="margin: 0;">{c.get('title', 'Chart')}</h3>
                        <span style="background: #EFF6FF; color: #1D4ED8; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 9999px;">{c.get('type', 'Visualization')}</span>
                    </div>
                    <p class="chart-desc">{c.get('description', '')}</p>
                    <div class="plot-container">{chart_html}</div>
                </div>
                """)

        chart_section_html = "\n".join(chart_divs) if chart_divs else "<p>No charts generated.</p>"

        # Takeaways list
        takeaways_html = "\n".join([f"<li>{t}</li>" for t in insights.get("takeaways", [])])

        # Recommendations list
        recs_html = "\n".join([f"<li>{r}</li>" for r in insights.get("recommendations", [])])

        # Cleaning actions list
        cleaning_html = "\n".join([f"<li>{c}</li>" for c in cleaning_log])

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #2563eb;
            --primary-light: #eff6ff;
            --surface: #ffffff;
            --bg: #f8fafc;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
            --accent: #6366f1;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 2.2rem; font-weight: 700; margin-bottom: 10px; }}
        .header p {{ color: #94a3b8; font-size: 1.1rem; }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--surface);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .kpi-label {{ font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); }}
        .kpi-value {{ font-size: 2rem; font-weight: 700; color: var(--primary); margin-top: 6px; }}
        
        .section-card {{
            background: var(--surface);
            padding: 30px;
            border-radius: 14px;
            border: 1px solid var(--border);
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .section-card h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 18px;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 10px;
        }}
        ul.insights-list {{
            list-style: none;
        }}
        ul.insights-list li {{
            position: relative;
            padding-left: 28px;
            margin-bottom: 12px;
            font-size: 1.05rem;
        }}
        ul.insights-list li::before {{
            content: "✓";
            position: absolute;
            left: 0;
            top: 2px;
            color: var(--success);
            font-weight: bold;
        }}
        
        .chart-card {{
            background: var(--surface);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 25px;
        }}
        .chart-card h3 {{ font-size: 1.2rem; margin-bottom: 6px; }}
        .chart-desc {{ color: var(--text-muted); font-size: 0.95rem; margin-bottom: 15px; }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p>Generated automatically by Autonomous Self-Healing Data Analyst Agent</p>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Analyzed Rows</div>
                <div class="kpi-value">{profile.get('total_rows', 0):,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Features / Columns</div>
                <div class="kpi-value">{profile.get('total_cols', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Data Health Score</div>
                <div class="kpi-value">{profile.get('health_score', 100)}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Charts Generated</div>
                <div class="kpi-value">{len(charts)}</div>
            </div>
        </div>

        <div class="section-card">
            <h2>💡 Executive Findings & Discoveries</h2>
            <ul class="insights-list">
                {takeaways_html}
            </ul>
        </div>

        <div class="section-card">
            <h2>🎯 Recommended Strategic Actions</h2>
            <ul class="insights-list">
                {recs_html}
            </ul>
        </div>

        <div class="section-card">
            <h2>📈 Interactive Visual Analytics</h2>
            {chart_section_html}
        </div>

        <div class="section-card">
            <h2>🧹 Automated Data Cleansing Audit</h2>
            <ul class="insights-list">
                {cleaning_html}
            </ul>
        </div>

        <div class="footer">
            <p>Produced autonomously by Self-Healing RAG Data Analysis System</p>
        </div>
    </div>
</body>
</html>"""
        return html_template

    @staticmethod
    def generate_markdown_report(pipeline_results: Dict[str, Any]) -> str:
        """Generates a portable Markdown summary of the analysis."""
        profile = pipeline_results.get("profile", {})
        insights = pipeline_results.get("insights", {})
        cleaning_log = pipeline_results.get("cleaning_log", [])

        md_lines = [
            "# Executive Data Analysis Report",
            "",
            f"**Total Records:** {profile.get('total_rows', 0):,}  ",
            f"**Columns:** {profile.get('total_cols', 0)}  ",
            f"**Data Quality Score:** {profile.get('health_score', 100)}/100  ",
            "",
            "## 💡 Key Executive Discoveries",
            ""
        ]
        for t in insights.get("takeaways", []):
            md_lines.append(f"- {t}")

        md_lines.extend([
            "",
            "## 🎯 Strategic Recommendations",
            ""
        ])
        for r in insights.get("recommendations", []):
            md_lines.append(f"- {r}")

        md_lines.extend([
            "",
            "## 🧹 Automated Data Cleansing Actions",
            ""
        ])
        for c in cleaning_log:
            md_lines.append(f"- {c}")

        return "\n".join(md_lines)
