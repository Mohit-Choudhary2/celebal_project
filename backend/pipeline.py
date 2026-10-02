"""Autonomous End-to-End Data Analysis Pipeline.
Automates ingestion, data cleaning, statistical pattern discovery,
professional chart generation, and executive insights extraction.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend.data_profiler import DataProfiler
from backend.visualizer import Visualizer


class AnalysisPipeline:
    """Orchestrates the entire automated data analysis lifecycle."""

    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Automated smart data cleaning and standardization.

        Returns (cleaned_df, cleaning_actions_log).
        """
        cleaned = df.copy()
        log = []

        # 1. Strip column whitespace
        orig_cols = list(cleaned.columns)
        cleaned.columns = [str(c).strip() for c in cleaned.columns]
        if orig_cols != list(cleaned.columns):
            log.append("Sanitized column names: trimmed whitespace and invisible characters.")

        # 2. De-duplicate rows
        dup_count = int(cleaned.duplicated().sum())
        if dup_count > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            log.append(f"Removed {dup_count:,} duplicate rows to ensure sample uniqueness.")

        # 3. Handle Datetime Conversions
        for col in cleaned.columns:
            if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
                sample = cleaned[col].dropna().head(30)
                if len(sample) >= 5:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notnull().mean() > 0.8:
                                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                                log.append(f"Parsed and converted '{col}' to standardized ISO datetime format.")
                        except Exception:
                            pass

        # 4. Handle Numeric Column Formatting (e.g. '$1,000' or percentages)
        for col in cleaned.columns:
            if pd.api.types.is_object_dtype(cleaned[col]):
                sample_str = cleaned[col].dropna().astype(str).head(20)
                if any(sample_str.str.contains(r"^[\$€£]|%$", regex=True)):
                    cleaned[col] = (
                        cleaned[col]
                        .astype(str)
                        .str.replace(r"[\$,€£%]", "", regex=True)
                        .str.strip()
                    )
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    log.append(f"Cleaned currency/percentage formatting in '{col}' and cast to float.")

        # 5. Missing Value Imputation
        for col in cleaned.columns:
            null_count = int(cleaned[col].isnull().sum())
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    median_val = float(cleaned[col].median())
                    cleaned[col] = cleaned[col].fillna(median_val)
                    log.append(f"Imputed {null_count:,} missing values in '{col}' with median ({median_val:,.2f}).")
                elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                    # Forward-fill / back-fill
                    cleaned[col] = cleaned[col].ffill().bfill()
                    log.append(f"Interpolated {null_count:,} missing timestamps in '{col}' using forward/backward fill.")
                else:
                    mode_val = cleaned[col].mode()
                    fill_val = str(mode_val[0]) if len(mode_val) > 0 else "Unknown"
                    cleaned[col] = cleaned[col].fillna(fill_val)
                    log.append(f"Imputed {null_count:,} missing categories in '{col}' with mode ('{fill_val}').")

        if not log:
            log.append("Dataset is already pristine. No cleaning transformations were required.")

        return cleaned, log

    @staticmethod
    def generate_automated_charts(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generates a comprehensive suite of professional interactive Plotly charts across 10+ visualization archetypes."""
        charts = []
        numeric_cols = profile.get("numeric_cols", [])
        categorical_cols = profile.get("categorical_cols", [])
        datetime_cols = profile.get("datetime_cols", [])

        # 1. Temporal Trend (Line Chart)
        if datetime_cols and numeric_cols:
            date_col = datetime_cols[0]
            val_col = numeric_cols[0]
            try:
                temp_df = df.dropna(subset=[date_col, val_col]).copy()
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                temp_df = temp_df.sort_values(by=date_col)
                daily_trend = temp_df.groupby(temp_df[date_col].dt.date, as_index=False)[val_col].sum()
                daily_trend.columns = ["Date", f"Total_{val_col}"]

                fig_time = Visualizer.line_chart(
                    daily_trend, x="Date", y=f"Total_{val_col}",
                    title=f"📈 Temporal Trend: {val_col} Evolution Over Time"
                )
                charts.append({
                    "id": "time_series_trend",
                    "category": "Trends & Evolution",
                    "type": "Line Chart",
                    "title": f"Temporal Trend: {val_col} Over Time",
                    "description": f"Tracks how aggregate {val_col} evolves across the historical timeline.",
                    "figure": fig_time
                })

                # 2. Cumulative Area Chart
                fig_area = Visualizer.area_chart(
                    daily_trend, x="Date", y=f"Total_{val_col}",
                    title=f"🌊 Cumulative Volume: {val_col} Over Time"
                )
                charts.append({
                    "id": "cumulative_area",
                    "category": "Trends & Evolution",
                    "type": "Area Chart",
                    "title": f"Cumulative Volume: {val_col}",
                    "description": f"Displays cumulative growth and area density of {val_col}.",
                    "figure": fig_area
                })
            except Exception:
                pass

        # 3. Categorical Breakdown (Ranked Bar Chart)
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            try:
                fig_bar = Visualizer.bar_chart(
                    df, x=cat_col, y=num_col, agg_func="mean", top_n=10,
                    title=f"🏆 Segment Performance: Average {num_col} by {cat_col}"
                )
                charts.append({
                    "id": "category_breakdown",
                    "category": "Ranking & Comparison",
                    "type": "Bar Chart",
                    "title": f"Segment Performance: {num_col} by {cat_col}",
                    "description": f"Compares and ranks top {cat_col} groups by average {num_col}.",
                    "figure": fig_bar
                })
            except Exception:
                pass

        # 4. Pareto 80/20 Analysis Chart
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            try:
                fig_pareto = Visualizer.pareto_chart(
                    df, category_col=cat_col, value_col=num_col,
                    title=f"⚖️ Pareto 80/20 Analysis: {num_col} Concentration by {cat_col}"
                )
                charts.append({
                    "id": "pareto_analysis",
                    "category": "Ranking & Comparison",
                    "type": "Pareto Chart",
                    "title": f"Pareto 80/20 Analysis: {cat_col}",
                    "description": "Visualizes the 80/20 rule to pinpoint key driver categories accounting for majority volume.",
                    "figure": fig_pareto
                })
            except Exception:
                pass

        # 5. Composition / Market Share (Donut Chart)
        if categorical_cols:
            cat_col = categorical_cols[0]
            val_col = numeric_cols[0] if numeric_cols else None
            try:
                fig_donut = Visualizer.donut_chart(
                    df, names=cat_col, values=val_col,
                    title=f"🍩 Category Share: Composition by {cat_col}"
                )
                charts.append({
                    "id": "category_composition_donut",
                    "category": "Composition & Hierarchy",
                    "type": "Donut Chart",
                    "title": f"Composition Share: {cat_col}",
                    "description": f"Proportional market share and distribution percentages across {cat_col}.",
                    "figure": fig_donut
                })
            except Exception:
                pass

        # 6. Hierarchical Treemap
        if len(categorical_cols) >= 2 and numeric_cols:
            path_cols = categorical_cols[:2]
            val_col = numeric_cols[0]
            try:
                fig_tree = Visualizer.treemap_chart(
                    df, path=path_cols, values=val_col,
                    title=f"🗺️ Hierarchical Treemap: {' > '.join(path_cols)} by {val_col}"
                )
                charts.append({
                    "id": "treemap_hierarchy",
                    "category": "Composition & Hierarchy",
                    "type": "Treemap",
                    "title": f"Treemap Hierarchy: {' > '.join(path_cols)}",
                    "description": f"Nested rectangular decomposition of volume across multiple categorical tiers.",
                    "figure": fig_tree
                })
            except Exception:
                pass

        # 7. Distribution Spread (Histogram with Marginal Box)
        if numeric_cols:
            primary_num = numeric_cols[0]
            try:
                fig_hist = Visualizer.histogram_chart(
                    df, x=primary_num, marginal="box",
                    title=f"📊 Distribution & Outlier Spread: {primary_num}"
                )
                charts.append({
                    "id": "distribution_primary",
                    "category": "Distribution & Outliers",
                    "type": "Histogram & Boxplot",
                    "title": f"Distribution Analysis: {primary_num}",
                    "description": f"Displays frequency spread, skewness, and IQR outlier boundaries of {primary_num}.",
                    "figure": fig_hist
                })
            except Exception:
                pass

        # 8. Multi-Category Variance (Violin / Box Plot)
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            try:
                fig_violin = Visualizer.box_violin_chart(
                    df, x=cat_col, y=num_col, chart_type="violin",
                    title=f"🎻 Variance & Density: {num_col} Across {cat_col}"
                )
                charts.append({
                    "id": "variance_box_violin",
                    "category": "Distribution & Outliers",
                    "type": "Violin Plot",
                    "title": f"Variance Comparison: {num_col} by {cat_col}",
                    "description": f"Compares probability density, median, and interquartile spread across {cat_col}.",
                    "figure": fig_violin
                })
            except Exception:
                pass

        # 9. Multi-Dimensional Relationship (Bubble Chart)
        if len(numeric_cols) >= 3:
            c1 = numeric_cols[0]
            c2 = numeric_cols[1]
            c3 = numeric_cols[2]
            c_cat = categorical_cols[0] if categorical_cols else None
            try:
                fig_bubble = Visualizer.bubble_chart(
                    df, x=c1, y=c2, size=c3, color=c_cat,
                    title=f"🔮 4D Bubble Chart: {c1} vs {c2} (Size: {c3})"
                )
                charts.append({
                    "id": "multi_dim_bubble",
                    "category": "Relationships & Correlation",
                    "type": "Bubble Chart",
                    "title": f"Multi-Dimensional Relationship: {c1} vs {c2}",
                    "description": f"Evaluates multi-variable dynamics mapping {c1} (X), {c2} (Y), {c3} (Bubble Size), and categories.",
                    "figure": fig_bubble
                })
            except Exception:
                pass

        # 10. Bivariate Relationship (Scatter Plot)
        if len(numeric_cols) >= 2:
            c1 = numeric_cols[0]
            c2 = numeric_cols[1]
            c_cat = categorical_cols[0] if (categorical_cols and df[categorical_cols[0]].nunique() <= 8) else None
            try:
                fig_scatter = px.scatter(
                    df, x=c1, y=c2, color=c_cat,
                    title=f"🔍 Bivariate Relationship: {c1} vs {c2}",
                    template="plotly_white", opacity=0.75
                )
                charts.append({
                    "id": "bivariate_relationship",
                    "category": "Relationships & Correlation",
                    "type": "Scatter Plot",
                    "title": f"Bivariate Relationship: {c1} vs {c2}",
                    "description": f"Scatter correlation revealing clustering patterns between {c1} and {c2}.",
                    "figure": fig_scatter
                })
            except Exception:
                pass

        # 11. Pearson Correlation Heatmap
        if len(numeric_cols) >= 3:
            try:
                corr_cols = numeric_cols[:8]
                corr_df = df[corr_cols].corr().round(2)
                fig_heatmap = px.imshow(
                    corr_df, text_auto=True, aspect="auto",
                    title="🔥 Pearson Correlation Matrix",
                    template="plotly_white", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                charts.append({
                    "id": "correlation_matrix",
                    "category": "Relationships & Correlation",
                    "type": "Correlation Heatmap",
                    "title": "Correlation Matrix",
                    "description": "Multi-variable correlation matrix uncovering linear dependencies.",
                    "figure": fig_heatmap
                })
            except Exception:
                pass

        # 12. 2D Density Heatmap
        if len(numeric_cols) >= 2:
            c1 = numeric_cols[0]
            c2 = numeric_cols[1]
            try:
                fig_dens = Visualizer.density_heatmap_chart(
                    df, x=c1, y=c2,
                    title=f"🌡️ 2D Density Concentration: {c1} vs {c2}"
                )
                charts.append({
                    "id": "density_heatmap",
                    "category": "Relationships & Correlation",
                    "type": "2D Density Heatmap",
                    "title": f"2D Density Concentration: {c1} vs {c2}",
                    "description": f"Identifies data point clustering and peak concentration between {c1} and {c2}.",
                    "figure": fig_dens
                })
            except Exception:
                pass

        return charts

    @staticmethod
    def generate_executive_insights(df: pd.DataFrame, profile: Dict[str, Any], cleaning_log: List[str]) -> Dict[str, Any]:
        """Synthesizes high-level business intelligence, strategic takeaways, and recommendations."""
        total_rows = profile.get("total_rows", len(df))
        total_cols = profile.get("total_cols", len(df.columns))
        health_score = profile.get("health_score", 100)
        correlations = profile.get("correlations", [])
        num_cols = profile.get("numeric_cols", [])
        cat_cols = profile.get("categorical_cols", [])

        # Executive Takeaways
        takeaways = []
        takeaways.append(f"**Data Foundation**: Successfully ingested and validated **{total_rows:,} records** across **{total_cols} dimensions** with a data health index of **{health_score}/100**.")

        if correlations:
            top_c = correlations[0]
            takeaways.append(f"**Key Driver Identified**: Strongest relationship discovered between **{top_c['col1']}** and **{top_c['col2']}** (r = {top_c['correlation']}). Fluctuations in {top_c['col1']} strongly impact {top_c['col2']}.")

        if cat_cols and num_cols:
            top_cat = cat_cols[0]
            top_num = num_cols[0]
            grouped = df.groupby(top_cat)[top_num].mean()
            best_cat = grouped.idxmax()
            best_val = grouped.max()
            lowest_cat = grouped.idxmin()
            lowest_val = grouped.min()
            spread_ratio = round(best_val / lowest_val, 1) if lowest_val > 0 else "significant"
            takeaways.append(f"**Segment Variance**: In `{top_cat}`, top performer **'{best_cat}'** averages **{best_val:,.2f}** {top_num}, outperforming '{lowest_cat}' by **{spread_ratio}x**.")

        # Outlier risk
        outlier_cols = [c for c, d in profile.get("col_details", {}).items() if d.get("outliers_count", 0) > 0]
        if outlier_cols:
            takeaways.append(f"**Anomalies & Extreme Outliers**: Detected statistical outliers in `{', '.join(outlier_cols[:3])}`. These observations warrant targeted investigation for fraud or special-case dynamics.")

        # Recommendations
        recommendations = [
            f"Leverage high-performing `{best_cat if (cat_cols and num_cols) else 'segments'}` as operational benchmarks across weaker cohorts.",
            "Establish continuous automated monitoring on key drivers to detect early performance drift.",
            "Integrate cleaned data outputs into downstream reporting workflows for recurring executive dashboards."
        ]

        return {
            "summary_headline": f"Automated Pipeline Insights: {total_rows:,} Records Analyzed",
            "health_score": health_score,
            "takeaways": takeaways,
            "recommendations": recommendations,
            "cleaning_actions": cleaning_log
        }

    @classmethod
    def run_full_pipeline(
        cls,
        df: pd.DataFrame,
        on_progress: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """Runs the entire automated 5-stage pipeline from raw data to executive delivery."""
        def step(msg: str, pct: int):
            if on_progress:
                on_progress(msg, pct)

        step("Ingesting raw data and mapping data types...", 15)
        raw_profile = DataProfiler.profile_dataset(df)

        step("Applying intelligent automated data cleaning...", 35)
        cleaned_df, cleaning_log = cls.clean_dataset(df)

        step("Extracting statistical patterns, correlations, and anomalies...", 60)
        cleaned_profile = DataProfiler.profile_dataset(cleaned_df)

        step("Generating professional interactive visual dashboards...", 80)
        charts = cls.generate_automated_charts(cleaned_df, cleaned_profile)

        step("Synthesizing strategic executive insights and takeaways...", 95)
        insights = cls.generate_executive_insights(cleaned_df, cleaned_profile, cleaning_log)

        step("Analysis complete!", 100)

        return {
            "raw_df": df,
            "cleaned_df": cleaned_df,
            "raw_profile": raw_profile,
            "profile": cleaned_profile,
            "cleaning_log": cleaning_log,
            "charts": charts,
            "insights": insights
        }
