"""Comprehensive Visualization Engine.
Provides a rich gallery of professional interactive Plotly chart generators:
Bar (Grouped/Stacked/Horizontal), Line, Area, Donut/Pie, Treemap, Sunburst,
Scatter, Bubble (4D), Box, Violin, Correlation Heatmap, 2D Density, Pareto (80/20),
Radar/Spider, and Funnel charts.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Universal professional styling helper
def apply_pro_styling(fig: go.Figure, title: str = "", height: int = 500, template: str = "plotly_white") -> go.Figure:
    """Applies modern typography, clean margins, and professional layouts to figures."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.96,
            "x": 0.03,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 18, "family": "Inter, -apple-system, sans-serif", "color": "#0F172A"}
        },
        template=template,
        height=height,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#334155"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.6)"
        ),
        hoverlabel=dict(
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    return fig


class Visualizer:
    """Enterprise visual chart generators across 14+ visualization archetypes."""

    # 1. Bar Chart (Grouped, Stacked, Horizontal, or Ranked)
    @staticmethod
    def bar_chart(
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        barmode: str = "group",
        orientation: str = "v",
        title: Optional[str] = None,
        top_n: Optional[int] = None,
        agg_func: str = "sum"
    ) -> go.Figure:
        plot_df = df.copy()
        if agg_func and x and y:
            group_keys = [x] if not color else [x, color]
            agg_dict = {y: "mean" if agg_func == "mean" else ("count" if agg_func == "count" else "sum")}
            plot_df = plot_df.groupby(group_keys, as_index=False).agg(agg_dict)
            if top_n and not color:
                plot_df = plot_df.sort_values(by=y, ascending=(orientation == "h")).tail(top_n) if orientation == "h" else plot_df.sort_values(by=y, ascending=False).head(top_n)

        title = title or f"Bar Chart: {y} by {x}"
        if orientation == "h":
            fig = px.bar(plot_df, x=y, y=x, color=color, barmode=barmode, orientation="h",
                         color_discrete_sequence=px.colors.qualitative.Prism)
        else:
            fig = px.bar(plot_df, x=x, y=y, color=color, barmode=barmode, orientation="v",
                         color_discrete_sequence=px.colors.qualitative.Prism)
        return apply_pro_styling(fig, title)

    # 2. Line Chart with Markers and Range Selector
    @staticmethod
    def line_chart(
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        markers: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=[x, y]).sort_values(by=x)
        title = title or f"Trend Analysis: {y} Over {x}"
        fig = px.line(plot_df, x=x, y=y, color=color, markers=markers,
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(line=dict(width=2.5))
        return apply_pro_styling(fig, title)

    # 3. Cumulative Area Chart
    @staticmethod
    def area_chart(
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=[x, y]).sort_values(by=x)
        title = title or f"Area Volume: {y} Across {x}"
        fig = px.area(plot_df, x=x, y=y, color=color,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        return apply_pro_styling(fig, title)

    # 4. Donut & Pie Chart (Composition & Market Share)
    @staticmethod
    def donut_chart(
        df: pd.DataFrame,
        names: str,
        values: Optional[str] = None,
        hole: float = 0.45,
        title: Optional[str] = None,
        max_slices: int = 8
    ) -> go.Figure:
        plot_df = df.copy()
        if values:
            grouped = plot_df.groupby(names, as_index=False)[values].sum().sort_values(by=values, ascending=False)
        else:
            grouped = plot_df[names].value_counts().reset_index()
            grouped.columns = [names, "Count"]
            values = "Count"

        # Group smaller slices into 'Other'
        if len(grouped) > max_slices:
            top_part = grouped.head(max_slices - 1)
            other_sum = grouped.iloc[max_slices - 1:][values].sum()
            other_row = pd.DataFrame([{names: "Other", values: other_sum}])
            grouped = pd.concat([top_part, other_row], ignore_index=True)

        title = title or f"Composition Share by {names}"
        fig = px.pie(grouped, names=names, values=values, hole=hole,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return apply_pro_styling(fig, title)

    # 5. Treemap (Hierarchical Proportionality)
    @staticmethod
    def treemap_chart(
        df: pd.DataFrame,
        path: List[str],
        values: str,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=path + [values]).copy()
        plot_df = plot_df[plot_df[values] > 0]  # Treemap values must be positive
        title = title or f"Treemap Hierarchy: {' > '.join(path)} by {values}"
        fig = px.treemap(
            plot_df,
            path=path,
            values=values,
            color=color or values,
            color_continuous_scale="Blues",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        return apply_pro_styling(fig, title)

    # 6. Sunburst (Multi-Level Radial Hierarchy)
    @staticmethod
    def sunburst_chart(
        df: pd.DataFrame,
        path: List[str],
        values: str,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=path + [values]).copy()
        plot_df = plot_df[plot_df[values] > 0]
        title = title or f"Sunburst Decomposition: {' > '.join(path)}"
        fig = px.sunburst(
            plot_df,
            path=path,
            values=values,
            color=color or values,
            color_continuous_scale="Tealgrn"
        )
        return apply_pro_styling(fig, title)

    # 7. Scatter & Multi-Dimensional Bubble Chart (X, Y, Size, Color)
    @staticmethod
    def bubble_chart(
        df: pd.DataFrame,
        x: str,
        y: str,
        size: Optional[str] = None,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        cols_needed = [x, y]
        if size:
            cols_needed.append(size)
        if color:
            cols_needed.append(color)
        plot_df = df.dropna(subset=cols_needed).copy()

        # If size is given, ensure strictly positive
        if size and size in plot_df.columns:
            plot_df[size] = pd.to_numeric(plot_df[size], errors="coerce").fillna(1)
            min_pos = plot_df[plot_df[size] > 0][size].min() if (plot_df[size] > 0).any() else 1
            plot_df[size] = plot_df[size].apply(lambda v: v if v > 0 else min_pos)

        title = title or f"Bubble Relationship: {x} vs {y}" + (f" (Size: {size})" if size else "")
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            size=size,
            color=color,
            opacity=0.75,
            size_max=35,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        return apply_pro_styling(fig, title)

    # 8. Box & Violin Plots (Multi-Category Variance & Distribution)
    @staticmethod
    def box_violin_chart(
        df: pd.DataFrame,
        y: str,
        x: Optional[str] = None,
        chart_type: str = "box",
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=[y]).copy()
        title = title or f"{chart_type.capitalize()} Distribution: {y}" + (f" grouped by {x}" if x else "")
        if chart_type.lower() == "violin":
            fig = px.violin(plot_df, x=x, y=y, color=color or x, box=True, points="outliers",
                            color_discrete_sequence=px.colors.qualitative.Vivid)
        else:
            fig = px.box(plot_df, x=x, y=y, color=color or x, points="outliers",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        return apply_pro_styling(fig, title)

    # 9. Histogram with Marginal Density / Box
    @staticmethod
    def histogram_chart(
        df: pd.DataFrame,
        x: str,
        marginal: str = "box",
        nbins: int = 35,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=[x]).copy()
        title = title or f"Distribution Spread: {x}"
        fig = px.histogram(
            plot_df,
            x=x,
            marginal=marginal,
            nbins=nbins,
            color=color,
            opacity=0.8,
            color_discrete_sequence=["#3B82F6"]
        )
        fig.update_layout(bargap=0.08)
        return apply_pro_styling(fig, title)

    # 10. Pareto Analysis Chart (80/20 Rule)
    @staticmethod
    def pareto_chart(
        df: pd.DataFrame,
        category_col: str,
        value_col: str,
        title: Optional[str] = None
    ) -> go.Figure:
        # Group, sort descending, compute cumulative percentage
        grouped = df.groupby(category_col, as_index=False)[value_col].sum()
        grouped = grouped.sort_values(by=value_col, ascending=False).reset_index(drop=True)
        total_val = grouped[value_col].sum()
        grouped["CumulativePct"] = (grouped[value_col].cumsum() / total_val) * 100

        # Truncate to top 15 categories for clean reading
        grouped = grouped.head(15)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Bars
        fig.add_trace(
            go.Bar(
                x=grouped[category_col],
                y=grouped[value_col],
                name=value_col,
                marker_color="#3B82F6"
            ),
            secondary_y=False
        )
        # Cumulative Line
        fig.add_trace(
            go.Scatter(
                x=grouped[category_col],
                y=grouped["CumulativePct"],
                name="Cumulative %",
                mode="lines+markers",
                marker=dict(color="#EF4444", size=7),
                line=dict(color="#EF4444", width=2.5)
            ),
            secondary_y=True
        )
        # 80% benchmark horizontal line
        fig.add_hline(y=80, line_dash="dash", line_color="#10B981",
                      annotation_text="80% Threshold", secondary_y=True)

        fig.update_yaxes(title_text=f"Total {value_col}", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
        title = title or f"Pareto 80/20 Analysis: {value_col} by {category_col}"
        return apply_pro_styling(fig, title)

    # 11. 2D Density Heatmap / Contour
    @staticmethod
    def density_heatmap_chart(
        df: pd.DataFrame,
        x: str,
        y: str,
        nbinsx: int = 25,
        nbinsy: int = 25,
        title: Optional[str] = None
    ) -> go.Figure:
        plot_df = df.dropna(subset=[x, y]).copy()
        title = title or f"2D Density Concentration: {x} vs {y}"
        fig = px.density_heatmap(
            plot_df,
            x=x,
            y=y,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            color_continuous_scale="Viridis",
            marginal_x="histogram",
            marginal_y="histogram"
        )
        return apply_pro_styling(fig, title)

    # 12. Radar / Spider Benchmark Chart
    @staticmethod
    def radar_chart(
        df: pd.DataFrame,
        category_col: str,
        metric_cols: List[str],
        title: Optional[str] = None
    ) -> go.Figure:
        # Group and take mean of metric columns
        grouped = df.groupby(category_col)[metric_cols].mean().reset_index()
        # Limit to top 5 categories
        grouped = grouped.head(5)

        # Normalize metrics to 0-100 scale for fair radar comparison
        norm_df = grouped.copy()
        for m in metric_cols:
            min_v = norm_df[m].min()
            max_v = norm_df[m].max()
            norm_df[m] = 100 * (norm_df[m] - min_v) / (max_v - min_v) if max_v > min_v else 50.0

        fig = go.Figure()
        for idx, row in norm_df.iterrows():
            r_vals = [row[m] for m in metric_cols]
            r_vals.append(r_vals[0])  # close loop
            theta_vals = metric_cols + [metric_cols[0]]
            fig.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                name=str(row[category_col])
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True
        )
        title = title or "Radar Benchmark Profile (Normalized Metrics)"
        return apply_pro_styling(fig, title)

    # 13. Funnel Chart (Conversion / Drop-off)
    @staticmethod
    def funnel_chart(
        df: pd.DataFrame,
        stage_col: str,
        value_col: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        if value_col:
            grouped = df.groupby(stage_col, as_index=False)[value_col].sum().sort_values(by=value_col, ascending=False)
        else:
            grouped = df[stage_col].value_counts().reset_index()
            grouped.columns = [stage_col, "Count"]
            value_col = "Count"

        title = title or f"Funnel Progression: {stage_col}"
        fig = px.funnel(
            grouped,
            x=value_col,
            y=stage_col,
            color=stage_col,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        return apply_pro_styling(fig, title)
