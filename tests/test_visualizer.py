"""Unit tests for the Visualizer engine across all chart archetypes."""

import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backend.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 25,
            "category": ["Electronics", "Furniture", "Clothing", "Toys"] * 25,
            "sales": np.random.uniform(100, 1000, 100),
            "profit": np.random.uniform(10, 300, 100),
            "units": np.random.randint(1, 50, 100),
            "date": pd.date_range("2024-01-01", periods=100)
        })

    def test_bar_chart(self):
        fig = Visualizer.bar_chart(self.df, x="region", y="sales", agg_func="sum")
        self.assertIsInstance(fig, go.Figure)

    def test_line_chart(self):
        fig = Visualizer.line_chart(self.df, x="date", y="sales")
        self.assertIsInstance(fig, go.Figure)

    def test_area_chart(self):
        fig = Visualizer.area_chart(self.df, x="date", y="sales")
        self.assertIsInstance(fig, go.Figure)

    def test_donut_chart(self):
        fig = Visualizer.donut_chart(self.df, names="region", values="sales")
        self.assertIsInstance(fig, go.Figure)

    def test_treemap_chart(self):
        fig = Visualizer.treemap_chart(self.df, path=["region", "category"], values="sales")
        self.assertIsInstance(fig, go.Figure)

    def test_sunburst_chart(self):
        fig = Visualizer.sunburst_chart(self.df, path=["region", "category"], values="sales")
        self.assertIsInstance(fig, go.Figure)

    def test_bubble_chart(self):
        fig = Visualizer.bubble_chart(self.df, x="sales", y="profit", size="units", color="region")
        self.assertIsInstance(fig, go.Figure)

    def test_box_violin_chart(self):
        fig_box = Visualizer.box_violin_chart(self.df, x="region", y="sales", chart_type="box")
        fig_violin = Visualizer.box_violin_chart(self.df, x="region", y="sales", chart_type="violin")
        self.assertIsInstance(fig_box, go.Figure)
        self.assertIsInstance(fig_violin, go.Figure)

    def test_histogram_chart(self):
        fig = Visualizer.histogram_chart(self.df, x="sales", marginal="box")
        self.assertIsInstance(fig, go.Figure)

    def test_pareto_chart(self):
        fig = Visualizer.pareto_chart(self.df, category_col="category", value_col="sales")
        self.assertIsInstance(fig, go.Figure)

    def test_density_heatmap_chart(self):
        fig = Visualizer.density_heatmap_chart(self.df, x="sales", y="profit")
        self.assertIsInstance(fig, go.Figure)

    def test_radar_chart(self):
        fig = Visualizer.radar_chart(self.df, category_col="region", metric_cols=["sales", "profit", "units"])
        self.assertIsInstance(fig, go.Figure)

    def test_funnel_chart(self):
        fig = Visualizer.funnel_chart(self.df, stage_col="region", value_col="sales")
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
