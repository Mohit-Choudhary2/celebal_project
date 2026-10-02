"""Unit tests for multi-format DataLoader (CSV, Excel, JSON)."""

import unittest
import pandas as pd
from backend.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def test_load_csv(self):
        df, meta = DataLoader.load_file("sample_datasets/customer_churn.csv", "customer_churn.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 100)
        self.assertEqual(meta["format"], "CSV")
        # Ensure column whitespace was stripped
        self.assertIn("CustomerID", df.columns)
        self.assertNotIn(" CustomerID ", df.columns)

    def test_load_excel_sheets(self):
        df, meta = DataLoader.load_file("sample_datasets/sales_performance.xlsx", "sales_performance.xlsx")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(meta["format"], "Excel")
        self.assertIn("Daily_Sales", meta["sheets"])
        self.assertIn("Regional_Targets", meta["sheets"])

    def test_load_json(self):
        df, meta = DataLoader.load_file("sample_datasets/web_analytics.json", "web_analytics.json")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(meta["format"], "JSON")
        self.assertIn("channel", df.columns)
        self.assertIn("session_id", df.columns)


if __name__ == "__main__":
    unittest.main()
