"""Unit and integration tests for the automated end-to-end pipeline."""

import unittest
import pandas as pd
from backend.data_loader import DataLoader
from backend.pipeline import AnalysisPipeline
from backend.report_generator import ReportGenerator


class TestPipeline(unittest.TestCase):
    def test_end_to_end_pipeline_churn(self):
        df, meta = DataLoader.load_file("sample_datasets/customer_churn.csv", "customer_churn.csv")
        results = AnalysisPipeline.run_full_pipeline(df)

        self.assertIn("cleaned_df", results)
        self.assertIn("profile", results)
        self.assertIn("charts", results)
        self.assertIn("insights", results)

        # Ensure charts generated
        self.assertGreaterEqual(len(results["charts"]), 2)

        # Ensure cleaning took place
        self.assertGreater(len(results["cleaning_log"]), 0)

        # Test report generation
        html_report = ReportGenerator.generate_html_report(results)
        self.assertIn("Executive Data Analysis Report", html_report)
        self.assertIn("plotly", html_report.lower())

        md_report = ReportGenerator.generate_markdown_report(results)
        self.assertIn("# Executive Data Analysis Report", md_report)


if __name__ == "__main__":
    unittest.main()
