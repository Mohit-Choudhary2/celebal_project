"""Unit tests for CodeSandbox execution and error diagnostics."""

import unittest
import pandas as pd
from backend.code_sandbox import CodeSandbox


class TestCodeSandbox(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "product": ["A", "B", "C", "A", "B"],
            "revenue": [100.0, 150.0, 200.0, 120.0, 180.0]
        })

    def test_successful_execution_and_figure_capture(self):
        code = """
import plotly.express as px
fig = px.bar(df, x='product', y='revenue', title='Revenue by Product')
mean_rev = df['revenue'].mean()
print(f"Mean Revenue: {mean_rev}")
"""
        res = CodeSandbox.execute(code, self.df)
        self.assertTrue(res.success)
        self.assertEqual(len(res.figures), 1)
        self.assertIn("Mean Revenue: 150.0", res.stdout)
        self.assertIsNone(res.error_type)

    def test_runtime_error_capture(self):
        code = """
# Deliberate error on line 3
val = df['non_existent_column'] + 10
"""
        res = CodeSandbox.execute(code, self.df)
        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "KeyError")
        self.assertIsNotNone(res.failing_line)
        self.assertIn("non_existent_column", res.failing_code or res.error_message)


if __name__ == "__main__":
    unittest.main()
