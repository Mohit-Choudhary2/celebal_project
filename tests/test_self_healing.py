"""Integration tests for AnalystAgent self-healing loop."""

import unittest
import pandas as pd
from backend.agent import AnalystAgent
from backend.code_sandbox import CodeSandbox


class TestSelfHealing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            " Region ": ["North", "South", "East", "West"],
            " Sales ": [1200, 3400, 2100, 4500],
            " Profit ": [300, 850, 420, 1100]
        })
        self.agent = AnalystAgent(provider="heuristic")

    def test_agent_heals_column_whitespace_and_executes(self):
        # Even with messy columns with leading/trailing spaces,
        # the agent runs analysis, repairs access, and outputs figures
        result = self.agent.run_analysis(self.df, "Compare sales across regions")
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["figures"]), 1)
        self.assertGreater(len(result["insights"]), 0)

    def test_direct_heuristic_healing(self):
        prompt = """
An error occurred:
Error Type: KeyError
Error Message: 'sales'
Previous Code:
```python
fig = px.bar(df, x='region', y='sales')
```
"""
        healed_code = self.agent._heuristic_heal_code(prompt)
        self.assertIn("df.columns.str.strip()", healed_code)


if __name__ == "__main__":
    unittest.main()
