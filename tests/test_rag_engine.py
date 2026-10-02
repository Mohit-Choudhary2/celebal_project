"""Unit tests for RAG Engine retrieval and context formatting."""

import unittest
from backend.rag_engine import rag_engine


class TestRAGEngine(unittest.TestCase):
    def test_keyerror_retrieval(self):
        docs = rag_engine.retrieve("KeyError: 'sales' not in index", error_type="KeyError")
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("KeyError", docs[0]["error_types"])
        self.assertIn("Column Access", docs[0]["title"])

    def test_settingwithcopy_retrieval(self):
        docs = rag_engine.retrieve("A value is trying to be set on a copy of a slice from a DataFrame", error_type="SettingWithCopyWarning")
        self.assertGreaterEqual(len(docs), 1)
        self.assertEqual(docs[0]["id"], "pandas_settingwithcopy")

    def test_groupby_retrieval(self):
        docs = rag_engine.retrieve("groupby agg failed numeric_only multiindex", error_type="TypeError")
        self.assertGreaterEqual(len(docs), 1)
        titles = [d["title"] for d in docs]
        self.assertTrue(any("GroupBy" in t for t in titles))

    def test_format_rag_context(self):
        docs = rag_engine.retrieve("KeyError", error_type="KeyError")
        context = rag_engine.format_rag_context(docs)
        self.assertIn("OFFICIAL PYTHON/PANDAS DOCUMENTATION", context)
        self.assertIn("Verified fix", context)


if __name__ == "__main__":
    unittest.main()
