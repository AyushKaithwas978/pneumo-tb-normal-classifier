import tempfile
import unittest
from pathlib import Path

from rag_assistant import KnowledgeBase, build_explanation


class RagAssistantTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.kb_path = Path(self.tmpdir.name)
        (self.kb_path / "pneumonia.md").write_text(
            """# Pneumonia

## Overview
Pneumonia is an infection that can inflame air sacs in one or both lungs.

## Symptoms
Common symptoms include cough, fever, chills, and difficulty breathing.
""",
            encoding="utf-8",
        )
        (self.kb_path / "safety.md").write_text(
            """# Safety

## Disclaimer
This tool is educational and is not a medical diagnosis.
""",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_retrieves_relevant_pneumonia_sections(self):
        kb = KnowledgeBase(self.kb_path)

        snippets = kb.search("pneumonia cough fever", top_k=2)

        self.assertGreaterEqual(len(snippets), 1)
        self.assertEqual(snippets[0]["source"], "pneumonia.md")
        self.assertIn("Pneumonia", snippets[0]["title"])
        self.assertIn("infection", snippets[0]["text"])

    def test_builds_cited_prediction_explanation(self):
        kb = KnowledgeBase(self.kb_path)

        explanation = build_explanation(
            label="Pneumonia",
            confidence=0.83,
            probabilities={
                "Normal": 0.05,
                "Pneumonia": 0.83,
                "Tuberculosis": 0.12,
            },
            knowledge_base=kb,
        )

        self.assertIn("Pneumonia", explanation["summary"])
        self.assertIn("83.00%", explanation["summary"])
        self.assertTrue(explanation["clinical_context"])
        self.assertTrue(explanation["next_steps"])
        self.assertTrue(explanation["suggested_questions"])
        self.assertTrue(explanation["citations"])
        self.assertIn("not a medical diagnosis", explanation["disclaimer"].lower())

    def test_unknown_label_uses_safe_fallback(self):
        kb = KnowledgeBase(self.kb_path)

        explanation = build_explanation(
            label="Uncertain",
            confidence=0.34,
            probabilities={},
            knowledge_base=kb,
        )

        self.assertIn("Uncertain", explanation["summary"])
        self.assertTrue(explanation["next_steps"])
        self.assertIn("not a medical diagnosis", explanation["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
