import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import chat_agent  # noqa: E402


def demo(name):
    with open(os.path.join(ROOT, "demo_results", name + ".json"), encoding="utf-8") as f:
        return json.load(f)


class ChatTests(unittest.TestCase):
    def test_answers_come_from_result_and_flag_sample(self):
        r = chat_agent.answer("Summarise the verdict", demo("image"))
        self.assertIn("SAMPLE", r["answer"])
        self.assertFalse(r["llm_used"])
        self.assertIn("get_summary", r["tools_used"])

    def test_reliability_question_uses_audit(self):
        r = chat_agent.answer("How reliable is this?", demo("video"))
        self.assertIn("audit", r["tools_used"])
        self.assertIn("certainty", r["answer"])

    def test_no_peak_data_for_audio(self):
        r = chat_agent.answer("why is the peak high", demo("audio"))
        self.assertIn("no per-moment data", r["answer"])


if __name__ == "__main__":
    unittest.main()
