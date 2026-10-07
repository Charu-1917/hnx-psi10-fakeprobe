"""Chat agent tests: routing, tool use, and 'no number that is not in the result'."""
import json
import os
import re
import unittest

import chat_agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHIPS = ["Summarise the verdict", "Why is the peak at 0.0s?", "When are the suspicious moments?",
         "Do the voice and lip-sync agree?", "How reliable is this?", "Why is this UNCERTAIN?"]


def load(name):
    return json.load(open(os.path.join(ROOT, "demo_results", name + ".json"), encoding="utf-8"))


def allowed_numbers(result):
    """Every number in the result, plus roundings, plus x100 percentages and small counts."""
    nums = set()

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            nums.add(float(o))
            nums.add(round(o * 100))
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            nums.add(float(len(o)))
            [walk(v) for v in o]
        elif isinstance(o, str):
            for x in re.findall(r"\d+(?:\.\d+)?", o):
                nums.add(float(x))
    walk(result)
    return nums


def number_ok(token, nums):
    v = float(token)
    dec = len(token.split(".")[1]) if "." in token else 0
    return any(abs(round(n, dec) - v) < 1e-9 or abs(n - v) < 1e-9 for n in nums) or (v <= 100 and v == int(v) and v in (1, 2, 3))


class ChatTests(unittest.TestCase):
    def test_every_chip_on_every_demo_uses_tools_and_invents_no_numbers(self):
        for name in ("video", "image", "audio"):
            r = load(name)
            nums = allowed_numbers(r)
            for q in CHIPS:
                out = chat_agent.answer(q, r)
                self.assertTrue(out["tools_used"], q)
                self.assertFalse(out["llm_used"])
                for tok in re.findall(r"\d+(?:\.\d+)?", out["answer"]):
                    self.assertTrue(number_ok(tok, nums), f"{name} / {q}: unexpected number {tok} in: {out['answer']}")

    def test_routing(self):
        self.assertEqual(chat_agent.route("Why is the peak at 0.0s?"), ["explain_peak"])
        self.assertIn("audit", chat_agent.route("How reliable is this?"))
        self.assertIn("audit", chat_agent.route("Why is this UNCERTAIN?"))
        self.assertIn("get_intervals", chat_agent.route("when are the suspicious moments"))

    def test_sample_note_and_no_data_cases(self):
        self.assertIn("sample", chat_agent.answer("Summarise", load("image"))["answer"].lower())
        a = chat_agent.answer("Why is the peak?", load("audio"))["answer"]
        self.assertIn("no per-moment", a)



if __name__ == "__main__":
    unittest.main()
