"""The judge: the paper's end-to-end prompt (arXiv 2601.08620 v2, Figure 24), word for word.

It grades an answer Correct, Partially Correct or Incorrect against the reference;
the paper reports a binary label (Appendix I), so only Correct counts as right. The
three-level verdict and the judge's explanation are kept for the Review tab, where
a person can disagree with it (`data/audits/judge_labels.jsonl`).
"""

from __future__ import annotations

import hashlib
from typing import Any

from vidore_loop.answer import claude

VERDICTS = ("Correct", "Partially Correct", "Incorrect")

PROMPT = """You are an expert judge evaluating the accuracy of a test answer against a gold-standard true answer.
Your goal is to determine if the test answer captures the essential "core information."
### Evaluation Criteria:
- Correct: The test answer contains all core information of the true answer. Minor omissions of non-essential details or the addition of minor, non-contradictory information should still be marked as "Correct."
- Partially Correct: The test answer captures some of the core information, but suffers from significant omissions or includes substantial extra information that was not requested or present in the true answer.
- Incorrect: The test answer is fundamentally wrong, contradicts the true answer, or misses the core information entirely.
### Input Data:
Query: {query}
True Answer: {true_answer}
Test Answer: {test_answer}
### Output Format:
Provide a very brief explanation for your judgment. You must output your final response in a JSON format with two fields: "explanation" and "judgment" (which must be "Correct", "Partially Correct", or "Incorrect")."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "judgment": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["explanation", "judgment"],
    "additionalProperties": False,
}


def prompt_version() -> str:
    return hashlib.sha256(PROMPT.encode()).hexdigest()[:8]


def judge(
    query: str, true_answer: str, test_answer: str, model: str = "opus", effort: str = "medium"
) -> dict[str, Any]:
    """{verdict, correct, explanation, usage}; `correct` is the paper's binary label."""
    text = PROMPT.format(query=query, true_answer=true_answer, test_answer=test_answer)
    rep = claude.ask("", [claude.text_block(text)], model=model, effort=effort, schema=SCHEMA)
    s = rep.structured if isinstance(rep.structured, dict) else {}
    verdict = s.get("judgment")
    if verdict not in VERDICTS:
        raise claude.ClaudeError(f"the judge gave no verdict: {rep.text[:200]!r}")
    return {
        "verdict": verdict,
        "correct": verdict == "Correct",
        "explanation": str(s.get("explanation", "")),
        "usage": rep.usage(),
    }
