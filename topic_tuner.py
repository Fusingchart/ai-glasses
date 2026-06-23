"""Tune topic extraction prompt on sample conversation snippets."""
import json
from pathlib import Path

from openai import OpenAI

from config import Config
from topic_extractor import TopicExtractor

SAMPLE_SNIPPETS: list[dict] = [
    {"text": "I've been reading about attention mechanisms in transformers", "expected": ["attention mechanisms", "transformers"]},
    {"text": "Can you believe the Fed raised rates again?", "expected": ["Federal Reserve", "interest rates"]},
    {"text": "I was thinking about going to the gym later", "expected": []},
    {"text": "The CRISPR paper we looked at last week was fascinating", "expected": ["CRISPR", "gene editing"]},
    {"text": "I'm not sure what to have for lunch", "expected": []},
    {"text": "Deep reinforcement learning is really hard to stabilize", "expected": ["deep reinforcement learning"]},
    {"text": "We should probably refactor the database layer", "expected": ["database refactoring"]},
    {"text": "Elon Musk tweeted something controversial again", "expected": ["Elon Musk"]},
    {"text": "The Krebs cycle is so elegant when you think about it", "expected": ["Krebs cycle", "cellular respiration"]},
    {"text": "I wonder if we're in a simulation", "expected": ["simulation hypothesis"]},
    {"text": "Can you pass the salt", "expected": []},
    {"text": "Quantum entanglement still blows my mind", "expected": ["quantum entanglement"]},
    {"text": "The housing market in Seattle is insane right now", "expected": ["housing market", "Seattle real estate"]},
    {"text": "I should really read more about stoicism", "expected": ["stoicism"]},
    {"text": "okay so whatever", "expected": []},
    {"text": "Byzantine fault tolerance is the key insight behind Paxos", "expected": ["Byzantine fault tolerance", "Paxos"]},
    {"text": "The mitochondria is the powerhouse of the cell", "expected": ["mitochondria"]},
    {"text": "I love this coffee", "expected": []},
    {"text": "SpaceX landed both boosters simultaneously", "expected": ["SpaceX", "reusable rockets"]},
    {"text": "The new Claude model is really impressive for coding", "expected": ["Claude", "LLM"]},
]


def evaluate(extractor: TopicExtractor, snippets: list[dict]) -> dict:
    hits = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for sample in snippets:
        result = extractor.extract(sample["text"])
        expected_empty = len(sample["expected"]) == 0
        got_empty = not result.has_topics

        if expected_empty and got_empty:
            true_negatives += 1
        elif expected_empty and not got_empty:
            false_positives += 1
        elif not expected_empty and result.has_topics:
            hits += 1
        else:
            false_negatives += 1

    total = len(snippets)
    return {
        "total": total,
        "hits": hits,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": round((hits + true_negatives) / total, 3),
    }


def run_tuning(output_path: str = "topic_eval.json") -> None:
    cfg = Config()
    client = OpenAI(api_key=cfg.openai_api_key)
    extractor = TopicExtractor(client=client, model=cfg.topic_model)

    print(f"[tuner] evaluating {len(SAMPLE_SNIPPETS)} snippets...")
    metrics = evaluate(extractor, SAMPLE_SNIPPETS)
    print(json.dumps(metrics, indent=2))

    history = extractor.history
    out = {"metrics": metrics, "history": history}
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"[tuner] results written to {output_path}")


if __name__ == "__main__":
    run_tuning()
