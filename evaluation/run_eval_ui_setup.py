"""
LangSmith UI Evaluation Setup

This script handles the programmatic parts that pair with the LangSmith UI workflow.
After running this, you'll use the LangSmith web UI to:
  • Browse the dataset
  • Configure and launch an experiment
  • Apply AI-assisted scoring
  • Compare experiments side-by-side

"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent.agent import run_agent

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Ensure dataset exists (same as SDK script)
# ─────────────────────────────────────────────────────────────────────────────

def setup_dataset():
    from langsmith import Client

    DATASET_NAME = "QA Agent Eval — Basic Factual"
    EXAMPLES = [
        {"question": "What is a normal resting heart rate?",
         "reference": "A normal resting heart rate for adults is 60 to 100 beats per minute."},
        {"question": "How do vaccines work?",
         "reference": "Vaccines stimulate the immune system to produce antibodies without causing disease."},
        {"question": "What is the difference between LDL and HDL cholesterol?",
         "reference": "LDL is bad cholesterol; HDL is good cholesterol that helps remove LDL."},
        {"question": "Do antibiotics work on viruses?",
         "reference": "No, antibiotics are ineffective against viruses."},
        {"question": "What are the warning signs of a stroke?",
         "reference": "FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services."},
    ]

    client = Client()
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))

    if existing:
        print(f"✓ Dataset already exists: '{DATASET_NAME}'")
        print(f"  URL: https://smith.langchain.com/datasets/{existing[0].id}")
        return existing[0].id

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Factual QA pairs for evaluating the LangGraph QA agent.",
    )
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in EXAMPLES],
        outputs=[{"reference": ex["reference"]} for ex in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"✓ Created dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples")
    print(f"  URL: https://smith.langchain.com/datasets/{dataset.id}")
    return dataset.id


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run a traced experiment so it shows up in the UI
# ─────────────────────────────────────────────────────────────────────────────

def run_traced_experiment(dataset_name: str = "QA Agent Eval — Basic Factual"):
    from langsmith.evaluation import evaluate

    print("\nRunning traced experiment (visible in LangSmith UI)…")

    def target(inputs):
        return run_agent(question=inputs["question"])

    def keyword_check(run, example):
        pred = (run.outputs or {}).get("answer", "").lower()
        ref = (example.outputs or {}).get("reference", "").lower()
        keywords = [w for w in ref.split() if len(w) > 3]
        hits = sum(1 for k in keywords if k in pred)
        score = hits / len(keywords) if keywords else 1.0
        return {"key": "keyword_coverage", "score": round(score, 3)}

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[keyword_check],
        experiment_prefix="qa-agent-ui-demo",
        metadata={"launched_from": "ui_setup script"},
        max_concurrency=1,
    )

    print("✓ Experiment complete. Open LangSmith UI to see results.")
    print("  → https://smith.langchain.com")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Export a local JSON snapshot 
# ─────────────────────────────────────────────────────────────────────────────

def export_local_snapshot():
    """Run a local mini-eval and save results to JSON (no LangSmith needed)."""
    questions = [
        ("What is a normal resting heart rate?", "60"),
        ("How do vaccines work?", "antibodies"),
        ("What BMI is considered obese?", "30"),
        ("What caused the 2020 Olympics to be postponed?", "don't have enough"),  # should fail gracefully
    ]

    results = []
    for question, expected_keyword in questions:
        result = run_agent(question)
        answer = result.get("answer", "")
        passed = expected_keyword.lower() in answer.lower()
        results.append({
            "question": question,
            "answer": answer,
            "expected_keyword": expected_keyword,
            "passed": passed,
        })
        status = "✓" if passed else "✗"
        print(f"  {status} Q: {question[:50]}")
        print(f"      A: {answer[:80]}")

    output_path = os.path.join(os.path.dirname(__file__), "local_snapshot.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["passed"])
    print(f"\n✓ Local snapshot: {passed}/{len(results)} passed")
    print(f"  Saved to: {output_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_langsmith = bool(os.environ.get("LANGSMITH_API_KEY"))

    print("=" * 60)
    print("LangSmith UI Setup Script")
    print("=" * 60)

    if use_langsmith:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "langgraph-qa-agent")
        setup_dataset()
        run_traced_experiment()
    else:
        print("No LANGSMITH_API_KEY found — running local snapshot only.\n")
        export_local_snapshot()
