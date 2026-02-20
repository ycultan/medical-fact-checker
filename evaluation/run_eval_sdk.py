"""
LangSmith SDK Evaluation — run this after setting LANGSMITH_API_KEY.

What this script does:
  1. Creates (or reuses) a LangSmith dataset with 10 QA examples
  2. Defines two evaluators:
       • exact_match  - deterministic keyword check
       • llm_judge    - Gemini 2.5 Flash Lite scores the answer 1-5 for correctness
  3. Runs evaluate() which sends each row through run_agent() and scores it
  4. Prints a summary table to stdout
  5. The full experiment is viewable at https://smith.langchain.com
"""

import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our agent
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent.agent import run_agent

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAME = "QA Agent Eval — Basic Factual"

EXAMPLES = [
    {
        "question": "What is a normal resting heart rate?",
        "reference": "A normal resting heart rate for adults is 60 to 100 beats per minute.",
    },
    {
        "question": "What blood pressure is considered hypertension?",
        "reference": "Hypertension is diagnosed at 130/80 mmHg or higher.",
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "reference": "Type 1 is an autoimmune condition with little insulin production; Type 2 involves insulin resistance.",
    },
    {
        "question": "How does aspirin work?",
        "reference": "Aspirin is an NSAID that reduces pain and inflammation by inhibiting COX enzymes.",
    },
    {
        "question": "What BMI is considered obese?",
        "reference": "A BMI of 30 or above is considered obese.",
    },
    {
        "question": "How do vaccines work?",
        "reference": "Vaccines stimulate the immune system to produce antibodies, providing immunity without causing disease.",
    },
    {
        "question": "What is the difference between LDL and HDL cholesterol?",
        "reference": "LDL is bad cholesterol that builds up in arteries; HDL is good cholesterol that helps remove LDL.",
    },
    {
        "question": "What are the warning signs of a stroke?",
        "reference": "FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.",
    },
    {
        "question": "Do antibiotics work on viruses?",
        "reference": "No, antibiotics are ineffective against viruses. They only target bacteria.",
    },
    {
        "question": "What brain chemicals are linked to depression?",
        "reference": "Depression is associated with reduced serotonin, dopamine, and norepinephrine activity.",
    },
]


def get_or_create_dataset(client: Client) -> str:
    """Return existing dataset ID or create a new one."""
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print(f"[dataset] Using existing dataset '{DATASET_NAME}' (id={datasets[0].id})")
        return datasets[0].id

    print(f"[dataset] Creating new dataset '{DATASET_NAME}'")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Basic factual QA pairs for evaluating the LangGraph agent.",
    )

    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in EXAMPLES],
        outputs=[{"reference": ex["reference"]} for ex in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"[dataset] Created {len(EXAMPLES)} examples")
    return dataset.id


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evaluators
# ─────────────────────────────────────────────────────────────────────────────

def exact_match_evaluator(run, example) -> dict:
    """
    Deterministic evaluator: checks that key tokens from the reference
    appear in the model's answer (case-insensitive).
    """
    prediction = (run.outputs or {}).get("answer", "").lower()
    reference = (example.outputs or {}).get("reference", "").lower()

    # Score = fraction of reference words (len>3) found in prediction
    ref_words = [w for w in reference.split() if len(w) > 3]
    if not ref_words:
        return {"key": "exact_match", "score": 1.0}

    hits = sum(1 for w in ref_words if w in prediction)
    score = hits / len(ref_words)

    return {
        "key": "exact_match",
        "score": score,
        "comment": f"Matched {hits}/{len(ref_words)} reference tokens",
    }


def build_llm_judge() -> callable:
    """
    LLM-as-a-judge evaluator.
    Returns a score 1-5 and a short explanation.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    def llm_judge(run, example) -> dict:
        prediction = (run.outputs or {}).get("answer", "")
        reference = (example.outputs or {}).get("reference", "")
        question = (example.inputs or {}).get("question", "")

        prompt = f"""You are an expert evaluator. Score the following answer from 1 to 5.

Question: {question}
Reference answer: {reference}
Model answer: {prediction}

Scoring rubric:
5 = Correct, complete, well-phrased
4 = Mostly correct, minor gaps
3 = Partially correct
2 = Contains errors or is mostly wrong
1 = Completely wrong or irrelevant

Respond in the format:
SCORE: <integer 1-5>
REASON: <one sentence>"""

        response = llm.invoke(prompt)
        text = response.content.strip()

        score = 3  # default
        reason = ""
        for line in text.splitlines():
            if line.startswith("SCORE:"):
                try:
                    score = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        # Normalize to 0-1
        normalized = (score - 1) / 4.0

        return {
            "key": "llm_correctness",
            "score": normalized,
            "comment": f"LLM score {score}/5: {reason}",
        }

    return llm_judge


# ─────────────────────────────────────────────────────────────────────────────
# 3. Target function (what LangSmith will call per row)
# ─────────────────────────────────────────────────────────────────────────────

def agent_target(inputs: dict) -> dict:
    """Adapter: LangSmith passes the dataset row's 'inputs' dict here."""
    return run_agent(question=inputs["question"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(experiment_prefix: str = "qa-agent-v1"):
    client = Client()

    dataset_id = get_or_create_dataset(client)
    llm_judge = build_llm_judge()

    print(f"\n[eval] Starting experiment '{experiment_prefix}'…")
    results = evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[exact_match_evaluator, llm_judge],
        experiment_prefix=experiment_prefix,
        metadata={
            "model": "gemini-2.5-flash-lite",
            "agent_version": "1.0",
            "description": "Baseline medical QA agent",
        },
        max_concurrency=2,   # be gentle with rate limits
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("EVALUATION RESULTS")
    print("═" * 60)

    scores = {"exact_match": [], "llm_correctness": []}
    for r in results:
        for eval_result in r.get("evaluation_results", {}).get("results", []):
            key = eval_result.key
            if key in scores and eval_result.score is not None:
                scores[key].append(eval_result.score)

    for metric, vals in scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {metric:25s}  avg={avg:.3f}  n={len(vals)}")

    print("\n🔗 View in LangSmith UI:")
    print(f"   https://smith.langchain.com/projects (look for '{experiment_prefix}')")
    print("═" * 60)

    return results


if __name__ == "__main__":
    # Make sure these are set before running
    required = ["GOOGLE_API_KEY", "LANGSMITH_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing env vars: {missing}")
        print("Set them in .env or export them before running.")
        sys.exit(1)

    # Enable LangSmith tracing
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "langgraph-qa-agent")

    run_evaluation(experiment_prefix="qa-agent-v1")
