"""
LangGraph QA Agent — a simple but realistic multi-step question-answering agent.

Architecture:
  [classify] → [retrieve] → [answer] → [grade_confidence]
                    ↑                        |
                    └────── (low conf) ───────┘

State fields:
  question        - the user's raw question
  category        - "factual" | "opinion" | "calculation"
  context         - retrieved passages (fake KB here, swap with real retriever)
  answer          - generated answer
  confidence      - 0.0-1.0 self-reported confidence
  retry_count     - how many retrieval retries have happened
  final_answer    - the value returned to the caller
"""

import os
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Fake knowledge base (swap with a real vectorstore / retriever) ──────────
KNOWLEDGE_BASE = {
    "heart rate pulse": "A normal resting heart rate for adults is 60-100 beats per minute. Athletes may have rates as low as 40 bpm.",
    "blood pressure hypertension": "Normal blood pressure is below 120/80 mmHg. Hypertension is diagnosed at 130/80 mmHg or higher.",
    "diabetes insulin glucose": "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin. Type 2 diabetes involves insulin resistance and is strongly associated with lifestyle factors.",
    "aspirin ibuprofen anti-inflammatory": "Aspirin and ibuprofen are NSAIDs (non-steroidal anti-inflammatory drugs). They reduce pain, fever, and inflammation by inhibiting COX enzymes.",
    "bmi body mass index obesity": "BMI is calculated as weight (kg) divided by height (m) squared. A BMI of 18.5-24.9 is considered normal; 25-29.9 is overweight; 30+ is obese.",
    "vaccine immunity antibody": "Vaccines stimulate the immune system to produce antibodies without causing disease, providing immunity against future infections.",
    "cholesterol ldl hdl cardiovascular": "LDL ('bad') cholesterol contributes to plaque buildup in arteries. HDL ('good') cholesterol helps remove LDL. High LDL increases cardiovascular disease risk.",
    "antibiotic bacteria penicillin": "Antibiotics kill or inhibit bacterial growth. Penicillin, discovered by Alexander Fleming in 1928, was the first widely used antibiotic. Antibiotics are ineffective against viruses.",
    "stroke brain cerebrovascular": "A stroke occurs when blood supply to part of the brain is cut off. FAST stands for Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.",
    "depression mental health serotonin": "Depression is a mood disorder characterized by persistent sadness and loss of interest. It is associated with reduced serotonin, dopamine, and norepinephrine activity in the brain.",
}

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    category: str
    context: str
    answer: str
    confidence: float
    retry_count: int
    final_answer: str


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def classify_question(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Classify the question into a category to guide retrieval strategy."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a question classifier. Reply with exactly one word: "
            "'factual', 'opinion', or 'calculation'."
        )),
        HumanMessage(content=state["question"]),
    ])
    category = response.content.strip().lower()
    if category not in {"factual", "opinion", "calculation"}:
        category = "factual"  # safe default

    print(f"[classify] '{state['question']}' → {category}")
    return {**state, "category": category}


def retrieve_context(state: AgentState) -> AgentState:
    """Keyword search over the fake KB. Replace with vector retrieval in prod."""
    question_lower = state["question"].lower()
    matched = []
    for key, passage in KNOWLEDGE_BASE.items():
        if any(word in question_lower for word in key.split()):
            matched.append(passage)

    context = "\n\n".join(matched) if matched else "No relevant context found."
    print(f"[retrieve] found {len(matched)} passage(s)")
    return {**state, "context": context}


def generate_answer(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Generate an answer grounded in the retrieved context."""
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the context doesn't contain the answer, say "
        "'I don't have enough information to answer that.' "
        "After your answer, on a new line write: CONFIDENCE: <0.0-1.0>"
    )
    user_msg = f"Context:\n{state['context']}\n\nQuestion: {state['question']}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    text = response.content.strip()

    # Parse confidence score out of the response
    confidence = 0.5
    answer_text = text
    if "CONFIDENCE:" in text:
        parts = text.rsplit("CONFIDENCE:", 1)
        answer_text = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except ValueError:
            pass

    print(f"[answer] confidence={confidence:.2f}")
    return {**state, "answer": answer_text, "confidence": confidence}


def grade_and_finalize(state: AgentState) -> AgentState:
    """If confidence is low and we haven't retried too many times, signal retry."""
    # Retry loop disabled to save Gemini tokens (re-enable to loop on low confidence)
    # if state["confidence"] < 0.6 and state["retry_count"] < 2:
    #     print(f"[grade] low confidence, retry #{state['retry_count'] + 1}")
    #     return {
    #         **state,
    #         "retry_count": state["retry_count"] + 1,
    #         "final_answer": "",   # signals not done
    #     }

    final = state["answer"] if state["answer"] else "I could not find a reliable answer."
    print(f"[grade] finalized → {final[:60]}…")
    return {**state, "final_answer": final}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    """Edge: retry retrieval if confidence is low, otherwise finish."""
    # Retry loop disabled to save Gemini tokens (re-enable to loop back to retrieve)
    # if state["final_answer"] == "":
    #     return "retrieve"   # loop back
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_agent(model: str = "gemini-2.5-flash-lite") -> StateGraph:
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    # Bind LLM into node callables via closures
    def _classify(state):
        return classify_question(state, llm)

    def _answer(state):
        return generate_answer(state, llm)

    graph = StateGraph(AgentState)

    graph.add_node("classify", _classify)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("answer", _answer)
    graph.add_node("grade", grade_and_finalize)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "grade")
    # Retry loop disabled to save Gemini tokens (re-enable to retry on low confidence)
    # graph.add_conditional_edges("grade", should_retry, {
    #     "retrieve": "retrieve",
    #     END: END,
    # })
    graph.add_edge("grade", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Helper for single invocations
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(question: str, **kwargs) -> dict:
    """
    Thin wrapper used by LangSmith evaluation.

    Args:
        question: the question string
        **kwargs: extra keys from the dataset row (ignored)

    Returns:
        dict with 'answer' key (LangSmith evaluators key on this)
    """
    print("Preparing for blast off! 🚀")
    agent = build_agent()
    result = agent.invoke({
        "question": question,
        "category": "",
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0,
        "final_answer": "",
    })
    return {"answer": result["final_answer"]}


# Quick smoke test
if __name__ == "__main__":
    # result = run_agent("What is a normal resting heart rate?")
    # result = run_agent("Apple cider vinegar prevents cancer")
    result = run_agent("What causes a stroke?")
    print("\nFinal answer:", result["answer"])
