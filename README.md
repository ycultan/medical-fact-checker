# LangGraph + LangSmith Evaluation Demo

A minimal but realistic QA agent built with LangGraph, evaluated with LangSmith.

## Project Structure

```
langgraph-langsmith-eval/
├── agent/
│   └── agent.py               # LangGraph agent (classify → retrieve → answer → grade)
├── evaluation/
│   ├── run_eval_sdk.py        # SDK evaluation with two evaluators
│   └── run_eval_ui_setup.py   # Sets up dataset for UI evaluation
├── .env.example
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys

python agent/agent.py                    # smoke test
python evaluation/run_eval_sdk.py        # full SDK evaluation
python evaluation/run_eval_ui_setup.py   # UI setup
```

## Agent Design

```
[classify] → [retrieve] → [answer] → [grade]
                 ↑                       |
                 └──── (retry loop) ─────┘
```

