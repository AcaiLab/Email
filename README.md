# Prompt Injection Detection for Email Agents Through Attack Chain Modeling

This repository contains the code accompanying the paper "Prompt Injection Detection for Email Agents Through Attack Chain Modeling," accepted at IEEE ICTAI 2026.

## Folder Layout

```text
src/llmail_research/    data loading, feature, metric, and modeling code
scripts/run_pipeline.py One-command pipeline runner
scripts/steps/          Numbered experiment stages
```
## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```
The scripts download public datasets from Hugging Face and GitHub as needed.

Use this before a full reproduction:

```bash
python scripts/run_pipeline.py --quick --skip-detectors
```

This checks that data loading, labels, and the main modeling code work.

## Full Pipeline

```bash
python scripts/run_pipeline.py
```

This runs:

1. dataset preparation
2. stage-motivation experiments
3. cross-dataset and NotInject hard-negative experiments
4. final framework component ablation
5. comparison against accessible published prompt-injection detectors

The OpenAI LLM judge is intentionally excluded from the default command because
it requires API credits.

## Optional OpenAI LLM Judge (when API credits available)

```bash
export OPENAI_API_KEY="..."
python scripts/steps/05_openai_judge_optional.py --full --workers 4
```
