# Bias Evaluation Pipeline

Evaluates how well a thematic analysis pipeline preserves indicator proportions across three sampling scenarios (balanced / imbalanced / rare_heavy).

## Structure

```
bias_eval/
├── run_bias_experiment.py       # Main controller: runs full pipeline + evaluation
├── evaluate_theme_recovery.py   # Two-level evaluation (cluster↔indicator, meta-theme↔dimension)
├── generate_report.py           # HTML report from evaluation results
├── analyze_bias_flow.py         # Sankey + UMAP visualization of pipeline flow
├── run_trace_diagnosis.py       # LLM-as-judge diagnosis for failed sentences
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

An OpenAI-compatible LLM server (e.g. SGLang) must be running for stages that call the LLM.  
The embedding model (`Qwen3-Embedding-0.6B`) should be placed under `agents/weights/`.

## Pipeline Overview

```
scenario CSV (balanced / imbalanced / rare_heavy)
        ↓
run_bias_experiment.py
  ├─ Open coding          (LLM)  → codes_{scenario}.json
  ├─ Axial coding         (KMeans) → clustered_{scenario}.json
  ├─ High-level labeling  (LLM)  → codebook_{scenario}.json
  ├─ Refine               (LLM)  → refined_{scenario}.json
  ├─ Meta-theme grouping  (LLM)  → meta_themes_{scenario}.json
  ├─ evaluate_theme_recovery.py  → theme_recovery_{scenario}.json
  └─ generate_report.py          → theme_recovery_report_{scenario}.html
       → bias_results.json
       → bias_report.html
```

## Usage

### Full pipeline

```bash
python run_bias_experiment.py \
    --dataset climate \
    --dataset-dir /path/to/data/bias_sampled/climate \
    --output-dir /path/to/output/run_1 \
    --model-type qwen
```

Supported datasets: `ai_healthcare`, `school_burnout`, `ai_posts`, `climate`

### Flow visualization (no LLM needed)

```bash
python analyze_bias_flow.py \
    --run-dir /path/to/run_1 \
    --dataset climate \
    --scenario balanced
```

Output: `bias_flow_balanced.html`

### Trace diagnosis (LLM needed)

```bash
python run_trace_diagnosis.py \
    --run-dir /path/to/run_1 \
    --dataset climate \
    --scenario balanced \
    --api-base http://localhost:8000/v1
```

Output: `trace_diagnosis_balanced.json` + `trace_diagnosis_balanced.html`

## Parallelism

LLM call concurrency is controlled via environment variables (default: 8 workers each):

```bash
export GT_OPEN_CODING_WORKERS=16
export GT_HIGH_LEVEL_WORKERS=8
```

## Output Files

| File | Description |
|------|-------------|
| `codes_{sc}.json` | Open coding results per text |
| `clustered_{sc}.json` | Cluster assignments |
| `codebook_{sc}.json` | Cluster labels |
| `refined_{sc}.json` | Post-refine assignments |
| `meta_themes_{sc}.json` | Meta-theme groupings |
| `theme_recovery_{sc}.json` | Two-level evaluation metrics |
| `bias_results.json` | MAE, Pearson r, per-indicator recall |
| `bias_report.html` | Summary HTML report |
| `bias_flow_{sc}.html` | Pipeline flow visualization |
| `trace_diagnosis_{sc}.html` | Per-sentence failure diagnosis |
