# GT Workflow

![GT Pipeline Diagram](artifacts/GT%20Pipeline%20Diagram.svg)

## Research report (post global graph)

After `gt_global_graph.json` is built, the full pipeline (`agents/scripts/launch_sgl.sh` via Slurm `run.sh`) **stops the Qwen SGLang server**, starts **Mistral 7B Instruct v0.3** on the same port, runs `python -m agents.cli --report-only`, then stops Mistral.

- **Weights**: `agents/weights/Mistral-7B-Instruct-v0.3` (install with `agents/scripts/download_mistral_7b_v03.sh` if missing; the job exits with an error if this directory is absent).
- **Output**: `agents/outputs/data/research_report.md`
- **Mistral server log**: `agents/report_server.log`
- **Wall time**: Adds one extra model load + one LLM call; budget extra Slurm time accordingly.

### Standalone report step

With Mistral (or any OpenAI-compatible server) already listening:

```bash
python -m agents.cli --report-only --research-question "Your question here"
```

Optional: `--graph-path PATH`, `--report-api-base URL` (default `http://localhost:8000/v1` or env `REPORT_OPENAI_BASE`), `--report-model NAME` (default `llm` or env `REPORT_MODEL_NAME`). Large graphs are truncated for context (nodes kept; edges trimmed first); truncation is logged.
