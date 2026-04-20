# `agents`

Top-level package for the grounded-theory agent pipeline: CLI entry point, LangGraph implementation, prompt skills, and small operational scripts. Repo-root `data/` CSVs are consumed from here via `agents/core/paths.py`.

## Layout

| Path | Role |
|------|------|
| `cli.py` | Argument parsing and orchestration around the compiled graph (`core.app`). |
| `core/` | Pipeline code (graph, tools, paths, prompts, reporting). See `core/README.md`. |
| `skills/` | Markdown “skills” loaded at runtime for tool behavior (`skills/README.md` for loader env vars). |
| `scripts/` | Shell helpers to bring up inference servers / batch runs, plus Python utilities. |
| `weights/` | Where to place **local** model weights or Hugging Face caches when you run servers from this tree (not pushed to git).
