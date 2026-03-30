# GT Agent Skills

This repo supports stable, long-term behavioral instructions (“skills”) as Markdown files.

## Where skills live

`agents/skills/*.md`

## Enable/disable

Environment variables:

- `GT_USE_SKILLS` (default: `1`) — set to `0`/`false` to disable.
- `GT_SKILLS_DIR` — override the skills directory.

## How skills are selected

The loader at `agents/core/skills.py` loads:

`agents/skills/{skill_key}.md`

The pipeline maps tool/phase names to skill keys (see `agents/core/tools.py` and `agents/core/report.py`).

