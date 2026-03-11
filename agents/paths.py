"""Centralized filesystem paths for the GT pipeline."""
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = AGENTS_DIR / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
LOGS_DIR = OUTPUTS_DIR / "logs"
WEIGHTS_DIR = AGENTS_DIR / "weights"
OLD_SLURM_DIR = AGENTS_DIR / "old_slurm.out"
DEFAULT_DATA_CSV = AGENTS_DIR.parent / "data" / "review_text_english_50k.csv"

GT_CODES_ONLY_PATH = DATA_DIR / "gt_codes_only.json"
CLUSTERED_CODES_PATH = DATA_DIR / "gt_clustered_codes.json"
CODEBOOK_PATH = DATA_DIR / "codebook.json"
HIERARCHY_EDGES_PATH = DATA_DIR / "gt_hierarchy_edges.jsonl"
HIERARCHY_PATH = DATA_DIR / "gt_hierarchy.json"
GRAPH_PATH = DATA_DIR / "gt_graph.json"
CROSS_CLUSTER_EDGES_PATH = DATA_DIR / "gt_cross_cluster_edges.jsonl"
GLOBAL_GRAPH_PATH = DATA_DIR / "gt_global_graph.json"
CLEANED_GLOBAL_GRAPH_PATH = DATA_DIR / "gt_global_graph_cleaned.json"
OPEN_CODES_MARKDOWN_PATH = DATA_DIR / "gt_open_codes_all_reviews.md"
GRAPH_HTML_PATH = DATA_DIR / "gt_graph.html"

GT_AGENT_TRACE_LOG_PATH = LOGS_DIR / "gt_agent_trace.log"
SERVER_LOG_PATH = AGENTS_DIR / "server.log"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(AGENTS_DIR))
    except ValueError:
        return str(path)
