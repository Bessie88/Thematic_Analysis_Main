"""
Bias Flow Analysis: Icicle Tree + UMAP (all pipeline stages).

Pipeline stages tracked:
  Stage 0: Gold Indicator (ground truth)
  Stage 1: Dominant open code cluster (pre-refine, majority of codes per text)
  Stage 2: Refined cluster label (post-refine)
  Stage 3: Meta theme

Tree (Icicle): Gold Indicator → Cluster Label → Meta Theme
UMAP: 4 panels showing same text positions, colored at each stage.

Usage:
  python analyze_bias_flow.py --run-dir /path/to/run_1 --dataset ai_healthcare --scenario imbalanced
"""
import argparse, json, csv, os, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def extract_code_label(code):
    """Strip evidence/context from open code strings; return just the label."""
    s = str(code)
    for prefix in ("Label:", "label:", "Code:", "code:"):
        if prefix in s:
            after = s.split(prefix, 1)[1].strip()
            for stop in ("\nEvidence:", "\nevidence:", "\nContext:", "\ncontext:", "\n"):
                if stop in after:
                    after = after.split(stop)[0].strip()
            return after[:40]
    return s.split("\n")[0][:40]

p = argparse.ArgumentParser()
p.add_argument("--run-dir",     required=True)
p.add_argument("--dataset",     required=True)
p.add_argument("--scenario",    required=True)
p.add_argument("--top-n",       type=int, default=99)
p.add_argument("--embed-model", default=None)
p.add_argument("--output",      default=None)
args = p.parse_args()

RUN_DIR = Path(args.run_dir)
SC      = args.scenario
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, "/scratch/yucai/Thematic_Analysis_Main")
from run_bias_experiment import (DATASET_FIELDS, EMBED_MODEL as DEFAULT_EMBED_MODEL,
                                  DATASET_EVAL_COLS, DATASET_EVAL_EXTRA, DATASET_IND_DESC)
EMBED_MODEL   = args.embed_model or DEFAULT_EMBED_MODEL
text_col      = DATASET_FIELDS[args.dataset]["text"]
ind_col       = DATASET_FIELDS[args.dataset]["indicator"]
dim_col       = DATASET_EVAL_COLS.get(args.dataset, {}).get("dimension_col", "")
_dim_map_str  = DATASET_EVAL_EXTRA.get(args.dataset, {}).get("dim_map", "")
dim_map       = json.loads(_dim_map_str) if _dim_map_str else {}
# Mapping from gold indicator abbreviation → full description for semantic similarity.
# Only populated for datasets whose gold labels are abbreviations (e.g. climate).
ind_desc      = DATASET_IND_DESC.get(args.dataset, {})
scenario_file = Path(f"/scratch/yucai/data/bias_sampled/{args.dataset}/{SC}.csv")
out_html      = args.output or str(RUN_DIR / f"bias_flow_{SC}.html")

# ── Load all intermediate files ───────────────────────────────────────────────
print("[1/5] Loading files...")
pos_to_text, pos_to_ind, pos_to_dim = {}, {}, {}
with open(scenario_file, newline="", encoding="utf-8") as f:
    for pos, row in enumerate(csv.DictReader(f), start=1):
        t   = str(row.get(text_col, "")).strip()
        ind = str(row.get(ind_col, "")).strip()
        if t and ind:
            pos_to_text[pos] = t
            pos_to_ind[pos]  = ind
            # dimension: from CSV column, or via dim_map, or same as indicator
            if dim_col:
                pos_to_dim[pos] = str(row.get(dim_col, "")).strip() or ind
            elif dim_map:
                pos_to_dim[pos] = dim_map.get(ind, ind)
            else:
                pos_to_dim[pos] = ind
all_ids    = list(pos_to_text.keys())
indicators = sorted(set(pos_to_ind.values()))
dimensions = sorted(set(pos_to_dim.values()))

codes_data  = json.loads((RUN_DIR / f"codes_{SC}.json").read_text())
codes_by_id_raw = {int(sid): codes for sid, codes in codes_data["codes_per_review"]}

def only_codes(entries):
    """Return only 'Code: ...' entries (skip Evidence:, Note:, etc.)."""
    result = [e for e in entries if str(e).lower().startswith("code:")]
    return result if result else entries  # fallback: keep all if no Code: entries found

codes_by_id = {sid: only_codes(entries) for sid, entries in codes_by_id_raw.items()}

# Open code labels (truncated, for UMAP coloring) — Code: entries only
_code_label_counts = Counter(
    extract_code_label(c) for sid in all_ids for c in codes_by_id.get(sid, [])
)
_top_code_set = set(c for c, _ in _code_label_counts.most_common(25))

def _get_open_code_label(sid):
    codes = codes_by_id.get(sid, [])
    if not codes: return "?"
    lbl = extract_code_label(codes[0])
    return lbl if lbl in _top_code_set else "other"

id_to_open_code = {sid: _get_open_code_label(sid) for sid in all_ids}

clustered       = json.loads((RUN_DIR / f"clustered_{SC}.json").read_text())
code_to_cluster = clustered["code_to_cluster"]       # pre-refinement

codebook_data   = json.loads((RUN_DIR / f"codebook_{SC}.json").read_text())
codebook        = codebook_data["codebook"]          # {cid: label}

refined             = json.loads((RUN_DIR / f"refined_{SC}.json").read_text())
code_to_cluster_r   = refined["code_to_cluster_refined"]   # post-refinement

meta_data         = json.loads((RUN_DIR / f"meta_themes_{SC}.json").read_text())
meta_themes_raw   = meta_data["meta_themes"]
meta_themes = ({e["name"]: e["cluster_ids"] for e in meta_themes_raw}
               if isinstance(meta_themes_raw, list) else meta_themes_raw)

bias_results  = json.loads((RUN_DIR / "bias_results.json").read_text())
per_indicator = bias_results[SC]["metrics"]["per_indicator"]

cluster_to_meta = {str(cid): ml for ml, cids in meta_themes.items() for cid in cids}

# ── Build per-text stage assignments ─────────────────────────────────────────
def majority_cid(sid, c2c):
    # Use all raw entries (Code + Evidence + Note) for cluster voting — matches how clustering was done
    counts = Counter(c2c.get(c) for c in codes_by_id_raw.get(sid, [])
                     if c2c.get(c) is not None)
    return counts.most_common(1)[0][0] if counts else None

# Stage 1: pre-refine cluster (using original code_to_cluster)
id_to_cid_pre  = {sid: majority_cid(sid, code_to_cluster)   for sid in all_ids}
id_to_clab_pre = {sid: codebook.get(str(id_to_cid_pre[sid]), "?") for sid in all_ids}

# Stage 2: post-refine cluster
id_to_cid_ref  = {sid: majority_cid(sid, code_to_cluster_r) for sid in all_ids}
id_to_clab_ref = {sid: codebook.get(str(id_to_cid_ref[sid]), "?") for sid in all_ids}

# Stage 3: meta theme (from refined cluster)
id_to_meta     = {sid: cluster_to_meta.get(str(id_to_cid_ref[sid]), "?") for sid in all_ids}

# Representative open code per sentence: the Code: entry that belongs to the majority pre-cluster
# (so open code → pre-cluster edge is always consistent, 1 sentence = 1 flow)
def representative_code(sid):
    target_cid = id_to_cid_pre.get(sid)
    for c in codes_by_id.get(sid, []):  # Code: entries only
        if code_to_cluster.get(c) == target_cid:
            return extract_code_label(c)
    # fallback: first Code: entry
    codes = codes_by_id.get(sid, [])
    return extract_code_label(codes[0]) if codes else "?"

id_to_repr_code = {sid: representative_code(sid) for sid in all_ids}

# ── Filter poorly-recovered indicators ───────────────────────────────────────
# "Poorly recovered" = output_proportion < 25% of input_proportion (or zero)
RECOVERY_THRESHOLD = 0.25
focus_inds = [
    ind for ind in indicators
    if per_indicator.get(ind, {}).get("input_proportion", 0) > 0
    and (per_indicator.get(ind, {}).get("output_proportion", 0) /
         per_indicator.get(ind, {}).get("input_proportion", 1e-9)) < RECOVERY_THRESHOLD
]
if not focus_inds:
    # Fallback: take the worst-recovered indicator if none meet threshold
    focus_inds = [min(indicators,
                      key=lambda i: per_indicator.get(i, {}).get("output_proportion", 0) /
                                    max(per_indicator.get(i, {}).get("input_proportion", 1e-9), 1e-9))]
    print(f"  No indicators below {RECOVERY_THRESHOLD*100:.0f}% recovery — using worst: {focus_inds}")
else:
    print(f"  Poorly recovered indicators (<{RECOVERY_THRESHOLD*100:.0f}%): {focus_inds}")

focus_ids  = [sid for sid in all_ids if pos_to_ind[sid] in focus_inds]

# ── Embed texts + labels ──────────────────────────────────────────────────────
print("[2/5] Embedding texts (CPU)...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")

texts      = [pos_to_text[sid] for sid in all_ids]
embeddings = np.asarray(embed_model.encode(texts, batch_size=64,
                         normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)

# Include open codes in label pool for similarity computation
all_open_codes = list(dict.fromkeys(
    code for sid in all_ids for code in codes_by_id.get(sid, [])
))
label_pool = list(dict.fromkeys(
    indicators + list(ind_desc.values()) + all_open_codes + list(codebook.values()) + list(meta_themes.keys())
))
label_embs = np.asarray(embed_model.encode(label_pool, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)
lbl2emb    = {l: label_embs[i] for i, l in enumerate(label_pool)}

def sim(a, b):
    ea, eb = lbl2emb.get(a), lbl2emb.get(b)
    if ea is None or eb is None: return 0.0
    return float(np.dot(ea, eb))

# ── UMAP ─────────────────────────────────────────────────────────────────────
print("[3/5] UMAP projection...")
import umap as umap_lib
coords = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
                        ).fit_transform(embeddings)

# ── Build Plotly ──────────────────────────────────────────────────────────────
print("[4/5] Building figures...")
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

palette = px.colors.qualitative.Set1
fig_to_div = lambda f: pio.to_html(f, include_plotlyjs=False, full_html=False)

# Embed plotly JS inline so the HTML works without internet access
import plotly.offline as _poff
plotly_js_tag = f"<script>{_poff.get_plotlyjs()}</script>"

# ── 1. Bar chart ──────────────────────────────────────────────────────────────
ind_short = [i[:35] for i in indicators]
in_props  = [per_indicator[i]["input_proportion"]  for i in indicators]
out_props = [per_indicator[i]["output_proportion"] for i in indicators]
recalls   = [per_indicator[i]["recall"]            for i in indicators]
bar = go.Figure([
    go.Bar(name="Input",  x=ind_short, y=in_props,  marker_color="#4C72B0"),
    go.Bar(name="Output", x=ind_short, y=out_props, marker_color="#DD8452"),
])
bar.update_layout(barmode="group", height=420, xaxis_tickangle=-30,
                  title=f"Input vs Output Proportion [{args.dataset} / {SC}]",
                  margin=dict(b=130))

# ── 2. Overview cross-mapping Sankey: all Gold Indicators → Meta Themes ──────
# Stage colors (fixed per stage, not sim-based — easier to read)
STAGE_COLORS = {
    "gold": "rgba(55,126,184,0.92)",    # blue
    "code": "rgba(228,130,0,0.88)",     # orange
    "pre":  "rgba(138,100,210,0.88)",   # purple
    "ref":  "rgba(40,160,100,0.88)",    # green
    "meta": "rgba(210,60,60,0.88)",     # red
    "other":"rgba(170,170,170,0.7)",
}
LINK_COLORS = {
    "gold→code": "rgba(228,130,0,0.25)",
    "code→pre":  "rgba(138,100,210,0.25)",
    "pre→ref":   "rgba(40,160,100,0.25)",
    "ref→meta":  "rgba(210,60,60,0.25)",
    "gold→other":"rgba(170,170,170,0.2)",
}
STAGE_X = {"gold": 0.01, "code": 0.26, "pre": 0.51, "ref": 0.76, "meta": 0.99}

def make_overview_sankey():
    """Overview: Gold → Open Code → Pre-cluster → Post-cluster → Meta Theme, 1 COUNT PER SENTENCE."""
    flow_gc  = Counter()
    flow_cp  = Counter()
    flow_pr  = Counter()
    flow_rm  = Counter()
    for sid in all_ids:
        gold = pos_to_ind[sid]
        code = id_to_repr_code[sid]        # representative open code (1 per sentence)
        pre  = id_to_clab_pre.get(sid, "?")
        post = id_to_clab_ref.get(sid, "?")
        meta = id_to_meta.get(sid, "?")
        flow_gc[(gold, code)] += 1
        flow_cp[(code, pre)]  += 1
        flow_pr[(pre,  post)] += 1
        flow_rm[(post, meta)] += 1

    gold_sorted = sorted(indicators)
    code_sorted = sorted(set(id_to_repr_code.values()))
    pre_sorted  = sorted(set(id_to_clab_pre.values()))
    post_sorted = sorted(set(id_to_clab_ref.values()))
    meta_sorted = sorted(set(id_to_meta.values()))

    n_g = len(gold_sorted); n_c = len(code_sorted)
    n_p = len(pre_sorted);  n_r = len(post_sorted)

    offset_c = n_g
    offset_p = n_g + n_c
    offset_r = n_g + n_c + n_p
    offset_m = n_g + n_c + n_p + n_r

    idx = {}
    for i, v in enumerate(gold_sorted): idx[("gold", v)] = i
    for i, v in enumerate(code_sorted): idx[("code", v)] = offset_c + i
    for i, v in enumerate(pre_sorted):  idx[("pre",  v)] = offset_p + i
    for i, v in enumerate(post_sorted): idx[("post", v)] = offset_r + i
    for i, v in enumerate(meta_sorted): idx[("meta", v)] = offset_m + i

    pv = per_indicator
    node_labels = (
        [f"{g[:35]}<br>in={pv.get(g,{}).get('input_proportion',0):.2f} "
         f"out={pv.get(g,{}).get('output_proportion',0):.2f}" for g in gold_sorted] +
        [c[:30] for c in code_sorted] +
        [p[:30] for p in pre_sorted]  +
        [r[:30] for r in post_sorted] +
        [m[:35] for m in meta_sorted]
    )
    node_colors = (
        [STAGE_COLORS["gold"]] * n_g +
        [STAGE_COLORS["code"]] * n_c +
        [STAGE_COLORS["pre"]]  * n_p +
        [STAGE_COLORS["ref"]]  * n_r +
        [STAGE_COLORS["meta"]] * len(meta_sorted)
    )
    node_x = (
        [STAGE_X["gold"]] * n_g + [STAGE_X["code"]] * n_c +
        [STAGE_X["pre"]]  * n_p + [STAGE_X["ref"]]  * n_r +
        [STAGE_X["meta"]] * len(meta_sorted)
    )

    srcs, tgts, vals, cols = [], [], [], []
    def add(flow, sk1, sk2, col_key):
        for (a, b), cnt in flow.items():
            srcs.append(idx[(sk1, a)]); tgts.append(idx[(sk2, b)])
            vals.append(cnt);           cols.append(LINK_COLORS[col_key])

    add(flow_gc, "gold", "code", "gold→code")
    add(flow_cp, "code", "pre",  "code→pre")
    add(flow_pr, "pre",  "post", "pre→ref")
    add(flow_rm, "post", "meta", "ref→meta")

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=node_labels, color=node_colors, x=node_x,
            pad=10, thickness=16,
            line=dict(color="rgba(255,255,255,0.5)", width=0.8),
        ),
        link=dict(source=srcs, target=tgts, value=vals, color=cols),
    ))
    n_rows = max(n_g, n_c, n_p, n_r, len(meta_sorted))
    fig.update_layout(
        title="Full Pipeline Overview — All Gold Indicators (all sentences)",
        height=max(550, 28 * n_rows + 140),
        margin=dict(t=80, l=10, r=10, b=10),
        font=dict(size=10),
        annotations=[
            dict(x=x, y=1.06, xref="paper", yref="paper", showarrow=False,
                 text=f"<b>{lbl}</b>", font=dict(size=11, color=col), align="center")
            for x, lbl, col in [
                (STAGE_X["gold"], "Gold Indicator", "#377eb8"),
                (STAGE_X["code"], "Open Code",      "#c47a00"),
                (STAGE_X["pre"],  "Pre-cluster",    "#6b50c8"),
                (STAGE_X["ref"],  "Post-cluster",   "#1e8c58"),
                (STAGE_X["meta"], "Meta Theme",     "#d23c3c"),
            ]
        ],
        paper_bgcolor="white",
    )
    return fig

# ── 3. Detailed Sankey + Table (per poorly-recovered indicator) ───────────────
def make_sankey(ind):
    """Sentence-level flow: Gold → Open Code → Pre-cluster → Post-cluster → Meta Theme.
    Each sentence contributes exactly once via its representative open code."""
    ids_for = [sid for sid in focus_ids if pos_to_ind[sid] == ind]
    n = len(ids_for)
    if n == 0: return None

    pv = per_indicator[ind]

    # One path per sentence (not per code occurrence)
    path_counts = Counter()
    for sid in ids_for:
        code = id_to_repr_code[sid]
        pre  = id_to_clab_pre.get(sid, "?")
        post = id_to_clab_ref.get(sid, "?")
        meta = id_to_meta.get(sid, "?")
        path_counts[(code, pre, post, meta)] += 1

    if not path_counts: return None

    # Build nodes
    node_labels, node_colors, node_x_vals, node_hover = [], [], [], []
    node_idx = {}

    def get_node(stage_key, label):
        key = (stage_key, label)
        if key not in node_idx:
            node_idx[key] = len(node_labels)
            node_labels.append(label[:40])
            node_colors.append(STAGE_COLORS[stage_key])
            node_x_vals.append(STAGE_X[stage_key])
            s = sim(ind_desc.get(ind, ind), label) if stage_key != "gold" else 1.0
            node_hover.append(
                f"Gold: {ind}<br>"
                f"This node: {label}<br>"
                f"Sim to gold: {s:.3f}"
            )
        return node_idx[key]

    root_idx = get_node("gold", ind)
    node_labels[root_idx] = f"{ind[:38]}<br>n={n} sentences"

    srcs, tgts, vals, link_cols, link_hover = [], [], [], [], []
    def add_link(s, t, v, color_key, src_lbl, tgt_lbl):
        srcs.append(s); tgts.append(t); vals.append(v)
        link_cols.append(LINK_COLORS[color_key])
        link_hover.append(f"{src_lbl} → {tgt_lbl}<br>{v} sentences")

    for (code, pre, post, meta), cnt in path_counts.items():
        ci = get_node("code", code)
        pi = get_node("pre",  pre)
        ri = get_node("ref",  post)
        mi = get_node("meta", meta)
        add_link(root_idx, ci, cnt, "gold→code", ind,  code)
        add_link(ci, pi, cnt, "code→pre",  code, pre)
        add_link(pi, ri, cnt, "pre→ref",   pre,  post)
        add_link(ri, mi, cnt, "ref→meta",  post, meta)

    # Distribute y positions per stage column
    stage_nodes = defaultdict(list)
    for (sk, _), idx in node_idx.items():
        stage_nodes[sk].append(idx)

    node_y_vals = [0.5] * len(node_labels)
    for sk, idxs in stage_nodes.items():
        idxs_sorted = sorted(idxs)
        step = 0.85 / max(len(idxs_sorted), 1)
        for rank, idx in enumerate(idxs_sorted):
            node_y_vals[idx] = 0.05 + rank * step + step / 2

    annotations = [
        dict(x=x, y=1.06, xref="paper", yref="paper", showarrow=False,
             text=f"<b>{label}</b>",
             font=dict(size=12, color=STAGE_COLORS[sk].replace("0.88","1").replace("0.92","1")),
             align="center")
        for sk, x, label in [
            ("gold", STAGE_X["gold"], "Gold Indicator"),
            ("code", STAGE_X["code"], "Open Code"),
            ("pre",  STAGE_X["pre"],  "Pre-cluster"),
            ("ref",  STAGE_X["ref"],  "Post-cluster"),
            ("meta", STAGE_X["meta"], "Meta Theme"),
        ]
    ]

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=node_labels, color=node_colors,
                  x=node_x_vals, y=node_y_vals,
                  pad=10, thickness=20,
                  line=dict(color="rgba(255,255,255,0.5)", width=0.8),
                  customdata=node_hover,
                  hovertemplate="%{customdata}<extra></extra>"),
        link=dict(source=srcs, target=tgts, value=vals, color=link_cols,
                  customdata=link_hover,
                  hovertemplate="%{customdata}<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{ind}</b>  |  input={pv['input_proportion']:.3f}  "
                 f"output={pv['output_proportion']:.3f}  recall={pv['recall']:.2f}",
            font=dict(size=13),
        ),
        height=max(480, 30 * max(len(v) for v in stage_nodes.values()) + 130),
        margin=dict(t=80, l=10, r=10, b=20),
        font=dict(size=11),
        annotations=annotations,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig

sankey_htmls = []
for ind in focus_inds:
    fig = make_sankey(ind)
    if fig:
        sankey_htmls.append((ind, fig_to_div(fig)))

# ── Colored matrix table (per indicator) ─────────────────────────────────────
def sim_color(s):
    """Background color for a cell based on similarity to gold."""
    if s >= 0.70: return "#a8e6a3"   # green
    if s >= 0.50: return "#fde68a"   # yellow
    return "#fca5a5"                  # red

def cell(label, s=None):
    bg = f" style='background:{sim_color(s)}'" if s is not None else ""
    sim_str = f"<br><small>{s:.2f}</small>" if s is not None else ""
    return f"<td{bg}>{label[:40]}{sim_str}</td>"

def make_table(ind):
    ids_for = [sid for sid in focus_ids if pos_to_ind[sid] == ind]
    has_dim = any(pos_to_dim.get(s, s) != pos_to_ind[s] for s in ids_for)
    rows = []
    for sid in ids_for:
        text  = pos_to_text[sid]
        code  = id_to_repr_code[sid]
        pre   = id_to_clab_pre.get(sid, "?")
        post  = id_to_clab_ref.get(sid, "?")
        meta  = id_to_meta.get(sid, "?")
        gold  = pos_to_ind[sid]
        dim   = pos_to_dim.get(sid, "")
        rows.append(
            f"<tr>"
            f"<td title='{text}'>{text[:90]}{'…' if len(text)>90 else ''}</td>"
            f"{cell(code, sim(code, ind_desc.get(gold, gold)))}"
            f"{cell(pre,  sim(pre,  ind_desc.get(gold, gold)))}"
            f"{cell(post, sim(post, ind_desc.get(gold, gold)))}"
            f"<td>{meta[:40]}</td>"
            f"<td><b>{gold[:40]}</b></td>"
            + (f"<td>{dim[:40]}</td>" if has_dim else "")
            + "</tr>"
        )
    dim_th = "<th>Gold Dimension</th>" if has_dim else ""
    return f"""
<div style="overflow-x:auto;margin-top:12px">
<table style="border-collapse:collapse;width:100%;font-size:12px">
<thead style="background:#2c3e50;color:white">
<tr>
  <th style="min-width:220px">Sentence</th>
  <th>Open Code (sim↑)</th>
  <th>Pre-cluster (sim↑)</th>
  <th>Post-cluster (sim↑)</th>
  <th>Meta Theme</th>
  <th>Gold Indicator</th>
  {dim_th}
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</div>
<p style="font-size:11px;color:#888">
  🟢 sim ≥ 0.70 (close to gold) &nbsp; 🟡 0.50–0.70 &nbsp; 🔴 &lt; 0.50 (far from gold) &nbsp;|&nbsp;
  sim = cosine similarity between label embeddings
</p>"""

table_htmls = {ind: make_table(ind) for ind in focus_inds}

# ── UMAP — 2 panels (gold indicator + gold dimension) ────────────────────────
focus_set = set(focus_ids)

# Fixed color per focus indicator (consistent across all UMAP panels)
_focus_ind_list = sorted(set(pos_to_ind[s] for s in focus_ids))
_ind_cpal = px.colors.qualitative.Set1
IND2COL = {ind: _ind_cpal[i % len(_ind_cpal)] for i, ind in enumerate(_focus_ind_list)}

def make_umap(title, gold_fn, note):
    """Single UMAP panel colored by gold label (indicator or dimension).
    All sentences shown; focus sentences (poorly recovered) as large stars.
    Background dots colored by gold label to show ground-truth cluster structure.
    Hover on star: sentence text + gold label + pipeline assignments.
    """
    fig = go.Figure()
    seen, all_cats = set(), []
    for s in all_ids:
        v = gold_fn(s)
        if v not in seen:
            seen.add(v); all_cats.append(v)
    cpal    = px.colors.qualitative.Alphabet
    cat2col = {c: cpal[i % len(cpal)] for i, c in enumerate(all_cats)}

    # Background: all non-focus sentences, colored by gold label
    bg_by_cat = defaultdict(list)
    for sid in all_ids:
        if sid not in focus_set:
            bg_by_cat[gold_fn(sid)].append(sid)
    for cat in all_cats:
        sids_cat = bg_by_cat.get(cat, [])
        if not sids_cat: continue
        bidx = [all_ids.index(s) for s in sids_cat]
        fig.add_trace(go.Scatter(
            x=coords[bidx, 0], y=coords[bidx, 1], mode="markers",
            marker=dict(size=4, opacity=0.3, color=cat2col[cat]),
            text=[cat]*len(sids_cat), hoverinfo="text",
            name=cat[:35], legendgroup=cat, showlegend=True,
        ))

    # Focus stars: grouped by gold_fn label (indicator or dimension depending on panel)
    focus_by_label = defaultdict(list)
    for s in focus_ids:
        focus_by_label[gold_fn(s)].append(s)
    for label, sids in sorted(focus_by_label.items()):
        idx  = [all_ids.index(s) for s in sids]
        customdata = [[
            pos_to_text[s][:120],
            pos_to_ind[s],
            pos_to_dim.get(s, ""),
            id_to_repr_code[s],
            id_to_clab_pre.get(s, "?"),
            id_to_clab_ref.get(s, "?"),
            id_to_meta.get(s, "?"),
        ] for s in sids]
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers",
            marker=dict(size=14, symbol="star",
                        color=[cat2col.get(gold_fn(s), "#aaa") for s in sids],
                        line=dict(width=2, color="black")),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Gold indicator: %{customdata[1]}<br>"
                "Gold dimension: %{customdata[2]}<br>"
                "Open code: %{customdata[3]}<br>"
                "Pre-cluster: %{customdata[4]}<br>"
                "Post-cluster: %{customdata[5]}<br>"
                "Meta theme: %{customdata[6]}"
                "<extra></extra>"
            ),
            name=f"★ {label[:30]}", legendgroup=f"focus_{label}", showlegend=True,
        ))

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='font-size:11px'>{note}</span>",
        height=600, xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        legend=dict(orientation="v", x=1.01, font=dict(size=10)),
    )
    return fig_to_div(fig)

umap_dim_div = make_umap(
    "UMAP — Gold Dimension (Meta Theme level)",
    lambda s: pos_to_dim.get(s, pos_to_ind[s]),
    "Color = Gold Dimension (ground truth coarse-grained). "
    "★ = sentences from poorly recovered indicators (<25%), shown in coarse-grained space."
)

# ── Assemble HTML ─────────────────────────────────────────────────────────────
print("[5/5] Writing HTML...")

trees_html = ""
for i, (ind, div) in enumerate(sankey_htmls):
    v = per_indicator[ind]
    trees_html += f"""
<h3>#{i+1} &nbsp; {ind}
  <span style="font-weight:normal;font-size:13px;color:#777">
    input={v['input_proportion']:.3f} &nbsp; output={v['output_proportion']:.3f} &nbsp; recall={v['recall']:.2f}
  </span>
</h3>
{div}
{table_htmls.get(ind,'')}
"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Bias Flow: {args.dataset} / {SC}</title>
{plotly_js_tag}
<style>
body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto;
        padding: 24px; background: #fafafa; }}
h1   {{ color: #2c3e50; }}
h2   {{ color: #2c3e50; border-bottom: 2px solid #4C72B0; padding-bottom: 6px; margin-top: 40px; }}
h3   {{ color: #c0392b; margin-top: 28px; }}
.note {{ background: #eaf4fb; border-left: 4px solid #3498db;
         padding: 8px 14px; font-size: 13px; margin: 10px 0 16px; }}
table  {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; vertical-align: top; }}
thead  {{ background: #2c3e50; color: white; }}
tbody tr:hover {{ background: #f0f4f8; }}
</style>
</head>
<body>
<h1>Bias Flow Analysis</h1>
<p style="color:#777">Dataset: <b>{args.dataset}</b> &nbsp;|&nbsp; Scenario: <b>{SC}</b></p>

<h2>1. Input vs Output Proportion per Indicator</h2>
{fig_to_div(bar)}

<h2>2. Sentence Flow — Poorly Recovered Indicators (&lt;25% recovered)</h2>
<div class="note">
<b>Sankey (left→right):</b> Each line = 1 sentence, line width = sentence count. Gold → Open Code → Pre-cluster → Post-cluster → Meta Theme.<br>
<b>Table:</b> Each row = 1 sentence, color = semantic similarity between that label and the Gold Indicator (🟢 close 🟡 medium 🔴 far). Similarity is computed only up to the Indicator level (corresponding to clusters); Meta Theme corresponds to Dimension and is not compared.
</div>
{trees_html}

<h2>3. UMAP — Sentence positions in the full dataset</h2>
<div class="note">
Each point = 1 sentence, coordinates from text embeddings (fixed). Color = Gold label. <b>★ = sentences from indicators with &lt;25% recovery.</b><br>
Hover on ★ to see: sentence text + Gold Indicator/Dimension + pipeline stage assignments.
</div>
{umap_dim_div}

</body>
</html>"""

Path(out_html).write_text(html, encoding="utf-8")
print(f"Saved: {out_html}")
