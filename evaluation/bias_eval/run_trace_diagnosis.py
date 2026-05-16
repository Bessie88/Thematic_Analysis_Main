"""
Bias Flow Trace Diagnosis  (LLM-as-judge version)
==================================================
Applies the P/G/O/M/D/U/X diagnostic framework (bias_flow_trace_diagnosis.md)
to FAILED sentences only (theme_recovery.sentence_results where correct=False).

For each failed sentence, reconstructs the full pipeline trace:
  Gold → Open Code → Pre-cluster → Post-cluster → Meta-theme → Final mapped label

Diagnostic states are assigned by an LLM judge (SGLang server, OpenAI-compatible).
M (Mixed) is the only state computed without the LLM: it uses cluster gold purity.
X (Lost/missing) is detected before calling the LLM.

Priority (highest → lowest):  X > M > LLM-assigned (P/G/O/D/U)

Outputs per run:
  trace_diagnosis_<scenario>.json   -- full per-sentence diagnosis data
  trace_diagnosis_<scenario>.html   -- summary HTML (3 tables + trace cards)

Usage (single run):
  python run_trace_diagnosis.py \\
      --run-dir /path/to/run_1 --dataset school_burnout --scenario balanced \\
      --api-base http://localhost:8000/v1

Usage (all runs):
  python run_trace_diagnosis.py --all --api-base http://localhost:8000/v1
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
from collections import Counter, defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # Thematic_Analysis_Main
EXPERIMENT_DIR = Path("/scratch/yucai/bias_experiment")
SAMPLED_DIR    = Path("/scratch/yucai/data/bias_sampled")
sys.path.insert(0, str(REPO))

from run_bias_experiment import DATASET_FIELDS

# ── Thresholds ─────────────────────────────────────────────────────────────────
PURITY_M = 0.60   # cluster gold purity below this → M (Mixed)

# ── LLM judge prompt ──────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert qualitative researcher evaluating thematic analysis pipelines.
You will be given a sentence, its gold indicator label, and a generated label at one pipeline stage.
Your task is to assign exactly ONE diagnostic state code from the list below.

State codes (in priority order, highest first):
  U  Ungrounded   - the generated label does not describe the sentence content at all; it is semantically disconnected
  D  Drifted      - the generated label shifts toward a clearly different gold indicator / thematic direction
  O  Over-generalized - the label is too broad / vague; loses the specific thematic distinction of the gold indicator
  G  Acceptable generalization - broader than the gold signal but still captures the core meaning acceptably
  P  Preserved    - the label preserves the gold meaning or is a valid fine-grained refinement of it

Rules:
- If the label is unrelated to both the sentence content and the gold indicator, output U.
- If the label matches a different thematic category better than the gold indicator, output D.
- If the label is plausible but too vague (e.g. "negative experience", "life difficulties"), output O.
- If the label is somewhat broader but still thematically sensible, output G.
- If the label preserves or validly refines the gold indicator meaning, output P.

Respond with ONLY a JSON object:
{"state": "<one letter>", "reason": "<one sentence explanation>"}
"""

JUDGE_USER_TMPL = """Sentence: {sentence}

Gold indicator: {gold}

Generated label at stage '{stage}': {label}

Assign the diagnostic state."""


def call_judge(client, sentence, gold, stage, label, model="llm", retries=3):
    """Call LLM judge and return (state, reason). Falls back to 'O' on failure."""
    if not label or label in ("?", ""):
        return "X", "Label is missing or empty."

    prompt = JUDGE_USER_TMPL.format(
        sentence=sentence[:500],
        gold=gold,
        stage=stage,
        label=label,
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            # Extract JSON even if wrapped in markdown
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            state = data.get("state", "O").upper().strip()
            if state not in ("P", "G", "O", "D", "U", "X"):
                state = "O"
            return state, data.get("reason", "")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return "O", f"judge error: {e}"


def apply_m_override(state, cluster_purity):
    """M overrides P/G/O/D if cluster purity is low (but not X or U — those are worse)."""
    if state in ("X", "U"):
        return state
    if cluster_purity is not None and cluster_purity < PURITY_M:
        return "M"
    return state


def first_bad_stage(path):
    stages = ["open_code", "pre_cluster", "post_cluster", "meta_theme", "final_mapping"]
    for i, s in enumerate(path):
        if s not in ("P", "G"):
            return stages[i]
    return None


# ── Load one run ──────────────────────────────────────────────────────────────

def extract_code_label(code: str) -> str:
    s = str(code)
    for prefix in ("Label:", "label:", "Code:", "code:"):
        if prefix in s:
            after = s.split(prefix, 1)[1].strip()
            for stop in ("\nEvidence:", "\nevidence:", "\nContext:", "\ncontext:", "\n"):
                if stop in after:
                    after = after.split(stop)[0].strip()
            return after
    return s.split("\n")[0]


def only_codes(entries):
    result = [e for e in entries if str(e).lower().startswith("code:")]
    return result if result else entries


def majority_cid(sid, codes_raw, c2c):
    counts = Counter(
        c2c.get(c) for c in codes_raw.get(sid, []) if c2c.get(c) is not None
    )
    return counts.most_common(1)[0][0] if counts else None


def load_run(run_dir: Path, dataset: str, scenario: str):
    sc = scenario
    fields   = DATASET_FIELDS.get(dataset, {"text": "text", "indicator": "indicator"})
    text_col = fields["text"]
    ind_col  = fields["indicator"]

    scenario_file = SAMPLED_DIR / dataset / f"{sc}.csv"
    pos_to_text, pos_to_ind = {}, {}
    with open(scenario_file, newline="", encoding="utf-8") as f:
        for pos, row in enumerate(csv.DictReader(f), start=1):
            t   = str(row.get(text_col, "")).strip()
            ind = str(row.get(ind_col, "")).strip()
            if t and ind:
                pos_to_text[pos] = t
                pos_to_ind[pos]  = ind

    all_ids = list(pos_to_text.keys())

    codes_data  = json.loads((run_dir / f"codes_{sc}.json").read_text())
    codes_raw   = {int(sid): entries for sid, entries in codes_data["codes_per_review"]}
    codes_by_id = {sid: only_codes(entries) for sid, entries in codes_raw.items()}

    clustered       = json.loads((run_dir / f"clustered_{sc}.json").read_text())
    code_to_pre     = clustered["code_to_cluster"]

    codebook_data   = json.loads((run_dir / f"codebook_{sc}.json").read_text())
    codebook        = codebook_data["codebook"]

    refined         = json.loads((run_dir / f"refined_{sc}.json").read_text())
    code_to_post    = refined["code_to_cluster_refined"]

    meta_data     = json.loads((run_dir / f"meta_themes_{sc}.json").read_text())
    meta_raw      = meta_data["meta_themes"]
    meta_themes   = ({e["name"]: e["cluster_ids"] for e in meta_raw}
                     if isinstance(meta_raw, list) else meta_raw)
    cluster_to_meta = {str(cid): ml for ml, cids in meta_themes.items() for cid in cids}

    theme_rec    = json.loads((run_dir / f"theme_recovery_{sc}.json").read_text())
    sent_results = {r["sentence_id"]: r for r in theme_rec["sentence_results"]}

    # Per-sentence stage assignments
    id_to_cid_pre  = {sid: majority_cid(sid, codes_raw, code_to_pre)  for sid in all_ids}
    id_to_cid_post = {sid: majority_cid(sid, codes_raw, code_to_post) for sid in all_ids}
    id_to_clab_pre  = {sid: codebook.get(str(id_to_cid_pre[sid]),  "?") for sid in all_ids}
    id_to_clab_post = {sid: codebook.get(str(id_to_cid_post[sid]), "?") for sid in all_ids}
    id_to_meta = {
        sid: cluster_to_meta.get(str(id_to_cid_post[sid]), "?") for sid in all_ids
    }

    def representative_code(sid):
        target_cid = id_to_cid_pre.get(sid)
        for c in codes_by_id.get(sid, []):
            if code_to_pre.get(c) == target_cid:
                return extract_code_label(c)
        codes = codes_by_id.get(sid, [])
        return extract_code_label(codes[0]) if codes else "?"

    id_to_repr_code = {sid: representative_code(sid) for sid in all_ids}

    # Cluster gold purity
    def cluster_purity_map(cid_map):
        cluster_inds = defaultdict(list)
        for sid in all_ids:
            cid = cid_map.get(sid)
            if cid is not None:
                cluster_inds[str(cid)].append(pos_to_ind[sid])
        return {
            cid: Counter(inds).most_common(1)[0][1] / len(inds)
            for cid, inds in cluster_inds.items() if inds
        }

    purity_pre  = cluster_purity_map(id_to_cid_pre)
    purity_post = cluster_purity_map(id_to_cid_post)

    meta_purity = {}
    for mt_name, cids in meta_themes.items():
        sids_in_mt = [sid for sid in all_ids
                      if str(id_to_cid_post.get(sid)) in [str(c) for c in cids]]
        if sids_in_mt:
            inds = [pos_to_ind[s] for s in sids_in_mt]
            meta_purity[mt_name] = Counter(inds).most_common(1)[0][1] / len(inds)

    failed_ids = [
        int(sid) for sid, r in sent_results.items() if not r.get("correct", True)
    ]

    return {
        "all_ids": all_ids,
        "pos_to_text": pos_to_text,
        "pos_to_ind": pos_to_ind,
        "id_to_repr_code": id_to_repr_code,
        "id_to_clab_pre": id_to_clab_pre,
        "id_to_clab_post": id_to_clab_post,
        "id_to_meta": id_to_meta,
        "id_to_cid_pre": id_to_cid_pre,
        "id_to_cid_post": id_to_cid_post,
        "purity_pre": purity_pre,
        "purity_post": purity_post,
        "meta_purity": meta_purity,
        "sent_results": sent_results,
        "failed_ids": failed_ids,
    }


# ── Diagnose one run ──────────────────────────────────────────────────────────

STAGE_NAMES = ["open_code", "pre_cluster", "post_cluster", "meta_theme", "final_mapping"]

def diagnose_run(data, client, model):
    failed_ids = data["failed_ids"]
    results    = []
    total = len(failed_ids)

    for i, sid in enumerate(failed_ids):
        gold      = data["pos_to_ind"][sid]
        text      = data["pos_to_text"][sid]
        open_code = data["id_to_repr_code"].get(sid, "?")
        pre_lab   = data["id_to_clab_pre"].get(sid, "?")
        post_lab  = data["id_to_clab_post"].get(sid, "?")
        meta_lab  = data["id_to_meta"].get(sid, "?")
        sr        = data["sent_results"].get(sid, {})
        final_lab = sr.get("pred_set", ["?"])[0] if sr.get("pred_set") else "?"

        cid_pre   = str(data["id_to_cid_pre"].get(sid, ""))
        cid_post  = str(data["id_to_cid_post"].get(sid, ""))
        pur_pre   = data["purity_pre"].get(cid_pre)
        pur_post  = data["purity_post"].get(cid_post)
        pur_meta  = data["meta_purity"].get(meta_lab)

        stages = [
            ("open_code",     open_code, None),
            ("pre_cluster",   pre_lab,   pur_pre),
            ("post_cluster",  post_lab,  pur_post),
            ("meta_theme",    meta_lab,  pur_meta),
            ("final_mapping", final_lab, None),
        ]

        path, reasons = [], []
        for stage_name, label, purity in stages:
            # X: missing
            if not label or label in ("?", ""):
                state, reason = "X", "Label is missing or empty."
            else:
                state, reason = call_judge(client, text, gold, stage_name, label, model=model)
                state = apply_m_override(state, purity)
            path.append(state)
            reasons.append(reason)

        fbs = first_bad_stage(path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{total}] sid={sid} gold={gold[:30]} path={'→'.join(path)} fbs={fbs}")

        results.append({
            "sentence_id": sid,
            "text": text,
            "gold": gold,
            "open_code":    open_code,
            "pre_cluster":  pre_lab,
            "post_cluster": post_lab,
            "meta_theme":   meta_lab,
            "final_mapped": final_lab,
            "path": path,
            "reasons": reasons,
            "first_bad_stage": fbs,
        })

    return results


# ── Aggregate ─────────────────────────────────────────────────────────────────

BAD_STATES = ("O", "M", "D", "U", "X")

def aggregate(results):
    n = len(results)
    if n == 0:
        return {}

    fbs_counts = Counter(r["first_bad_stage"] for r in results)
    fbs_table  = [
        {"stage": s, "count": fbs_counts.get(s, 0), "pct": round(100 * fbs_counts.get(s, 0) / n, 1)}
        for s in (STAGE_NAMES + [None]) if fbs_counts.get(s, 0) > 0
    ]

    stage_error = defaultdict(Counter)
    for r in results:
        for i, stage in enumerate(STAGE_NAMES):
            st = r["path"][i]
            if st in BAD_STATES:
                stage_error[stage][st] += 1
    matrix = {stage: {s: stage_error[stage].get(s, 0) for s in BAD_STATES} for stage in STAGE_NAMES}

    leakage = defaultdict(Counter)
    for r in results:
        leakage[r["gold"]][r["final_mapped"]] += 1
    leakage_table = {gold: dest.most_common(3) for gold, dest in leakage.items()}

    return {"n_failed": n, "first_bad_stage": fbs_table,
            "stage_error_matrix": matrix, "gold_leakage": leakage_table}


# ── HTML output ───────────────────────────────────────────────────────────────

STATE_COLORS = {
    "P": "#a8e6a3", "G": "#d4f0a8",
    "O": "#fde68a", "M": "#fbbf24",
    "D": "#fca5a5", "U": "#f87171",
    "X": "#d1d5db",
}

def state_cell(s):
    bg = STATE_COLORS.get(s, "white")
    return f"<td style='background:{bg};text-align:center;font-weight:bold'>{s}</td>"

def render_html(results, agg, dataset, scenario, run_dir):
    n = agg.get("n_failed", 0)

    fbs_rows = "".join(
        f"<tr><td>{r['stage']}</td><td>{r['count']}</td><td>{r['pct']}%</td></tr>"
        for r in agg.get("first_bad_stage", [])
    )
    table1 = f"""
<h2>1. First-Bad-Stage Table <small style="font-weight:normal;color:#777">(n={n} failed sentences)</small></h2>
<table>
<thead><tr><th>First bad stage</th><th>Count</th><th>% of failed</th></tr></thead>
<tbody>{fbs_rows}</tbody>
</table>"""

    mat = agg.get("stage_error_matrix", {})
    mat_rows = "".join(
        f"<tr><td><b>{stage}</b></td>" +
        "".join(f"<td style='text-align:center;background:{STATE_COLORS[s]}'>{mat.get(stage,{}).get(s,0)}</td>" for s in BAD_STATES)
        + "</tr>" for stage in STAGE_NAMES
    )
    table2 = f"""
<h2>2. Stage × Error-type Matrix</h2>
<table>
<thead><tr><th>Stage</th>{"".join(f"<th>{s}</th>" for s in BAD_STATES)}</tr></thead>
<tbody>{mat_rows}</tbody>
</table>
<p style="font-size:12px;color:#777">O=Over-generalized &nbsp; M=Mixed (cluster purity &lt;{PURITY_M}) &nbsp; D=Drifted &nbsp; U=Ungrounded &nbsp; X=Lost</p>"""

    leak = agg.get("gold_leakage", {})
    leak_rows = "".join(
        f"<tr><td><b>{gold[:45]}</b></td>" +
        "".join(f"<td>{dest[:35]} <span style='color:#888'>({cnt})</span></td>" for dest, cnt in dests[:3])
        + "</tr>" for gold, dests in sorted(leak.items())
    )
    table3 = f"""
<h2>3. Gold-Label Leakage <small style="font-weight:normal;color:#777">(top-3 wrong destinations per gold indicator)</small></h2>
<table>
<thead><tr><th>Gold indicator</th><th>Top wrong #1</th><th>Top wrong #2</th><th>Top wrong #3</th></tr></thead>
<tbody>{leak_rows}</tbody>
</table>"""

    stage_priority = {s: i for i, s in enumerate(STAGE_NAMES)}
    sorted_results = sorted(results, key=lambda r: (stage_priority.get(r["first_bad_stage"], 99), r["sentence_id"]))[:30]

    cards = ""
    for r in sorted_results:
        fbs_badge = (
            f"<span style='background:#ef4444;color:white;padding:2px 8px;border-radius:4px'>{r['first_bad_stage']}</span>"
            if r["first_bad_stage"] else
            "<span style='background:#22c55e;color:white;padding:2px 8px;border-radius:4px'>none</span>"
        )
        rows = ""
        stage_labels = [
            ("Open code",     r["open_code"]),
            ("Pre-cluster",   r["pre_cluster"]),
            ("Post-cluster",  r["post_cluster"]),
            ("Meta-theme",    r["meta_theme"]),
            ("Final mapping", r["final_mapped"]),
        ]
        for j, (sname, lbl) in enumerate(stage_labels):
            st  = r["path"][j]
            rsn = r["reasons"][j] if j < len(r["reasons"]) else ""
            rows += (
                f"<tr><td>{sname}</td>"
                f"<td>{lbl[:55]}</td>"
                f"{state_cell(st)}"
                f"<td style='font-size:11px;color:#555'>{rsn[:100]}</td></tr>"
            )
        cards += f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin-bottom:14px;background:white">
  <p style="margin:0 0 4px"><b>#{r['sentence_id']}</b> &nbsp; Gold: <b>{r['gold']}</b> &nbsp; First bad stage: {fbs_badge}</p>
  <p style="font-size:12px;color:#555;margin:4px 0 8px">{r['text'][:200]}{'…' if len(r['text'])>200 else ''}</p>
  <table style="font-size:12px;width:100%">
  <thead><tr><th>Stage</th><th>Label</th><th>State</th><th>Reason (LLM)</th></tr></thead>
  <tbody>{rows}</tbody>
  </table>
  <p style="font-size:11px;color:#888;margin-top:6px">Path: {' → '.join(r['path'])}</p>
</div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trace Diagnosis: {dataset} / {scenario}</title>
<style>
body  {{ font-family: Arial, sans-serif; max-width: 1300px; margin: 0 auto; padding: 24px; background:#fafafa; }}
h1    {{ color:#2c3e50; }}
h2    {{ color:#2c3e50; border-bottom:2px solid #4C72B0; padding-bottom:6px; margin-top:36px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:10px; }}
th,td {{ border:1px solid #ddd; padding:6px 10px; text-align:left; vertical-align:top; }}
thead {{ background:#2c3e50; color:white; }}
tbody tr:hover {{ background:#f0f4f8; }}
</style>
</head>
<body>
<h1>Bias Flow Trace Diagnosis  <small style="font-size:14px;color:#777">(LLM-as-judge)</small></h1>
<p style="color:#777">Dataset: <b>{dataset}</b> &nbsp;|&nbsp; Scenario: <b>{scenario}</b>
&nbsp;|&nbsp; Run: <b>{run_dir.name}</b> &nbsp;|&nbsp; Failed sentences: <b>{n}</b></p>
<p style="font-size:12px">
  State legend: &nbsp;
  {"".join(f'<span style="background:{STATE_COLORS[s]};padding:2px 7px;border-radius:3px;margin-right:4px"><b>{s}</b></span>' for s in ("P","G","O","M","D","U","X"))}
  &nbsp; P=Preserved &nbsp; G=Acceptable &nbsp; O=Over-generalized &nbsp; M=Mixed &nbsp; D=Drifted &nbsp; U=Ungrounded &nbsp; X=Lost
</p>
{table1}{table2}{table3}
<h2>4. Representative Failed Sentence Trace Cards <small style="font-weight:normal;color:#777">(up to 30, sorted by first bad stage)</small></h2>
{cards}
</body></html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def run_one(run_dir: Path, dataset: str, scenario: str, client, model: str):
    out_json = run_dir / f"trace_diagnosis_{scenario}.json"
    if out_json.exists():
        print(f"  SKIP {dataset}/{scenario} (already done)")
        return

    print(f"  [{dataset}/{scenario}] loading run data...")
    data = load_run(run_dir, dataset, scenario)
    n_fail = len(data["failed_ids"])
    print(f"  [{dataset}/{scenario}] {n_fail} failed sentences → LLM judging ({n_fail * 5} calls)...")
    results = diagnose_run(data, client, model)
    agg = aggregate(results)

    out_json.write_text(json.dumps({"summary": agg, "traces": results}, indent=2, ensure_ascii=False))
    print(f"  → {out_json}")

    out_html = run_dir / f"trace_diagnosis_{scenario}.html"
    out_html.write_text(render_html(results, agg, dataset, scenario, run_dir), encoding="utf-8")
    print(f"  → {out_html}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",   default=None)
    p.add_argument("--dataset",   default=None)
    p.add_argument("--scenario",  default=None)
    p.add_argument("--all",       action="store_true")
    p.add_argument("--api-base",  default="http://localhost:8000/v1",
                   help="OpenAI-compatible API base URL (SGLang or OpenRouter)")
    p.add_argument("--api-key",   default=None,
                   help="API key (needed for OpenRouter; omit for local SGLang)")
    p.add_argument("--model",     default="llm",
                   help="Model name, e.g. 'llm' for SGLang or 'google/gemma-3-27b-it' for OpenRouter")
    args = p.parse_args()

    from openai import OpenAI
    api_key = (args.api_key
               or os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("OPENAI_API_KEY2")
               or os.environ.get("OPENAI_API_KEY4")
               or os.environ.get("OPENAI_API_KEY5")
               or "dummy")
    base_url = args.api_base or os.environ.get("OPENROUTER_BASE_URL") or "http://localhost:8000/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Quick health check
    try:
        client.models.list()
        print(f"LLM server reachable at {args.api_base}")
    except Exception as e:
        print(f"ERROR: Cannot reach LLM server at {args.api_base}: {e}")
        sys.exit(1)

    SCENARIOS = ["balanced", "imbalanced", "rare_heavy"]

    if args.all:
        DATASETS = [d.name for d in EXPERIMENT_DIR.iterdir()
                    if d.is_dir() and d.name != "slurm_logs"]
        for dataset in sorted(DATASETS):
            for model_dir in sorted((EXPERIMENT_DIR / dataset).iterdir()):
                if not model_dir.is_dir(): continue
                for run_dir in sorted(model_dir.iterdir()):
                    if not run_dir.is_dir(): continue
                    for sc in SCENARIOS:
                        if not (run_dir / f"theme_recovery_{sc}.json").exists(): continue
                        try:
                            run_one(run_dir, dataset, sc, client, args.model)
                        except Exception as e:
                            print(f"  ERROR {dataset}/{model_dir.name}/{run_dir.name}/{sc}: {e}")
    else:
        if not (args.run_dir and args.dataset and args.scenario):
            p.error("Provide --run-dir, --dataset, --scenario  OR  --all")
        run_one(Path(args.run_dir), args.dataset, args.scenario, client, args.model)


if __name__ == "__main__":
    main()
