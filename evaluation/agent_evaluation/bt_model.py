"""
Itemized Bipartite Bradley-Terry model fitting.

Model:  P(y=1) = σ(γ_a - α_b - δ_{a,i})
  γ_a : answerer strength
  α_b : benchmarker strength
  δ_{a,i} : item difficulty

Fit via logistic regression with L2 (MAP) regularization.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # σ(x)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_outcomes(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("outcome_y") is not None:
                rows.append(d)
    return rows


def _index(items: list) -> dict:
    return {v: i for i, v in enumerate(sorted(set(items)))}


# ── Full CRB-style model ──────────────────────────────────────────────────────

def fit_full_bt(
    rows: list[dict],
    lam: float = 1.0,
) -> dict:
    """
    Fit: P(y=1) = σ(γ_a - α_b - δ_{a,i})

    Falls back to generator_model when answerer_model is absent (old data).
    """
    answerers    = _index([r.get("answerer_model", r["generator_model"]) for r in rows])
    benchmarkers = _index([r["benchmarker_model"] for r in rows])
    # item = (benchmarker_model, codebook_id, code_id, component)
    item_keys = [(r["benchmarker_model"], r["codebook_id"], r["code_id"], r["component"])
                 for r in rows]
    items = _index(item_keys)

    n_ans   = len(answerers)
    n_bench = len(benchmarkers)
    n_items = len(items)
    n_params = n_ans + n_bench + n_items

    a_idx = np.array([answerers[r.get("answerer_model", r["generator_model"])] for r in rows])
    b_idx = np.array([benchmarkers[r["benchmarker_model"]] for r in rows])
    i_idx = np.array([items[(r["benchmarker_model"], r["codebook_id"],
                              r["code_id"], r["component"])] for r in rows])
    y     = np.array([r["outcome_y"] for r in rows], dtype=float)

    def neg_log_posterior(params):
        gamma = params[:n_ans]
        alpha = params[n_ans:n_ans + n_bench]
        delta = params[n_ans + n_bench:]

        logits = gamma[a_idx] - alpha[b_idx] - delta[i_idx]
        log_lik = np.sum(y * np.log(expit(logits) + 1e-12) +
                         (1 - y) * np.log(1 - expit(logits) + 1e-12))

        l2 = lam * (np.sum(gamma**2) + np.sum(alpha**2) + np.sum(delta**2))
        # identifiability: soft constraint sum(gamma)=0
        ident = 10.0 * np.sum(gamma)**2

        return -log_lik + l2 + ident

    x0 = np.zeros(n_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(neg_log_posterior, x0, method="L-BFGS-B",
                          options={"maxiter": 2000, "ftol": 1e-10})

    params = result.x
    gamma = params[:n_ans]
    alpha = params[n_ans:n_ans + n_bench]
    delta = params[n_ans + n_bench:]

    ans_ranking   = sorted(answerers.keys(),    key=lambda k: -gamma[answerers[k]])
    bench_ranking = sorted(benchmarkers.keys(), key=lambda k: -alpha[benchmarkers[k]])

    # P(y=1) = σ(γ_a - α_b - δ_i) for each item given its actual (answerer, benchmarker)
    item_win_prob: dict[str, float] = {}
    for r in rows:
        key = str((r["benchmarker_model"], r["codebook_id"], r["code_id"], r["component"]))
        a = answerers[r.get("answerer_model", r["generator_model"])]
        b = benchmarkers[r["benchmarker_model"]]
        i = items[(r["benchmarker_model"], r["codebook_id"], r["code_id"], r["component"])]
        item_win_prob[key] = float(expit(gamma[a] - alpha[b] - delta[i]))

    return {
        "model":          "full_crb",
        "lambda":         lam,
        "converged":      result.success,
        "n_observations": len(rows),
        "answerer_strength":    {a: float(gamma[i]) for a, i in answerers.items()},
        "benchmarker_strength": {b: float(alpha[i]) for b, i in benchmarkers.items()},
        "item_difficulty":      {str(k): float(delta[i]) for k, i in items.items()},
        "item_win_prob":        item_win_prob,
        "answerer_ranking":     ans_ranking,
        "benchmarker_ranking":  bench_ranking,
    }


# ── Sensitivity analysis ──────────────────────────────────────────────────────

LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0]


def sensitivity_analysis(rows: list[dict]) -> dict:
    """Fit the model for each lambda in LAMBDA_GRID and report ranking stability."""
    results = {}
    rankings_by_lambda: dict[float, list] = {}

    for lam in LAMBDA_GRID:
        res = fit_full_bt(rows, lam=lam)
        results[lam] = res
        rankings_by_lambda[lam] = res["answerer_ranking"]

    # Rank correlation across lambdas (Kendall's tau average)
    from itertools import combinations

    def kendall_tau(r1: list, r2: list) -> float:
        n = len(r1)
        if n < 2:
            return 1.0
        rank1 = {v: i for i, v in enumerate(r1)}
        rank2 = {v: i for i, v in enumerate(r2)}
        keys  = list(set(r1) & set(r2))
        concordant = discordant = 0
        for a, b in combinations(keys, 2):
            s1 = rank1[a] - rank1[b]
            s2 = rank2[a] - rank2[b]
            if s1 * s2 > 0:
                concordant += 1
            elif s1 * s2 < 0:
                discordant += 1
        denom = len(keys) * (len(keys) - 1) / 2
        return (concordant - discordant) / denom if denom > 0 else 1.0

    lam_pairs = list(combinations(LAMBDA_GRID, 2))
    taus = [kendall_tau(rankings_by_lambda[a], rankings_by_lambda[b])
            for a, b in lam_pairs]
    avg_tau = float(np.mean(taus)) if taus else 1.0

    return {
        "per_lambda": {str(lam): res for lam, res in results.items()},
        "avg_kendall_tau": avg_tau,
        "ranking_stable": avg_tau >= 0.8,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(sensitivity: dict) -> None:
    print("\n=== Bradley-Terry Sensitivity Analysis ===")
    print(f"Avg Kendall's τ across λ values: {sensitivity['avg_kendall_tau']:.3f}")
    print(f"Ranking stable (τ ≥ 0.8): {sensitivity['ranking_stable']}")
    print()
    for lam_str, res in sensitivity["per_lambda"].items():
        print(f"λ = {lam_str}  (converged={res['converged']}, n={res['n_observations']})")
        print("  Answerer ranking:")
        for rank, ans in enumerate(res["answerer_ranking"], 1):
            strength = res["answerer_strength"][ans]
            print(f"    {rank}. {ans:30s}  γ = {strength:+.4f}")
        if "benchmarker_strength" in res:
            print("  Benchmarker strength:")
            for bench, alpha in res["benchmarker_strength"].items():
                print(f"    {bench:30s}  α = {alpha:+.4f}")
        print()
