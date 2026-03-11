"""
Embedding & Clustering for codebook cleaning (LOGOS-style pipeline).

Steps:
1. Load codes from gt_codes_only.json (or from a variable).
2. Embed codes with Qwen3-Embedding-0.6B (or 4B/8B).
3. Cluster with K-means; select K using Silhouette + variance scores.
4. For large codebooks: MiniBatchKMeans with batch_size=1000.

Requires: sentence-transformers>=2.7.0, transformers>=4.51.0, scikit-learn
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Default model (paper uses 0.6B; 4B/8B available for better quality)
from paths import CLUSTERED_CODES_PATH, DATA_DIR, GT_CODES_ONLY_PATH, WEIGHTS_DIR, display_path

QWEN_EMBED_MODEL = str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B") if (WEIGHTS_DIR / "Qwen3-Embedding-0.6B").is_dir() else "Qwen/Qwen3-Embedding-0.6B"
MINIBATCH_SIZE = 1000
K_MIN = 2
K_MAX_DIVISOR = 3  # max K = n_codes // K_MAX_DIVISOR, capped at 50


def load_codes(path: str = str(GT_CODES_ONLY_PATH)):
    """Load all_codes and codes_per_review from JSON produced by gt_agents.py."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    all_codes = data["all_codes"]
    codes_per_review = [tuple(x) for x in data["codes_per_review"]]
    return all_codes, codes_per_review


def embed_codes(codes: list[str], model_name: str = QWEN_EMBED_MODEL, batch_size: int = 64) -> np.ndarray:
    """Embed code strings with Qwen3-Embedding. Returns (n_codes, dim) float32 array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # Document-style encoding (no query prompt); batch for memory
    embeddings = model.encode(codes, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def select_k(embeddings: np.ndarray, use_minibatch: bool, batch_size: int = MINIBATCH_SIZE) -> int:
    """
    Select number of clusters using Silhouette score and variance (inertia).
    Tries K in [K_MIN, min(50, n//K_MAX_DIVISOR)] and picks K that maximizes silhouette.
    """
    n = len(embeddings)
    k_max = min(50, max(K_MIN + 1, n // K_MAX_DIVISOR))
    k_range = range(K_MIN, k_max + 1)

    best_k = K_MIN
    best_sil = -1.0
    results = []

    for k in k_range:
        if use_minibatch:
            km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42, n_init=3)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, sample_size=min(5000, n))
        inertia = km.inertia_
        results.append((k, sil, inertia))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print("K selection (Silhouette + Inertia):")
    for k, sil, inertia in results:
        mark = " <-- chosen" if k == best_k else ""
        print(f"  K={k:3d}  silhouette={sil:.4f}  inertia={inertia:.0f}{mark}")
    return best_k


def cluster_codes(
    embeddings: np.ndarray,
    k: int,
    use_minibatch: bool,
    batch_size: int = MINIBATCH_SIZE,
) -> np.ndarray:
    """Return cluster labels (0 .. k-1) for each code."""
    if use_minibatch:
        km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42, n_init=3)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def run_pipeline(
    codes_path: str = str(GT_CODES_ONLY_PATH),
    model_name: str = QWEN_EMBED_MODEL,
    out_dir: str = str(DATA_DIR),
    k_fixed: Optional[int] = None,
    save_embeddings: bool = False,
) -> dict:
    """
    Load codes -> embed -> select K (or use k_fixed) -> cluster -> save results.
    Returns dict with all_codes, embeddings, labels, k, codes_per_review, cluster_to_codes.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_codes, codes_per_review = load_codes(codes_path)
    n_codes = len(all_codes)
    use_minibatch = n_codes >= 1000

    print(f"Loaded {n_codes} codes from {len(codes_per_review)} reviews.")
    print(f"Embedding with {model_name} ...")
    embeddings = embed_codes(all_codes, model_name=model_name)

    if k_fixed is not None:
        k = k_fixed
        print(f"Using fixed K={k}")
    else:
        k = select_k(embeddings, use_minibatch=use_minibatch)
        print(f"Selected K={k}")

    print("Clustering ...")
    labels = cluster_codes(embeddings, k, use_minibatch=use_minibatch)

    # Build cluster -> list of code strings (for codebook cleaning)
    cluster_to_codes = defaultdict(list)
    for code, label in zip(all_codes, labels):
        cluster_to_codes[int(label)].append(code)

    # Save outputs
    out_codes = out_path / "gt_clustered_codes.json"
    with open(out_codes, "w", encoding="utf-8") as f:
        json.dump({
            "all_codes": all_codes,
            "labels": labels.tolist(),
            "k": k,
            "codes_per_review": codes_per_review,
            "cluster_to_codes": {str(i): codes for i, codes in sorted(cluster_to_codes.items())},
        }, f, indent=2)
    print(f"Saved cluster mapping to {display_path(out_codes)}")

    if save_embeddings:
        np.save(out_path / "gt_code_embeddings.npy", embeddings)
        print(f"Saved embeddings to {out_path / 'gt_code_embeddings.npy'}")

    return {
        "all_codes": all_codes,
        "embeddings": embeddings,
        "labels": labels,
        "k": k,
        "codes_per_review": codes_per_review,
        "cluster_to_codes": dict(cluster_to_codes),
    }


def main():
    parser = argparse.ArgumentParser(description="Embed codes with Qwen3-Embedding and cluster (LOGOS-style).")
    parser.add_argument("--codes", default=str(GT_CODES_ONLY_PATH), help="Path to gt_codes_only.json")
    parser.add_argument("--model", default=QWEN_EMBED_MODEL, help="Qwen3-Embedding model name")
    parser.add_argument("--out-dir", default=str(DATA_DIR), help="Output directory")
    parser.add_argument("--k", type=int, default=None, help="Fixed K (default: auto from Silhouette + variance)")
    parser.add_argument("--save-embeddings", action="store_true", help="Save .npy embeddings")
    args = parser.parse_args()

    run_pipeline(
        codes_path=args.codes,
        model_name=args.model,
        out_dir=args.out_dir,
        k_fixed=args.k,
        save_embeddings=args.save_embeddings,
    )


if __name__ == "__main__":
    main()
