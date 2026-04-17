import io
import contextlib
import pandas as pd
import compute_semantic_metrics
import compute_hybrid_metrics
import compute_reranked_metrics

def run_quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()

semantic = run_quiet(compute_semantic_metrics.main)
hybrid = run_quiet(compute_hybrid_metrics.main)
reranked = run_quiet(compute_reranked_metrics.main)

results = {"semantic": semantic, "hybrid": hybrid, "reranked": reranked}

print("GLOBAL")
for name, res in results.items():
    g = res["global"]
    print(f"{name}|p3={g['mean_p@3']:.4f}|r5={g['mean_r@5']:.4f}|ndcg5={g['mean_ndcg@5']:.4f}|mrr={g['mean_mrr']:.4f}")

print("MODALITY")
for name, res in results.items():
    rows_key = "per_query_rows" if name == "reranked" else "per_query"
    df = pd.DataFrame(res[rows_key])
    agg = df.groupby("modality", as_index=False)[["p@3", "ndcg@5"]].mean().sort_values("modality")
    for _, row in agg.iterrows():
        print(f"{name}|{row['modality']}|p3={row['p@3']:.4f}|ndcg5={row['ndcg@5']:.4f}")
