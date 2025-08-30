#!/usr/bin/env python3
# Drop-in: global recipe selector with robust key canonicalization + overlap debug

import argparse
import os
import re
from pathlib import Path
import pandas as pd
from difflib import get_close_matches

def parse_args():
    p = argparse.ArgumentParser(description="Select global 'best recipe' across datasets (list input).")
    p.add_argument("csvs", nargs="+", help="List of result CSVs (one per dataset).")
    p.add_argument("--save-dir", required=True, help="Directory to save outputs (e.g., MoA root).")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Weight for std-based stability penalty (0 = disabled).")
    p.add_argument("--strip-prefix", default="Results_5fold_testfixed_",
                   help="Prefix to remove from ResultDir when forming recipe_key.")
    p.add_argument("--strip-seed", action="store_true",
                   help="Remove `_seedNN` suffixes in ResultDir.")
    p.add_argument("--dataset-name-from", choices=["filename", "parentdir"], default="parentdir",
                   help="Infer dataset name from CSV filename or parent directory.")
    p.add_argument("--recipe-key-regex", default=None,
                   help=("Optional regex with one capture group to extract recipe from ResultDir; "
                         "overrides other stripping if it matches."))
    p.add_argument("--ignore-backbone-anchor", default="MixtureOfAggregators",
                   help=("Anchor token; drop the token immediately before this anchor "
                         "(e.g., backbone like 'dinobloom-b', 'uni_features'). "
                         "Set empty '' to disable. Default: MixtureOfAggregators"))
    p.add_argument("--ignore-adapter", action="store_true",
                   help="If set, drop the 'adapter' token from the recipe key.")
    p.add_argument("--strip-experts-prefix", action="store_true",
                   help="Optionally remove the leading '<N>_experts/' segment from recipe_key.")
    p.add_argument("--debug-overlap", action="store_true",
                   help="Emit per-dataset canonicalization map and overlap report.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def infer_dataset_name(csv_path: str, mode: str) -> str:
    p = Path(csv_path)
    return p.stem if mode == "filename" else p.parent.name

def canonicalize_recipe_key(resultdir: str,
                            strip_prefix: str,
                            strip_seed: bool,
                            recipe_key_regex: str,
                            ignore_backbone_anchor: str,
                            strip_experts_prefix: bool,
                            ignore_adapter: bool) -> str:
    """Turn a noisy ResultDir into a canonical recipe key."""
    s = str(resultdir)
    # 0) lower-case & normalize slashes
    s = s.replace("\\", "/").strip().lower()

    # 1) optional regex extractor
    if recipe_key_regex:
        m = re.search(recipe_key_regex, s)
        if m:
            s = m.group(1)

    # 2) strip common prefix & seeds
    if strip_prefix and strip_prefix.lower() in s:
        s = s.replace(strip_prefix.lower(), "")
    if strip_seed:
        s = re.sub(r"_seed\d+\b", "", s)

    # 3) compact results prefix
    s = re.sub(r"(^|/)results_5fold_", r"\1r_", s)

    # 4) drop backbone immediately before the anchor
    if ignore_backbone_anchor:
        anchor = ignore_backbone_anchor.lower()
        s = re.sub(rf"/([^/]+?)_{re.escape(anchor)}", rf"/{anchor}", s)

    # 5) split into "<experts>/<rest>"
    experts_prefix = ""
    m = re.match(r"^(?P<experts>\d+_experts)/(.*)$", s)
    if m:
        experts_prefix = m.group("experts")
        s_rest = s[m.end("experts")+1:]
    else:
        s_rest = s

    # 6) Expect "mixtureofaggregators_<tokens...>"
    #    Split tokens and normalize/canonicalize them.
    #    Keep only: mode in {shared,separate}, optional 'adapter', router (topkK or dense),
    #    localhead in {localheadtrue, localheadfalse}, router_arch_*.
    # tokens like 'topk_topk2' => 'topk2'
    # preserve order: mode -> (adapter?) -> router(topkK/dense) -> localheadX -> router_arch_Y
    if "mixtureofaggregators" not in s_rest:
        # fallback: just keep the last path segment
        base = s_rest.split("/")[-1]
        canonical = base
    else:
        # take everything after mixtureofaggregators_
        parts = s_rest.split("mixtureofaggregators", 1)[1].lstrip("_")
        toks = [t for t in parts.split("_") if t]  # non-empty

        # find pieces
        mode = next((t for t in toks if t in {"shared","separate"}), "")
        adapter = "adapter" if ("adapter" in toks) else ""
        # router/topk
        # normalize 'topk','topk1','topk2','topk3', and collapse duplicates like 'topk','topk2' => 'topk2'
        k_tok = ""
        for t in toks:
            if t.startswith("topk") and len(t) > 4 and t[4:].isdigit():
                k_tok = t  # prefer explicit topkN
        if not k_tok:
            # if only bare 'topk' present without N, keep 'topk'
            if "topk" in toks:
                k_tok = "topk"
        if "dense" in toks and not k_tok:
            k_tok = "dense"

        # localhead
        lh = next((t for t in toks if t in {"localheadtrue","localheadfalse"}), "")

        # router_arch_*
        ra = ""
        for i, t in enumerate(toks):
            if t == "router" and i+2 < len(toks) and toks[i+1] == "arch":
                ra = f"router_arch_{toks[i+2]}"
                break
            if t.startswith("router_arch_"):
                ra = t
                break

        # optionally drop adapter
        if ignore_adapter:
            adapter = ""

        # build canonical order
        keep = ["mixtureofaggregators"]
        if mode: keep.append(mode)
        if adapter: keep.append(adapter)
        if k_tok: keep.append(k_tok)
        if lh: keep.append(lh)
        if ra: keep.append(ra)

        canonical = "_".join(keep)

    # 7) decide whether to keep experts prefix
    if not strip_experts_prefix and experts_prefix:
        canonical = f"{experts_prefix}/{canonical}"

    # 8) de-dup residual underscores and slashes
    canonical = re.sub(r"/+", "/", canonical).strip("/")
    canonical = re.sub(r"_+", "_", canonical)
    return canonical

def rank_within_dataset(df: pd.DataFrame, alpha: float, dataset_name: str) -> pd.DataFrame:
    for col in ["bAcc_mean", "wF1_mean"]:
        if col not in df.columns:
            raise ValueError(f"{dataset_name}: missing required column '{col}'.")
    df = df.copy()
    df["rank_bAcc"] = df["bAcc_mean"].rank(ascending=False, method="min")
    df["rank_wF1"] = df["wF1_mean"].rank(ascending=False, method="min")
    use_std = ("bAcc_std" in df.columns) and ("wF1_std" in df.columns) and (alpha > 0)
    if use_std:
        df["rank_bAcc_std"] = df["bAcc_std"].rank(ascending=True, method="min")
        df["rank_wF1_std"] = df["wF1_std"].rank(ascending=True, method="min")
        df["stability_penalty"] = alpha * (df["rank_bAcc_std"] + df["rank_wF1_std"]) / 2.0
    else:
        df["rank_bAcc_std"] = pd.NA
        df["rank_wF1_std"] = pd.NA
        df["stability_penalty"] = 0.0
    df["composite_rank"] = (df["rank_bAcc"] + df["rank_wF1"]) / 2.0 + df["stability_penalty"]
    df["dataset"] = dataset_name
    return df

def load_and_rank_one(csv_path: str, args) -> pd.DataFrame:
    dataset = infer_dataset_name(csv_path, args.dataset_name_from)
    df = pd.read_csv(csv_path)
    if "ResultDir" not in df.columns:
        raise ValueError(f"{csv_path}: 'ResultDir' column is required.")
    # map to canonical keys
    df["recipe_key"] = df["ResultDir"].astype(str).apply(
        lambda x: canonicalize_recipe_key(
            resultdir=x,
            strip_prefix=args.strip_prefix,
            strip_seed=args.strip_seed,
            recipe_key_regex=args.recipe_key_regexp if hasattr(args, "recipe_key_regexp") else args.recipe_key_regex,
            ignore_backbone_anchor=args.ignore_backbone_anchor,
            strip_experts_prefix=args.strip_experts_prefix,
            ignore_adapter=args.ignore_adapter,
        )
    )
    ranked = rank_within_dataset(df, alpha=args.alpha, dataset_name=dataset)
    keep = [
        "dataset","recipe_key","ResultDir",
        "bAcc_mean","bAcc_std","wF1_mean","wF1_std",
        "rank_bAcc","rank_wF1","rank_bAcc_std","rank_wF1_std",
        "stability_penalty","composite_rank"
    ]
    return ranked[[c for c in keep if c in ranked.columns]].copy()

def aggregate_across_datasets(per_ds: pd.DataFrame) -> pd.DataFrame:
    agg = per_ds.groupby("recipe_key").agg(
        avg_composite_rank=("composite_rank","mean"),
        n_datasets=("dataset","nunique"),
        mean_bAcc=("bAcc_mean","mean"),
        mean_wF1=("wF1_mean","mean"),
        std_composite_rank=("composite_rank","std"),
        mean_bAcc_std=("bAcc_std","mean"),
        mean_wF1_std=("wF1_std","mean"),
    ).reset_index()
    agg = agg.sort_values(
        by=["avg_composite_rank","mean_bAcc","mean_wF1"],
        ascending=[True,False,False],
        kind="mergesort"
    ).reset_index(drop=True)
    agg["global_rank"] = agg.index + 1
    return agg

def emit_overlap_debug(per_ds: pd.DataFrame, save_dir: Path):
    # map: dataset -> set(recipe_key)
    pivot = per_ds[["dataset","recipe_key","ResultDir"]].copy()
    out_map = save_dir / "canonicalization_map.csv"
    pivot.to_csv(out_map, index=False)

    # overlap summary
    datasets = sorted(pivot["dataset"].unique())
    lines = []
    sets = {d:set(pivot[pivot["dataset"]==d]["recipe_key"]) for d in datasets}
    inter = set.intersection(*sets.values()) if len(sets)>=2 else next(iter(sets.values()))
    lines.append(f"Datasets: {', '.join(datasets)}")
    lines.append(f"Overlap count (exact recipe_key match): {len(inter)}")
    for d in datasets:
        lines.append(f"- {d}: {len(sets[d])} unique recipes")

    # if no overlap, print near-misses (top 10 suggestions per dataset pair)
    if len(sets)>=2 and len(inter)==0:
        d1, d2 = datasets[0], datasets[1]
        near = []
        for key in sorted(sets[d1]):
            match = get_close_matches(key, sorted(sets[d2]), n=1, cutoff=0.6)
            if match:
                near.append((key, match[0]))
        lines.append("No exact overlaps; closest pairs:")
        for a,b in near[:10]:
            lines.append(f"  ~ {a}  <->  {b}")

    out_txt = save_dir / "overlap_report.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    return out_map, out_txt

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    per_ds = pd.concat([load_and_rank_one(p, args) for p in args.csvs], ignore_index=True)

    suffix = f"_alpha{args.alpha:g}"
    per_ds_out = Path(args.save_dir) / f"per_dataset_ranks{suffix}.csv"
    per_ds.to_csv(per_ds_out, index=False)

    if args.debug_overlap:
        cmap_csv, rep_txt = emit_overlap_debug(per_ds, Path(args.save_dir))
        print(f"[debug] Wrote canonicalization map → {cmap_csv}")
        print(f"[debug] Wrote overlap report → {rep_txt}")

    global_agg = aggregate_across_datasets(per_ds)
    global_out = Path(args.save_dir) / f"global_ranking{suffix}.csv"
    global_agg.to_csv(global_out, index=False)

    top = global_agg.iloc[0]
    print("\n===== Global Best Recipe =====")
    print(f"recipe_key         : {top['recipe_key']}")
    print(f"avg_composite_rank : {top['avg_composite_rank']:.3f}")
    print(f"n_datasets         : {int(top['n_datasets'])}")
    print(f"mean_bAcc          : {top['mean_bAcc']:.6f}")
    print(f"mean_wF1           : {top['mean_wF1']:.6f}")
    print(f"(Saved per-dataset ranks → {per_ds_out})")
    print(f"(Saved global ranking → {global_out})")

if __name__ == "__main__":
    main()
