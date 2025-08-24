import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataframe_image is optional; we guard its usage
try:
    import dataframe_image as dfi
    _HAS_DFI = True
except Exception:
    _HAS_DFI = False

# ===================== CONFIG =====================
ROOT = Path("/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators/Beluga_full/4_experts")
RESULT_DIR_PATTERN = r"^Results_5fold_.*"  # tweak if needed
CONF_FNAME = "test_conf_matrix.npy"        # single-head CM per fold
GATES_FNAME = "gates_test.npy"             # per-sample gating probs per fold
OUTDIR_NAME = "summary_single_head"        # output subdir inside each result dir
GLOBAL_OUTDIR = ROOT / "summary_global"    # global summary folder

# Optional pretty labels for CM (default: 0..C-1)
LEVEL_LABELS = None  # e.g., ["ClassA", "ClassB", "ClassC"]

# ===================== HELPERS =====================
def find_result_dirs(root: Path, name_pattern=RESULT_DIR_PATTERN):
    dirs = [d for d in root.iterdir() if d.is_dir() and re.match(name_pattern, d.name)]
    if not dirs:
        print(f"[WARN] No result directories found under {root}")
    return sorted(dirs)

def find_folds(base: Path):
    folds = sorted([p for p in base.glob("fold_*") if p.is_dir()],
                   key=lambda x: int(re.findall(r"\d+", x.name)[0]) if re.findall(r"\d+", x.name) else 999)
    return folds

def load_stack_square(folds, filename):
    """Load square matrices (e.g., confusion matrices) from each fold and stack -> (n_folds, C, C)."""
    mats, ref_shape, paths = [], None, []
    for fd in folds:
        fpath = fd / filename
        if not fpath.exists():
            print(f"[WARN] Missing {fpath}")
            continue
        try:
            cm = np.load(fpath)
        except Exception as e:
            print(f"[WARN] Failed to load {fpath}: {e}")
            continue
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            print(f"[WARN] {fpath} must be a square 2D matrix, got {cm.shape} — skipping this fold.")
            continue
        if ref_shape is None:
            ref_shape = cm.shape
        elif cm.shape != ref_shape:
            print(f"[WARN] Shape mismatch for {filename}: expected {ref_shape}, found {cm.shape} at {fpath} — skipping.")
            continue
        mats.append(cm.astype(float))
        paths.append(fpath)
    if not mats:
        # IMPORTANT: always return a tuple
        return None, []
    return np.stack(mats, axis=0), paths  # (n_folds, C, C)

def load_gates(folds, filename, reduce_k: str = "auto"):
    """
    Load gate arrays per fold and return a list of (N, E) row-normalized arrays.
    Accepts shapes: (N,E), (E,), (N,1,E), (N,K,E) with K>1 (mean/sum over K).
    """
    arrs, paths = [], []
    for fd in folds:
        fpath = fd / filename
        if not fpath.exists():
            print(f"[WARN] Missing {fpath}")
            continue
        try:
            g = np.load(fpath)
        except Exception as e:
            print(f"[WARN] Failed to load {fpath}: {e}")
            continue

        # --- Normalize shapes to (N, E) ---
        try:
            if g.ndim == 1:
                g = g[None, :]
            elif g.ndim == 2:
                pass
            elif g.ndim == 3:
                if g.shape[1] == 1:
                    g = np.squeeze(g, axis=1)
                else:
                    if reduce_k in ("auto", "mean"):
                        g = g.mean(axis=1)
                    elif reduce_k == "sum":
                        g = g.sum(axis=1)
                    else:
                        print(f"[WARN] Unknown reduce_k='{reduce_k}' for gates with shape {g.shape} — skipping.")
                        continue
            else:
                print(f"[WARN] {fpath} has unsupported ndim={g.ndim}, shape={g.shape} — skipping.")
                continue

            if g.ndim != 2:
                print(f"[WARN] {fpath} could not be brought to (N, E); got shape {g.shape} — skipping.")
                continue
        except Exception as e:
            print(f"[WARN] Failed to process shape for {fpath}: {e} — skipping.")
            continue

        g = g.astype(float)
        row_sums = g.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.divide(g, row_sums, out=np.zeros_like(g), where=row_sums > 0)

        arrs.append(g)
        paths.append(fpath)

    if not arrs:
        return None, []
    return arrs, paths

def row_normalize(cm):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

def try_export_img(df: pd.DataFrame, path: Path):
    if not _HAS_DFI:
        print(f"[INFO] dataframe_image not available; skipping image export for {path.name}")
        return
    try:
        dfi.export(df, path)
    except Exception as e:
        print(f"[WARN] Failed to export image {path}: {e}")

def heatmap_mean_std(mean_cm, std_cm, labels, title, save_path):
    try:
        fig, ax = plt.subplots(figsize=(1.1*mean_cm.shape[1]+2.5, 1.1*mean_cm.shape[0]+2.5), dpi=200)
        im = ax.imshow(mean_cm, cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
        ax.set_title(f"{title} normalised CM (mean ± SD)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                m = mean_cm[i, j]
                s = std_cm[i, j]
                ax.text(j, i, f"{m:.2f}\n±{s:.2f}", ha="center", va="center", fontsize=9, color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("mean")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to save heatmap {save_path}: {e}")

def metrics_from_cm(cm):
    cm = cm.astype(float)
    total = cm.sum()
    actual = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    tp = np.diag(cm)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
        bacc = float(np.mean(recall)) if recall.size else 0.0

        precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
        f1_cls = np.divide(2 * precision * recall, precision + recall,
                           out=np.zeros_like(tp), where=(precision + recall) > 0)

        w = actual.sum()
        weights = np.divide(actual, w, out=np.zeros_like(actual), where=w > 0)
        wF1 = float((f1_cls * weights).sum()) if weights.sum() > 0 else 0.0

    acc = float(tp.sum() / total) if total > 0 else 0.0
    return bacc, wF1, acc

def ensure_outdir(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath

# -------- GATE VISUALS --------
def plot_gate_means_across_folds(gates_list, save_path_bar, save_path_heatmap=None):
    try:
        per_fold_means = [g.mean(axis=0) for g in gates_list]  # each -> (E,)
        M = np.stack(per_fold_means, axis=0)                   # (F, E)
        mean = M.mean(axis=0)
        std = M.std(axis=0, ddof=0)
        E = mean.shape[0]
        x = np.arange(E)

        fig, ax = plt.subplots(figsize=(max(6, 0.5*E + 3), 4), dpi=200)
        ax.bar(x, mean, yerr=std, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Exp{j}" for j in x], rotation=45, ha="right")
        ax.set_ylabel("Mean gate probability")
        ax.set_title("Mixture-of-Experts: mean gate per expert (across folds)")
        fig.tight_layout()
        fig.savefig(save_path_bar, bbox_inches="tight")
        plt.close(fig)

        if save_path_heatmap is not None:
            fig, ax = plt.subplots(figsize=(0.4*E + 3, 0.4*M.shape[0] + 2.5), dpi=200)
            im = ax.imshow(M, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert")
            ax.set_ylabel("Fold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Exp{j}" for j in x], rotation=45, ha="right")
            ax.set_yticks(range(M.shape[0]))
            ax.set_yticklabels([f"{i+1}" for i in range(M.shape[0])])
            ax.set_title("Per-fold mean gate per expert")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("mean gate")
            fig.tight_layout()
            fig.savefig(save_path_heatmap, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed plotting gates mean/heatmap: {e}")

def plot_top1_expert_usage(gates_list, save_path_bar, save_path_perfold=None):
    try:
        all_top1 = []
        for g in gates_list:
            top = np.argmax(g, axis=1)
            all_top1.append(top)
        all_top1 = np.concatenate(all_top1, axis=0)
        E = gates_list[0].shape[1]
        counts = np.array([(all_top1 == j).sum() for j in range(E)], dtype=float)
        frac = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)

        fig, ax = plt.subplots(figsize=(max(6, 0.5*E + 3), 4), dpi=200)
        ax.bar(np.arange(E), frac)
        ax.set_xticks(np.arange(E))
        ax.set_xticklabels([f"Exp{j}" for j in range(E)], rotation=45, ha="right")
        ax.set_ylabel("Fraction of samples")
        ax.set_title("Top-1 expert usage (aggregated over folds)")
        fig.tight_layout()
        fig.savefig(save_path_bar, bbox_inches="tight")
        plt.close(fig)

        if save_path_perfold is not None:
            F = len(gates_list)
            per_fold = np.zeros((F, E), dtype=float)
            for i, g in enumerate(gates_list):
                t = np.argmax(g, axis=1)
                for j in range(E):
                    per_fold[i, j] = (t == j).mean() if t.size > 0 else 0.0

            fig, ax = plt.subplots(figsize=(0.4*E + 3, 0.4*F + 2.5), dpi=200)
            im = ax.imshow(per_fold, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert")
            ax.set_ylabel("Fold")
            ax.set_xticks(range(E))
            ax.set_xticklabels([f"Exp{j}" for j in range(E)], rotation=45, ha="right")
            ax.set_yticks(range(F))
            ax.set_yticklabels([f"{i+1}" for i in range(F)])
            ax.set_title("Top-1 expert usage per fold (fraction)")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("fraction")
            fig.tight_layout()
            fig.savefig(save_path_perfold, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed plotting top1 usage: {e}")

# ===================== MAIN PIPELINE =====================
def process_result_dir(res_dir: Path):
    """
    Processes a single result directory, writes per-dir summaries/plots,
    and RETURNS per-fold metrics for global aggregation.
    """
    out_dir = ensure_outdir(res_dir / OUTDIR_NAME)
    folds = find_folds(res_dir)
    if not folds:
        print(f"[INFO] Skipping {res_dir}: no fold_* directories.")
        return {"name": res_dir.name, "baccs": [], "wf1s": []}

    # -------- Confusion Matrices (single head) --------
    baccs, wf1s, accs = [], [], []
    stack_raw, cm_paths = load_stack_square(folds, CONF_FNAME)

    if stack_raw is not None and stack_raw.size > 0:
        # Visual mean/std from row-normalized
        stack_norm = np.stack([row_normalize(cm) for cm in stack_raw], axis=0)
        mean_cm = stack_norm.mean(axis=0)
        std_cm  = stack_norm.std(axis=0, ddof=0)
        C = mean_cm.shape[0]
        labels = LEVEL_LABELS if (LEVEL_LABELS and len(LEVEL_LABELS) == C) else [str(i) for i in range(C)]
        heatmap_mean_std(mean_cm, std_cm, labels,
                         title="Single-Head",
                         save_path=out_dir / "singlehead_normalised_confusion_mean_std.png")

        # Metrics per fold from raw counts
        for k in range(stack_raw.shape[0]):
            try:
                b, f, a = metrics_from_cm(stack_raw[k])
                baccs.append(b); wf1s.append(f); accs.append(a)
            except Exception as e:
                print(f"[WARN] Failed metrics on {res_dir.name} fold {k}: {e}")

        df = pd.DataFrame(
            {
                "Metric": ["Balanced Accuracy", "Weighted F1", "Accuracy"],
                "Mean":   [np.mean(baccs) if baccs else np.nan,
                           np.mean(wf1s) if wf1s else np.nan,
                           np.mean(accs) if accs else np.nan],
                "Std":    [np.std(baccs, ddof=0) if baccs else np.nan,
                           np.std(wf1s, ddof=0) if wf1s else np.nan,
                           np.std(accs, ddof=0) if accs else np.nan],
            }
        ).round(4)
        try_export_img(df, out_dir / "singlehead_metrics_mean_std.png")
        try:
            df.to_csv(out_dir / "singlehead_metrics_mean_std.csv", index=False)
        except Exception as e:
            print(f"[WARN] Failed saving CSV in {out_dir}: {e}")
    else:
        print(f"[WARN] {res_dir.name}: No valid confusion matrices found; skipping CM-based plots.")

    # -------- Gates (Mixture of Experts) --------
    try:
        gates_list, gate_paths = load_gates(folds, GATES_FNAME)
    except Exception as e:
        print(f"[WARN] {res_dir.name}: loading gates raised {e}")
        gates_list = None

    if gates_list:
        plot_gate_means_across_folds(
            gates_list,
            save_path_bar=out_dir / "gates_mean_per_expert_bar.png",
            save_path_heatmap=out_dir / "gates_mean_per_expert_per_fold_heatmap.png",
        )
        plot_top1_expert_usage(
            gates_list,
            save_path_bar=out_dir / "gates_top1_usage_global.png",
            save_path_perfold=out_dir / "gates_top1_usage_per_fold_heatmap.png",
        )

        # Per-expert mean±std CSV (guarded)
        try:
            per_fold_means = [g.mean(axis=0) for g in gates_list]  # list of (E,)
            M = np.stack(per_fold_means, axis=0)                   # (F, E)
            gate_df = pd.DataFrame({
                "Expert": [f"Exp{j}" for j in range(M.shape[1])],
                "Mean":   M.mean(axis=0),
                "Std":    M.std(axis=0, ddof=0),
            }).round(6)
            try_export_img(gate_df, out_dir / "gates_mean_per_expert.png")
            gate_df.to_csv(out_dir / "gates_mean_per_expert.csv", index=False)
        except Exception as e:
            print(f"[WARN] {res_dir.name}: failed to write gates CSV: {e}")
    else:
        print(f"[WARN] {res_dir.name}: No valid gates found.")

    # Return what we have (even if empty) so global CSV can be built
    return {"name": res_dir.name, "baccs": baccs, "wf1s": wf1s}

def build_global_summary(rows, save_dir: Path):
    """
    rows: list of dicts like {"name": <res_dir_name>, "baccs": [...], "wf1s": [...]}
    Writes a single CSV that contains each directory's per-fold bAcc/wF1 plus mean/std.
    """
    ensure_outdir(save_dir)
    max_folds = max((len(r["baccs"]) for r in rows), default=0)
    bacc_fold_cols = [f"bAcc_fold{i}" for i in range(max_folds)]
    wf1_fold_cols  = [f"wF1_fold{i}"  for i in range(max_folds)]

    records = []
    for r in rows:
        rec = {"ResultDir": r["name"]}
        # per-fold values, pad with NaN if fewer folds
        for i in range(max_folds):
            rec[bacc_fold_cols[i]] = r["baccs"][i] if i < len(r["baccs"]) else np.nan
            rec[wf1_fold_cols[i]]  = r["wf1s"][i]  if i < len(r["wf1s"])  else np.nan
        # Mean / Std
        baccs = np.array(r["baccs"], dtype=float) if r["baccs"] else np.array([])
        wf1s  = np.array(r["wf1s"],  dtype=float) if r["wf1s"]  else np.array([])
        rec["bAcc_mean"] = baccs.mean() if baccs.size else np.nan
        rec["bAcc_std"]  = baccs.std(ddof=0) if baccs.size else np.nan
        rec["wF1_mean"]  = wf1s.mean()  if wf1s.size  else np.nan
        rec["wF1_std"]   = wf1s.std(ddof=0)  if wf1s.size  else np.nan
        records.append(rec)

    cols = ["ResultDir"] + bacc_fold_cols + ["bAcc_mean", "bAcc_std"] + wf1_fold_cols + ["wF1_mean", "wF1_std"]
    df = pd.DataFrame.from_records(records, columns=cols).round(6)
    csv_path = save_dir / "summary_all_results_metrics.csv"
    try:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Wrote global summary → {csv_path}")
    except Exception as e:
        print(f"[WARN] Failed writing global CSV {csv_path}: {e}")


def main():
    result_dirs = find_result_dirs(ROOT, RESULT_DIR_PATTERN)
    if not result_dirs:
        return
    print(f"[INFO] Found {len(result_dirs)} result directories.")

    rows_for_global = []
    for rd in result_dirs:
        print(f"[INFO] Processing: {rd}")
        row = process_result_dir(rd)
        rows_for_global.append(row)

    build_global_summary(rows_for_global, GLOBAL_OUTDIR)
    print("[DONE] All result directories processed.")


if __name__ == "__main__":
    main()
