import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

# ===================== CONFIG =====================
ROOT = Path("/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators")
RESULT_DIR_PATTERN = r"^Results_5fold_.*"  # tweak if needed
CONF_FNAME = "test_conf_matrix.npy"        # single-head CM per fold
GATES_FNAME = "gates_test.npy"             # per-sample gating probs per fold
OUTDIR_NAME = "summary_single_head"        # output subdir inside each result dir

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
        cm = np.load(fpath)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"{fpath} must be a square 2D matrix, got {cm.shape}")
        if ref_shape is None:
            ref_shape = cm.shape
        elif cm.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {filename}: expected {ref_shape}, found {cm.shape} at {fpath}")
        mats.append(cm.astype(float))
        paths.append(fpath)
    if not mats:
        return None
    return np.stack(mats, axis=0), paths  # (n_folds, C, C)

def load_gates(folds, filename, reduce_k: str = "auto"):
    """
    Load gate arrays per fold and return a list of (N, E) row-normalized arrays.

    Accepts the following shapes per fold:
      - (N, E)                   : already fine
      - (E,)                     : single sample -> (1, E)
      - (N, 1, E)                : top-k=1 -> squeeze K
      - (N, K, E) with K > 1     : reduce over K (mean or sum), then renormalize rows

    Args:
      reduce_k: one of {"auto","mean","sum"}.
        - "auto": if K==1 -> squeeze; if K>1 -> mean over K.
        - "mean"/"sum": explicit reduction over K.
    """
    arrs, paths = [], []
    for fd in folds:
        fpath = fd / filename
        if not fpath.exists():
            print(f"[WARN] Missing {fpath}")
            continue

        g = np.load(fpath)

        # --- Normalize shapes to (N, E) ---
        if g.ndim == 1:
            # (E,) -> (1, E)
            g = g[None, :]

        elif g.ndim == 2:
            # (N, E) -> keep
            pass

        elif g.ndim == 3:
            # Expect (N, K, E) or (N, 1, E). Handle both.
            if g.shape[1] == 1:
                # (N, 1, E) -> squeeze K
                g = np.squeeze(g, axis=1)
            else:
                # (N, K, E) with K>1
                if reduce_k == "auto" or reduce_k == "mean":
                    g = g.mean(axis=1)        # (N, E)
                elif reduce_k == "sum":
                    g = g.sum(axis=1)         # (N, E)
                else:
                    raise ValueError(f"Unknown reduce_k='{reduce_k}' for gates with shape {g.shape}")
        else:
            raise ValueError(f"{fpath} has unsupported ndim={g.ndim}, shape={g.shape}")

        if g.ndim != 2:
            raise ValueError(f"{fpath} could not be brought to (N, E); got shape {g.shape}")

        g = g.astype(float)

        # --- Row-normalize to probabilities (safe divide) ---
        row_sums = g.sum(axis=1, keepdims=True)
        g = np.divide(g, row_sums, out=np.zeros_like(g), where=row_sums > 0)

        arrs.append(g)
        paths.append(fpath)

    if not arrs:
        return None, []

    return arrs, paths

def row_normalize(cm):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

def heatmap_mean_std(mean_cm, std_cm, labels, title, save_path):
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

def metrics_from_cm(cm):
    cm = cm.astype(float)
    total = cm.sum()
    actual = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    tp = np.diag(cm)

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
    """
    gates_list: list of (N_i, E) arrays, one per fold, row-normalized.
    Produces bar (mean±SD across folds of per-expert mean gate) and optional heatmap of per-fold means.
    """
    per_fold_means = [g.mean(axis=0) for g in gates_list]  # each -> (E,)
    M = np.stack(per_fold_means, axis=0)                   # (F, E)
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=0)
    E = mean.shape[0]
    x = np.arange(E)

    # Bar with error bars
    fig, ax = plt.subplots(figsize=(max(6, 0.5*E + 3), 4), dpi=200)
    ax.bar(x, mean, yerr=std, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Exp{j}" for j in x], rotation=45, ha="right")
    ax.set_ylabel("Mean gate probability")
    ax.set_title("Mixture-of-Experts: mean gate per expert (across folds)")
    fig.tight_layout()
    fig.savefig(save_path_bar, bbox_inches="tight")
    plt.close(fig)

    # Optional heatmap: fold x expert means
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

def plot_top1_expert_usage(gates_list, save_path_bar, save_path_perfold=None):
    """
    Argmax counts (expert chosen) frequency across samples.
    Produces global bar; optionally per-fold stacked bars.
    """
    # Global
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
        # per fold fractions
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

# ===================== MAIN PIPELINE =====================
def process_result_dir(res_dir: Path):
    out_dir = ensure_outdir(res_dir / OUTDIR_NAME)
    folds = find_folds(res_dir)
    if not folds:
        print(f"[INFO] Skipping {res_dir}: no fold_* directories.")
        return

    # -------- Confusion Matrices (single head) --------
    try:
        stack_raw, cm_paths = load_stack_square(folds, CONF_FNAME)
    except ValueError as e:
        print(f"[ERROR] {res_dir.name}: {e}")
        stack_raw, cm_paths = None, []

    if stack_raw is not None:
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
        baccs, wf1s, accs = [], [], []
        for k in range(stack_raw.shape[0]):
            b, f, a = metrics_from_cm(stack_raw[k])
            baccs.append(b); wf1s.append(f); accs.append(a)
        df = pd.DataFrame(
            {
                "Metric": ["Balanced Accuracy", "Weighted F1", "Accuracy"],
                "Mean":   [np.mean(baccs), np.mean(wf1s), np.mean(accs)],
                "Std":    [np.std(baccs, ddof=0), np.std(wf1s, ddof=0), np.std(accs, ddof=0)],
            }
        ).round(4)
        # Save both image and CSV
        dfi.export(df, out_dir / "singlehead_metrics_mean_std.png")
        df.to_csv(out_dir / "singlehead_metrics_mean_std.csv", index=False)

    else:
        print(f"[WARN] {res_dir.name}: No valid confusion matrices found.")

    # -------- Gates (Mixture of Experts) --------
    try:
        gates_list, gate_paths = load_gates(folds, GATES_FNAME)
    except ValueError as e:
        print(f"[ERROR] {res_dir.name}: {e}")
        gates_list, gate_paths = None, []

    if gates_list is not None and len(gates_list) > 0:
        # Bar (mean±SD) across folds; heatmap per-fold means
        plot_gate_means_across_folds(
            gates_list,
            save_path_bar=out_dir / "gates_mean_per_expert_bar.png",
            save_path_heatmap=out_dir / "gates_mean_per_expert_per_fold_heatmap.png",
        )
        # Top-1 usage (global + per fold heatmap)
        plot_top1_expert_usage(
            gates_list,
            save_path_bar=out_dir / "gates_top1_usage_global.png",
            save_path_perfold=out_dir / "gates_top1_usage_per_fold_heatmap.png",
        )

        # Also dump a CSV summary of per-expert mean±std across folds
        per_fold_means = [g.mean(axis=0) for g in gates_list]  # list of (E,)
        M = np.stack(per_fold_means, axis=0)                   # (F, E)
        gate_df = pd.DataFrame({
            "Expert": [f"Exp{j}" for j in range(M.shape[1])],
            "Mean":   M.mean(axis=0),
            "Std":    M.std(axis=0, ddof=0),
        }).round(6)
        dfi.export(gate_df, out_dir / "gates_mean_per_expert.png")
        gate_df.to_csv(out_dir / "gates_mean_per_expert.csv", index=False)

    else:
        print(f"[WARN] {res_dir.name}: No valid gates found.")

def main():
    result_dirs = find_result_dirs(ROOT, RESULT_DIR_PATTERN)
    if not result_dirs:
        return
    print(f"[INFO] Found {len(result_dirs)} result directories.")
    for rd in result_dirs:
        print(f"[INFO] Processing: {rd}")
        process_result_dir(rd)
    print("[DONE] All result directories processed.")

if __name__ == "__main__":
    main()
