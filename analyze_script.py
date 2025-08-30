import re
import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional pretty export
try:
    import dataframe_image as dfi
    _HAS_DFI = True
except Exception:
    _HAS_DFI = False

# ===================== CONFIG =====================
ROOT = Path("/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators/Beluga_all_ablations")
EXPERT_DIR_PATTERN = r"^\d+_experts$"
RESULT_DIR_PATTERN = r"^Results_5fold_.*"

VAL_CSV_NAME  = "metadata_results_val.csv"
TEST_CSV_NAME = "metadata_results_test.csv"

# Fallback CMs only if CSVs are missing
CONF_TEST_FNAME = "test_conf_matrix.npy"
CONF_VAL_FNAME  = "val_conf_matrix.npy"

# Gates (optional but recommended)
GATES_TEST_FNAME = "gates_test.npy"
GATES_VAL_FNAME  = "gates_val.npy"

OUTDIR_NAME = "summary_single_head"
GLOBAL_OUTDIR = None

LEVEL_LABELS = None  # e.g. ["A","B","C"]

# ===================== SIMPLE HELPERS =====================
def ensure_outdir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    return d

def try_export_img(df: pd.DataFrame, path: Path):
    if not _HAS_DFI:
        return
    try:
        dfi.export(df, path)
    except Exception as e:
        print(f"[WARN] Failed to export image {path}: {e}")

def find_result_dirs(root: Path):
    return sorted([d for d in root.iterdir() if d.is_dir() and re.match(RESULT_DIR_PATTERN, d.name)])

def find_folds(base: Path):
    return sorted(
        [p for p in base.glob("fold_*") if p.is_dir()],
        key=lambda x: int(re.findall(r"\d+", x.name)[0]) if re.findall(r"\d+", x.name) else 999
    )

def discover_search_roots(root: Path):
    expert_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(EXPERT_DIR_PATTERN, d.name)],
        key=lambda p: int(re.findall(r"\d+", p.name)[0]) if re.findall(r"\d+", p.name) else 10**9
    )
    if expert_dirs:
        print(f"[INFO] Discovered {len(expert_dirs)} expert dirs under {root}")
        return root, expert_dirs
    if re.match(EXPERT_DIR_PATTERN, root.name):
        print(f"[INFO] ROOT is a single expert dir: {root}")
        return root.parent, [root]
    print(f"[INFO] Treating {root} as a flat results directory.")
    return root, [root]

# ===================== METRICS / CMs =====================
def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)

def metrics_from_cm(cm: np.ndarray):
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

def heatmap_mean_std(mean_cm, std_cm, labels, title, save_path):
    try:
        fig, ax = plt.subplots(figsize=(1.1*mean_cm.shape[1]+2.5, 1.1*mean_cm.shape[0]+2.5), dpi=200)
        im = ax.imshow(mean_cm, cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
        ax.set_title(f"{title} normalised CM (mean ± SD)")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                ax.text(j, i, f"{mean_cm[i,j]:.2f}\n±{std_cm[i,j]:.2f}", ha="center", va="center", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("mean")
        fig.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to save heatmap {save_path}: {e}")

# ===================== CSV PARSING (TARGETED) =====================
def clean_and_parse(pred_str: str):
    """
    Remove 'np.float32(...)' wrappers and parse the list string → [float,...]
    Matches your provided snippet; intentionally NOT general.
    """
    cleaned = re.sub(r'np\.float32\((.*?)\)', r'\1', pred_str)
    # safer than eval; switch to eval(cleaned) if you prefer exact mimic
    vals = ast.literal_eval(cleaned)
    return [float(v) for v in vals]

def cm_from_fixed_csv(csv_path: Path, num_classes_hint: Optional[int]):
    """
    Expect CSV columns: patient,label,prediction
      - label: integer class id
      - prediction: string like "[np.float32(...), ...]"
    Returns (C,C) CM and the label array (for gate alignment).
    """
    df = pd.read_csv(csv_path)
    # labels
    y_true = df["label"].astype(int).to_numpy()
    # logits → argmax labels
    logits = np.vstack(df["prediction"].apply(clean_and_parse).values)  # (N,E)
    y_pred = np.argmax(logits, axis=1).astype(int)

    C = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    if num_classes_hint is not None:
        C = max(C, int(num_classes_hint))

    cm = np.zeros((C, C), dtype=float)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < C and 0 <= p < C:
            cm[t, p] += 1.0
    return cm, y_true

def scan_max_class_index_fixed(folds, which: str) -> Optional[int]:
    """
    Read label column from {val|test}.csv to guess C.
    """
    fname = VAL_CSV_NAME if which == "val" else TEST_CSV_NAME
    max_idx = -1
    for fd in folds:
        p = fd / fname
        if not p.exists():
            continue
        try:
            y = pd.read_csv(p)["label"].astype(int).to_numpy()
            if y.size > 0:
                max_idx = max(max_idx, int(np.nanmax(y)))
        except Exception as e:
            print(f"[WARN] scan classes ({which}): failed on {p}: {e}")
    return (max_idx + 1) if max_idx >= 0 else None

def load_stack_square(folds, filename):
    mats, ref_shape = [], None
    for fd in folds:
        fpath = fd / filename
        if not fpath.exists():
            continue
        cm = np.load(fpath)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            continue
        if ref_shape is None:
            ref_shape = cm.shape
        elif cm.shape != ref_shape:
            continue
        mats.append(cm.astype(float))
    if not mats:
        return None, []
    return np.stack(mats, axis=0), []

# ===================== GATES + SPECIALIZATION =====================
def load_gates(folds, filename, reduce_k: str = "auto"):
    arrs = []
    for fd in folds:
        fpath = fd / filename
        if not fpath.exists():
            continue
        g = np.load(fpath)
        # normalize to (N,E)
        if g.ndim == 1:
            g = g[None, :]
        elif g.ndim == 2:
            pass
        elif g.ndim == 3:
            if g.shape[1] == 1:
                g = np.squeeze(g, axis=1)
            else:
                g = g.mean(axis=1) if reduce_k in ("auto", "mean") else g.sum(axis=1)
        else:
            continue
        g = g.astype(float)
        rs = g.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.divide(g, rs, out=np.zeros_like(g), where=rs > 0)
        arrs.append(g)
    if not arrs:
        return None, []
    return arrs, []

def plot_gate_means_across_folds(gates_list, save_path_bar, save_path_heatmap=None):
    try:
        per_fold_means = [g.mean(axis=0) for g in gates_list]  # (E,)
        M = np.stack(per_fold_means, axis=0)                   # (F,E)
        mean = M.mean(axis=0); std = M.std(axis=0, ddof=0)
        E = mean.shape[0]; x = np.arange(E)
        fig, ax = plt.subplots(figsize=(max(6, 0.5*E + 3), 4), dpi=200)
        ax.bar(x, mean, yerr=std, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([f"Exp{j}" for j in x], rotation=45, ha="right")
        ax.set_ylabel("Mean gate probability"); ax.set_title("Mean gate per expert (across folds)")
        fig.tight_layout(); fig.savefig(save_path_bar, bbox_inches="tight"); plt.close(fig)
        if save_path_heatmap is not None:
            fig, ax = plt.subplots(figsize=(0.4*E + 3, 0.4*M.shape[0] + 2.5), dpi=200)
            im = ax.imshow(M, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert"); ax.set_ylabel("Fold")
            ax.set_xticks(x); ax.set_xticklabels([f"Exp{j}" for j in x], rotation=45, ha="right")
            ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"{i+1}" for i in range(M.shape[0])])
            ax.set_title("Per-fold mean gate per expert")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("mean gate")
            fig.tight_layout(); fig.savefig(save_path_heatmap, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed plotting gates mean/heatmap: {e}")

def plot_top1_expert_usage(gates_list, save_path_bar, save_path_perfold=None):
    try:
        all_top1 = np.concatenate([np.argmax(g, axis=1) for g in gates_list], axis=0)
        E = gates_list[0].shape[1]
        counts = np.array([(all_top1 == j).sum() for j in range(E)], dtype=float)
        frac = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        fig, ax = plt.subplots(figsize=(max(6, 0.5*E + 3), 4), dpi=200)
        ax.bar(np.arange(E), frac)
        ax.set_xticks(np.arange(E)); ax.set_xticklabels([f"Exp{j}" for j in range(E)], rotation=45, ha="right")
        ax.set_ylabel("Fraction of samples"); ax.set_title("Top-1 expert usage (global)")
        fig.tight_layout(); fig.savefig(save_path_bar, bbox_inches="tight"); plt.close(fig)
        if save_path_perfold is not None:
            F = len(gates_list)
            per_fold = np.zeros((F, E), dtype=float)
            for i, g in enumerate(gates_list):
                t = np.argmax(g, axis=1); 
                for j in range(E):
                    per_fold[i, j] = (t == j).mean() if t.size > 0 else 0.0
            fig, ax = plt.subplots(figsize=(0.4*E + 3, 0.4*F + 2.5), dpi=200)
            im = ax.imshow(per_fold, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert"); ax.set_ylabel("Fold")
            ax.set_xticks(range(E)); ax.set_xticklabels([f"Exp{j}" for j in range(E)], rotation=45, ha="right")
            ax.set_yticks(range(F)); ax.set_yticklabels([f"{i+1}" for i in range(F)])
            ax.set_title("Top-1 expert usage per fold")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("fraction")
            fig.tight_layout(); fig.savefig(save_path_perfold, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed plotting top1 usage: {e}")

def plot_gate_per_sample(g, y, save_path):
    try:
        if g is None or y is None or g.shape[0] != y.shape[0] or g.size == 0 or y.size == 0:
            return
        order = np.argsort(y, kind="stable")
        g_sorted = g[order]; y_sorted = y[order]
        E = g_sorted.shape[1]; N = g_sorted.shape[0]
        boundaries = np.where(np.diff(y_sorted) != 0)[0] + 1
        fig_h = min(0.02 * N + 2.5, 16)
        fig, ax = plt.subplots(figsize=(0.45*E + 3, fig_h), dpi=200)
        im = ax.imshow(g_sorted, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
        ax.set_xlabel("Expert"); ax.set_ylabel("Samples (sorted by class)")
        ax.set_xticks(range(E)); ax.set_xticklabels([f"Exp{j}" for j in range(E)], rotation=45, ha="right")
        for b in boundaries: ax.axhline(b - 0.5, color="white", linewidth=0.6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("gate prob")
        fig.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed plotting per-sample gate heatmap {save_path}: {e}")

def plot_gate_by_class(gates_list, labels_list, C, out_dir: Path, split_name: str):
    try:
        if not gates_list or not labels_list:
            return
        m_means, m_top1 = [], []
        for g, y in zip(gates_list, labels_list):
            if g is None or y is None or g.shape[0] != y.shape[0]:
                continue
            E = g.shape[1]
            class_means = np.zeros((C, E), dtype=float)
            class_fracs = np.zeros((C, E), dtype=float)
            top1 = np.argmax(g, axis=1)
            for c in range(C):
                idx = (y == c)
                if idx.any():
                    class_means[c] = g[idx].mean(axis=0)
                    for e in range(E):
                        class_fracs[c, e] = (top1[idx] == e).mean()
            m_means.append(class_means); m_top1.append(class_fracs)
        if not m_means: return
        A = np.stack(m_means, axis=0).mean(axis=0)
        B = np.stack(m_top1, axis=0).mean(axis=0)
        labels = LEVEL_LABELS if (LEVEL_LABELS and len(LEVEL_LABELS) == C) else [str(i) for i in range(C)]
        # Mean gate per class
        fig, ax = plt.subplots(figsize=(0.5*A.shape[1]+4, 0.5*A.shape[0]+4), dpi=200)
        im = ax.imshow(A, cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{split_name}: Mean gate probability per class")
        ax.set_xlabel("Expert"); ax.set_ylabel("Class")
        ax.set_xticks(range(A.shape[1])); ax.set_xticklabels([f"Exp{j}" for j in range(A.shape[1])], rotation=45, ha="right")
        ax.set_yticks(range(C)); ax.set_yticklabels(labels)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("mean gate")
        fig.tight_layout(); fig.savefig(out_dir / f"gates_mean_per_class_{split_name}.png", bbox_inches="tight"); plt.close(fig)
        # Top-1 fraction per class
        fig, ax = plt.subplots(figsize=(0.5*B.shape[1]+4, 0.5*B.shape[0]+4), dpi=200)
        im = ax.imshow(B, cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{split_name}: Top-1 expert fraction per class")
        ax.set_xlabel("Expert"); ax.set_ylabel("Class")
        ax.set_xticks(range(B.shape[1])); ax.set_xticklabels([f"Exp{j}" for j in range(B.shape[1])], rotation=45, ha="right")
        ax.set_yticks(range(C)); ax.set_yticklabels(labels)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("fraction")
        fig.tight_layout(); fig.savefig(out_dir / f"gates_top1_per_class_{split_name}.png", bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed plotting gate-by-class ({split_name}): {e}")

# ===================== PER-DIR PIPELINE =====================
def process_result_dir(res_dir: Path, rel_prefix: str = ""):
    out_dir = ensure_outdir(res_dir / OUTDIR_NAME)
    folds = find_folds(res_dir)
    if not folds:
        print(f"[INFO] Skipping {res_dir}: no fold_* directories.")
        return {"name": (rel_prefix + res_dir.name),
                "val": {"baccs": [], "wf1s": [], "accs": []},
                "test":{"baccs": [], "wf1s": [], "accs": []}}

    # class-count hints from labels
    C_val  = scan_max_class_index_fixed(folds, "val")
    C_test = scan_max_class_index_fixed(folds, "test")

    results = {"name": (rel_prefix + res_dir.name),
               "val": {"baccs": [], "wf1s": [], "accs": []},
               "test":{"baccs": [], "wf1s": [], "accs": []}}

    labels_val_list, labels_test_list = [], []

    for split, fname, cmfb, Chint in [
        ("val",  VAL_CSV_NAME,  CONF_VAL_FNAME,  C_val),
        ("test", TEST_CSV_NAME, CONF_TEST_FNAME, C_test),
    ]:
        per_fold_cms, per_fold_labels = [], []
        for fd in folds:
            csvp = fd / fname
            if csvp.exists():
                try:
                    cm, y_true = cm_from_fixed_csv(csvp, Chint)
                    per_fold_cms.append(cm); per_fold_labels.append(y_true)
                except Exception as e:
                    print(f"[WARN] {res_dir.name} {fd.name} {split}: CSV parse failed: {e}")
            else:
                print(f"[INFO] {res_dir.name} {fd.name}: missing {fname}; trying fallback CM.")

        if not per_fold_cms:
            stack_raw, _ = load_stack_square(folds, cmfb)
            if stack_raw is not None:
                per_fold_cms = [stack_raw[k] for k in range(stack_raw.shape[0])]
                per_fold_labels = [None]*len(per_fold_cms)

        if per_fold_cms:
            stack_raw = np.stack(per_fold_cms, axis=0)
            stack_norm = np.stack([row_normalize(cm) for cm in stack_raw], axis=0)
            mean_cm = stack_norm.mean(axis=0); std_cm = stack_norm.std(axis=0, ddof=0)
            C = mean_cm.shape[0]
            labels = LEVEL_LABELS if (LEVEL_LABELS and len(LEVEL_LABELS) == C) else [str(i) for i in range(C)]
            heatmap_mean_std(mean_cm, std_cm, labels, split.upper(),
                             out_dir / f"{split}_normalised_confusion_mean_std.png")

            baccs, wf1s, accs = [], [], []
            for k in range(stack_raw.shape[0]):
                b, f, a = metrics_from_cm(stack_raw[k])
                baccs.append(b); wf1s.append(f); accs.append(a)

            df = pd.DataFrame({
                "Metric": ["Balanced Accuracy", "Weighted F1", "Accuracy"],
                "Mean":   [np.mean(baccs), np.mean(wf1s), np.mean(accs)],
                "Std":    [np.std(baccs, ddof=0), np.std(wf1s, ddof=0), np.std(accs, ddof=0)]
            }).round(4)
            try_export_img(df, out_dir / f"{split}_metrics_mean_std.png")
            df.to_csv(out_dir / f"{split}_metrics_mean_std.csv", index=False)

            results[split]["baccs"] = baccs; results[split]["wf1s"] = wf1s; results[split]["accs"] = accs

            if split == "val":
                labels_val_list = per_fold_labels
                C_val = C
            else:
                labels_test_list = per_fold_labels
                C_test = C
        else:
            print(f"[WARN] {res_dir.name}: No {split} confusion matrices available.")

    # Gates (global quick stats)
    gates_test_list, _ = load_gates(folds, GATES_TEST_FNAME)
    gates_val_list,  _ = load_gates(folds, GATES_VAL_FNAME)
    g_any = gates_test_list if gates_test_list else gates_val_list
    if g_any:
        plot_gate_means_across_folds(
            g_any,
            save_path_bar=out_dir / "gates_mean_per_expert_bar.png",
            save_path_heatmap=out_dir / "gates_mean_per_expert_per_fold_heatmap.png",
        )
        plot_top1_expert_usage(
            g_any,
            save_path_bar=out_dir / "gates_top1_usage_global.png",
            save_path_perfold=out_dir / "gates_top1_usage_per_fold_heatmap.png",
        )

    # Gate specialization by class (needs labels from CSV)
    if gates_test_list and labels_test_list and any(lbl is not None for lbl in labels_test_list) and C_test is not None:
        plot_gate_by_class(gates_test_list, labels_test_list, C_test, out_dir, "test")
        for i, (g, y) in enumerate(zip(gates_test_list, labels_test_list)):
            if g is not None and y is not None and g.shape[0] == y.shape[0]:
                plot_gate_per_sample(g, y, out_dir / f"gates_per_sample_test_fold{i}.png")
    if gates_val_list and labels_val_list and any(lbl is not None for lbl in labels_val_list) and C_val is not None:
        plot_gate_by_class(gates_val_list, labels_val_list, C_val, out_dir, "val")
        for i, (g, y) in enumerate(zip(gates_val_list, labels_val_list)):
            if g is not None and y is not None and g.shape[0] == y.shape[0]:
                plot_gate_per_sample(g, y, out_dir / f"gates_per_sample_val_fold{i}.png")

    return results

# ===================== GLOBAL SUMMARY =====================
def build_global_summary(rows, save_dir: Path):
    ensure_outdir(save_dir)
    def _max_folds(rows, split, key):
        return max((len(r.get(split, {}).get(key, [])) for r in rows), default=0)

    mvb = _max_folds(rows, "val", "baccs"); mvf = _max_folds(rows, "val", "wf1s")
    mtb = _max_folds(rows, "test","baccs"); mtf = _max_folds(rows, "test","wf1s")
    vF = max(mvb, mvf); tF = max(mtb, mtf)

    cols = ["ResultDir"]
    v_b_cols = [f"Val_bAcc_fold{i}" for i in range(vF)]
    v_f_cols = [f"Val_wF1_fold{i}"  for i in range(vF)]
    t_b_cols = [f"Test_bAcc_fold{i}" for i in range(tF)]
    t_f_cols = [f"Test_wF1_fold{i}"  for i in range(tF)]
    cols += v_b_cols + ["Val_bAcc_mean","Val_bAcc_std"] + v_f_cols + ["Val_wF1_mean","Val_wF1_std"]
    cols += t_b_cols + ["Test_bAcc_mean","Test_bAcc_std"] + t_f_cols + ["Test_wF1_mean","Test_wF1_std"]

    recs = []
    for r in rows:
        rec = {"ResultDir": r["name"]}
        vb, vf = r.get("val", {}).get("baccs", []), r.get("val", {}).get("wf1s", [])
        tb, tf = r.get("test", {}).get("baccs", []), r.get("test", {}).get("wf1s", [])
        for i in range(vF):
            rec[v_b_cols[i]] = vb[i] if i < len(vb) else np.nan
            rec[v_f_cols[i]] = vf[i] if i < len(vf) else np.nan
        for i in range(tF):
            rec[t_b_cols[i]] = tb[i] if i < len(tb) else np.nan
            rec[t_f_cols[i]] = tf[i] if i < len(tf) else np.nan
        vb_arr = np.array(vb, dtype=float) if vb else np.array([])
        vf_arr = np.array(vf, dtype=float) if vf else np.array([])
        tb_arr = np.array(tb, dtype=float) if tb else np.array([])
        tf_arr = np.array(tf, dtype=float) if tf else np.array([])
        rec["Val_bAcc_mean"] = vb_arr.mean() if vb_arr.size else np.nan
        rec["Val_bAcc_std"]  = vb_arr.std(ddof=0) if vb_arr.size else np.nan
        rec["Val_wF1_mean"]  = vf_arr.mean() if vf_arr.size else np.nan
        rec["Val_wF1_std"]   = vf_arr.std(ddof=0) if vf_arr.size else np.nan
        rec["Test_bAcc_mean"] = tb_arr.mean() if tb_arr.size else np.nan
        rec["Test_bAcc_std"]  = tb_arr.std(ddof=0) if tb_arr.size else np.nan
        rec["Test_wF1_mean"]  = tf_arr.mean() if tf_arr.size else np.nan
        rec["Test_wF1_std"]   = tf_arr.std(ddof=0) if tf_arr.size else np.nan
        recs.append(rec)

    df = pd.DataFrame.from_records(recs, columns=cols).round(6)
    outp = save_dir / "summary_all_results_metrics_val_and_test.csv"
    df.to_csv(outp, index=False)
    print(f"[INFO] Wrote global summary → {outp}")

# ===================== DRIVER =====================
def main():
    global GLOBAL_OUTDIR
    base_root, search_roots = discover_search_roots(ROOT)
    GLOBAL_OUTDIR = base_root / "summary_global"

    rows_for_global = []
    total_dirs = 0
    for sr in search_roots:
        rel_prefix = "" if sr == base_root else (sr.name + "/")
        rdirs = find_result_dirs(sr)
        total_dirs += len(rdirs)
        for rd in rdirs:
            print(f"[INFO] Processing: {rd}")
            rows_for_global.append(process_result_dir(rd, rel_prefix=rel_prefix))

    if not rows_for_global:
        print("[WARN] No results processed; nothing to summarize.")
        return

    print(f"[INFO] Processed {total_dirs} result directories across {len(search_roots)} expert roots.")
    build_global_summary(rows_for_global, GLOBAL_OUTDIR)
    print("[DONE] All result directories processed.")

if __name__ == "__main__":
    main()
