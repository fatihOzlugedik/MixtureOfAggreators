Mixture of Aggregators (MoA)

This repository implements Mixture-of-Aggregators (MoA) for Multiple Instance Learning (MIL) on slide/patient-level feature bags.
Rather than committing to one pooling/attention scheme, MoA uses a router/gating mechanism to select or combine multiple experts per input bag.

Drop in your per-patient/slide feature files (.h5 or .pt) and 5-fold CSVs. Run train.py to train single baselines or a MoA model. The script saves per-fold metrics and confusion matrices automatically.

✨ Features

Mixture of aggregators

Router styles: topk (sparse) or dense (mixture)

Optional Gumbel-Softmax (annealed) for exploration

Load-balancing loss to reduce expert collapse

Experts: Transformer, ABMIL; sharing modes shared, separate, shared_adapter

Routers: linear, mlp, transformer

5-fold cross-validation with per-fold artifacts (.npy + .png)

Unified loader for .h5 and .pt feature bags

📦 Installation
conda env create -f environment.yaml
conda activate moa-mil
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

📁 Data layout

train.py expects a root directory with 5 subfolders named data_fold_0 … data_fold_4.
Each subfolder must contain train.csv, val.csv, test.csv. You must also pass a label mapping CSV via --label_map_csv.

csv_root/
├─ data_fold_0/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
├─ data_fold_1/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
...
├─ data_fold_4/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
└─ 3class_label_mapping.csv      # example filename (required; pass path)

CSV columns (per fold)

patient_file — path to a feature bag (.h5 or .pt)

If relative, it is resolved under --data_path; absolute paths also work.

labels — class id (int/string; mapped via --label_map_csv)

Feature files

.h5 — dataset with a tensor of shape (num_instances, feature_dim)

.pt — tensor or dict containing a tensor of shape (num_instances, feature_dim)
Select with --extension {h5,pt}.

🚀 Quick start
Baseline (single aggregator)
python train.py \
  --saving_name runs/BRACS_baseline \
  --data_path /path/to/feature_bags \
  --csv_root /path/to/csv_root \
  --label_map_csv /path/to/csv_root/3class_label_mapping.csv \
  --extension h5 \
  --arch Transformer \
  --ep 150 --es 15 --lr 5e-5 --wd 0.01 --scheduler ReduceLROnPlateau \
  --seed 38

Mixture of Aggregators (MoA)
python train.py \
  --saving_name runs/BRACS_moa \
  --data_path /path/to/feature_bags \
  --csv_root /path/to/csv_root \
  --label_map_csv /path/to/csv_root/3class_label_mapping.csv \
  --extension h5 \
  --arch Transformer \
  --expert_arch Transformer \
  --expert_mode shared \
  --router_style topk \
  --topk 2 \
  --use_local_head \
  --num_expert 4 \
  --router_type mlp \
  --use_lb_loss --lb_coef 0.01 \
  --use_gumbel --gumbel_tau_start 2.0 --gumbel_tau_min 0.5 --gumbel_decay 0.95 \
  --ep 150 --es 15 --lr 5e-5 --wd 0.01 --scheduler ReduceLROnPlateau \
  --seed 38


Notes

--arch controls the outer classifier/aggregator wrapper (e.g., Transformer-style head).

--expert_arch controls the per-expert aggregator (Transformer or ABMIL).

--num_expert sets how many experts are instantiated.

With --router_style topk, set --topk for how many experts to use per sample.

Gumbel annealing: tau(epoch) = max(gumbel_tau_min, gumbel_tau_start * gumbel_decay ** epoch).

🎛️ CLI reference (matches train.py)

Training

--lr (float, 5e-5), --grad_accum (int, 16), --wd (float, 0.01)

--scheduler (ReduceLROnPlateau), --ep (epochs, 150), --es (early stop patience, 15)

--metric {loss,f1} (default loss)

--seed (int, 38)

--checkpoint /path/to/ckpt (optional; strict load)

Data & results

--data_path (required) — root dir for feature files

--csv_root (default data_cross_val_3_classes_Bracs) — contains data_fold_{0..4}/train|val|test.csv

--label_map_csv (required) — path to label mapping CSV

--extension {h5,pt} (default h5)

--saving_name (required) — base output directory

--result_folder (unused; kept for compatibility)

--prepared_fold (flag; see “Known quirks”)

MoA / experts / router

--arch (default Transformer) — outer head

--expert_arch {Transformer,ABMIL} (default Transformer)

--expert_mode {shared,separate,shared_adapter} (default None)

--router_style {topk,dense} (default topk)

--topk (int, 1) — used when router_style=topk

--use_local_head (flag) — per-expert heads

--save_gates (flag) — save router gate activations

--num_expert (int, 1)

--router_type {linear,mlp,transformer} (default linear)

Regularization & exploration

--use_lb_loss (flag) and --lb_coef (float, 0.0)

If --lb_coef > 0 without the flag, the script still enables LB (back-compat).

--use_gumbel (flag)

--gumbel_tau_start (float, 2.0), --gumbel_tau_min (float, 0.5), --gumbel_decay (float, 0.95)

📊 Outputs

For each fold k = 0..4:

<saving_name>/<num_expert>_experts/Results_5fold_testfixed_<BACKBONE>_<arch>_<expert_mode>_<router_style>_topk<T>_localhead<flag>_router_arch_<router_type>_seed<seed>_lb<coef>_gumbel<flag>/fold_k/
├─ test_conf_matrix.npy
└─ confusion_matrix.png


<BACKBONE> is derived from the last directory name of --data_path.

If either fold_4/confusion_matrix.png or fold_4/test_conf_matrix.npy exists, the entire run is skipped (assumed complete).

A final report (runtime, config summary) is printed to the console per fold.

🧠 Reproducibility

Fixed seed (--seed, default 38) for PyTorch, CUDA, NumPy.

Multi-GPU uses nn.DataParallel automatically when available.

DataLoader workers default to 0 in the script.

🧰 Troubleshooting

Checkpoint load errors: the script strips "module." prefixes; ensure keys match.

CUDA OOM: reduce bag length (cap instances), switch --router_type mlp, reduce --num_expert, or use --topk 1.

Unexpected skip: delete fold_4/confusion_matrix.png and/or fold_4/test_conf_matrix.npy to re-run a configuration.
