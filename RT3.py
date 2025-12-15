#!/usr/bin/env python3
"""
ViT-based multi-output regression from H&E patches → replication timing (RT) features.

HARDENED VERSION
- Robust label handling: accepts "case id" / "case_id" / similar; normalizes & de-dupes columns.
- Skips slides with no patches; skips corrupted PNGs (oversized ICC/text chunks, truncated files).
- GPU-safe: encodes patches in CHUNKS with AMP to prevent OOM; supports --max_patches.
- No leakage: GroupKFold by case_id. Reports PCC/Spearman/MAE/MSE/R² per RT feature with BH-FDR.

Example:
python vit_rt_train.py \
  --patches_dir /fs/scratch/PAS2942/TCGA_DS_1/5x/BRCA/patches \
  --labels_csv /fs/scratch/PAS2942/Alejandro/datasets/5x_RT.csv \
  --out_dir /fs/scratch/PAS2942/Alejandro/outputs/rt_vit \
  --epochs 50 --folds 5 --max_patches 128 --batch_size 1
"""

import os, math, random, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError, PngImagePlugin, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# NEW: plotting (matplotlib only; no seaborn dependency)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pillow robustness for weird PNG metadata
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # allow large iCCP/text chunks
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Backbone factory (ViT)
# =========================
_BACKBONE_DIM = 768
try:
    import timm
    def make_vit():
        return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
except Exception:
    timm = None
    from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
    def make_vit():
        m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        m.heads = nn.Identity()
        return m

# =========================
# Utils
# =========================

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    new_df = pd.DataFrame(index=df.index)
    for col in pd.unique(df.columns):
        same = [c for c in df.columns if c == col]
        sub = df[same]
        if isinstance(sub, pd.Series):
            new_df[col] = sub
        else:
            new_df[col] = sub.bfill(axis=1).iloc[:, 0]
    return new_df

def find_case_id_column(df: pd.DataFrame) -> str:
    candidates = ["case_id", "case", "caseid", "case_id_slides", "tcga_id", "patient", "patient_id"]
    for c in candidates:
        if c in df.columns: return c
    # fuzzy
    for c in df.columns:
        if c.replace("_", "") in {"caseid", "caseids"}: return c
    raise ValueError(f"Could not find a case id column. Columns: {list(df.columns)[:30]} ...")

def benjamini_hochberg(pvals: List[float], q: float = 0.05) -> List[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    sp = np.array(pvals)[order]
    thresh = q * (np.arange(1, m + 1) / m)
    sig_sorted = sp <= thresh
    if np.any(sig_sorted):
        last_true = np.where(sig_sorted)[0].max()
        sig_sorted[: last_true + 1] = True
        sig_sorted[last_true + 1 :] = False
    sig = np.zeros_like(sig_sorted, dtype=bool); sig[order] = sig_sorted
    return sig.tolist()

# =========================
# Dataset
# =========================

class SlideBagDataset(Dataset):
    def __init__(self, slide_dirs, case_ids, case_targets, patches_root, max_patches=128, train=True):
        self.slide_dirs, self.case_ids = slide_dirs, case_ids
        self.case_targets, self.patches_root = case_targets, patches_root
        self.max_patches, self.train = max_patches, train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def __len__(self):
        return len(self.slide_dirs)
    def _list_pngs(self, slide_dir):
        return [str(p) for p in (Path(self.patches_root) / slide_dir).glob("*.png")]
    def _load_image(self, path):
        try:
            img = Image.open(path)
            img.load()  # force decode early
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return None
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        if self.train and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return (x - self.mean) / self.std
    def __getitem__(self, idx):
        slide_dir, case_id = self.slide_dirs[idx], self.case_ids[idx]
        target = torch.tensor(self.case_targets[case_id], dtype=torch.float32)
        files = self._list_pngs(slide_dir)
        if len(files) == 0:
            return None
        if self.train:
            random.shuffle(files)
        imgs = []
        for p in files:
            x = self._load_image(p)
            if x is not None:
                imgs.append(x)
            if len(imgs) >= self.max_patches:
                break
        if len(imgs) == 0:
            return None
        X = torch.stack(imgs, 0)
        return X, target, case_id, slide_dir

# Safe collate: drop Nones

def safe_collate(batch):
    item = batch[0]
    return item if item is not None else None

# =========================
# Model
# =========================

class AttnMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=256):
        super().__init__()
        self.a = nn.Linear(in_dim, attn_dim)
        self.tanh = nn.Tanh()
        self.b = nn.Linear(attn_dim, 1)
    def forward(self, H):
        A = self.b(self.tanh(self.a(H))).squeeze(-1)
        w = torch.softmax(A, dim=0)
        z = (w.unsqueeze(-1) * H).sum(0)
        return z, w

class WSItoRT(nn.Module):
    def __init__(self, out_dim, freeze_backbone_blocks=0):
        super().__init__()
        self.backbone = make_vit()
        self.mil = AttnMIL(_BACKBONE_DIM, 256)
        self.head = nn.Sequential(
            nn.Linear(_BACKBONE_DIM, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, out_dim)
        )
        if timm is not None and freeze_backbone_blocks > 0:
            blocks = getattr(self.backbone, "blocks", [])
            for i, blk in enumerate(blocks):
                if i < freeze_backbone_blocks:
                    for p in blk.parameters():
                        p.requires_grad = False
    def encode_patches(self, X, chunk: int = 64):
        feats = []
        for i in range(0, X.size(0), chunk):
            Xi = X[i:i+chunk]
            with autocast(enabled=True):
                h = self.backbone(Xi)  # [chunk, d]
            feats.append(h)
        return torch.cat(feats, dim=0)
    def forward(self, X):
        H = self.encode_patches(X)
        z, w = self.mil(H)
        y = self.head(z)
        return y, z, w

# =========================
# NEW: Figure helpers
# =========================

def _denorm(img_t: torch.Tensor) -> np.ndarray:
    # img_t: [3, H, W] normalized by ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(img_t.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(img_t.device)
    x = (img_t * std + mean).clamp(0,1).cpu().permute(1,2,0).numpy()
    return (x * 255).astype(np.uint8)

def save_attention_weights_plot(weights: np.ndarray, out_png: str, title: str):
    order = np.argsort(weights)[::-1]
    ws = weights[order]
    plt.figure(figsize=(6,3))
    plt.plot(np.arange(len(ws)), ws, linewidth=2)
    plt.xlabel("Patches (ranked)")
    plt.ylabel("Attention weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def save_attention_top_patches_contactsheet(X: torch.Tensor, weights: torch.Tensor, out_png: str, k: int = 25, cols: int = 5):
    # X: [N,3,224,224]; weights: [N]
    k = min(k, X.size(0))
    order = torch.argsort(weights, descending=True)[:k].cpu().tolist()
    rows = (k + cols - 1) // cols
    cell = 224
    canvas = Image.new("RGB", (cols*cell, rows*cell), (255,255,255))
    for i, idx in enumerate(order):
        r, c = divmod(i, cols)
        img = Image.fromarray(_denorm(X[idx]))
        canvas.paste(img, (c*cell, r*cell))
    # border & caption (simple)
    canvas.save(out_png)

# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, optimizer, device, loss_fn, scaler: GradScaler):
    model.train(); tot = 0.0; n = 0
    for b in loader:
        if b is None or (isinstance(b, (list, tuple)) and b[0] is None):
            continue
        X, y, _, _ = b
        X = X.to(device)
        y = y.to(device).view(-1)  # [K]
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            y_hat, _, _ = model(X)
            loss = loss_fn(y_hat, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot += loss.item(); n += 1
    return tot / max(1, n)

@torch.no_grad()
def eval_slides(model, loader, device, figs_dir: str | None = None, attn_examples_per_fold: int = 3):
    model.eval(); preds_by_case, targets_by_case, slides_by_case = {}, {}, {}
    saved_examples = 0
    for b in loader:
        if b is None or (isinstance(b, (list, tuple)) and b[0] is None):
            continue
        X, y, case_ids, slide_dirs = b
        X = X.to(device)
        y = y.to(device).view(-1)
        with autocast(enabled=True):
            y_hat, z, w = model(X)
        cid = case_ids[0] if isinstance(case_ids, (list, tuple)) else case_ids
        sd  = slide_dirs[0] if isinstance(slide_dirs, (list, tuple)) else slide_dirs
        preds_by_case.setdefault(cid, []).append(y_hat.cpu().numpy())
        targets_by_case[cid] = y.cpu().numpy()
        slides_by_case.setdefault(cid, []).append(sd)

        # NEW: save attention visuals for a few validation slides
        if figs_dir is not None and saved_examples < attn_examples_per_fold:
            ensure_dir(figs_dir)
            w_cpu = w.detach().float().cpu().numpy()
            save_attention_weights_plot(
                w_cpu,
                os.path.join(figs_dir, f"attn_{cid}_{sd}_weights.png"),
                title=f"Attention weights — {cid}/{sd}"
            )
            # contact sheet of top patches (denorm from X)
            save_attention_top_patches_contactsheet(
                X.detach().float().cpu(), w.detach().float().cpu(),
                os.path.join(figs_dir, f"attn_{cid}_{sd}_toppatches.png"),
                k=25, cols=5
            )
            saved_examples += 1
    return preds_by_case, targets_by_case, slides_by_case

@torch.no_grad()
def compute_metrics(preds_by_case, targets_by_case, target_names: List[str]) -> pd.DataFrame:
    cases = sorted(targets_by_case.keys())
    y_true = np.stack([targets_by_case[c] for c in cases], axis=0)
    y_pred = np.stack([np.mean(np.stack(preds_by_case[c], axis=0), axis=0) for c in cases], axis=0)
    rows, pearson_ps = [], []
    for k, name in enumerate(target_names):
        t, p = y_true[:, k], y_pred[:, k]
        if np.std(t) < 1e-8 or np.std(p) < 1e-8:
            pr = np.nan; pr_p = 1.0; sr = np.nan; sr_p = 1.0
        else:
            pr, pr_p = pearsonr(t, p)
            sr, sr_p = spearmanr(t, p)
        mae = mean_absolute_error(t, p)
        mse = mean_squared_error(t, p)
        r2  = r2_score(t, p)
        rows.append({"feature": name, "pearson_r": pr, "pearson_p": pr_p, "spearman_rho": sr, "spearman_p": sr_p, "mae": mae, "mse": mse, "r2": r2})
        pearson_ps.append(pr_p)
    dfm = pd.DataFrame(rows)
    dfm["pearson_fdr_sig"] = benjamini_hochberg(pearson_ps, q=0.05)
    return dfm

# =========================
# Main
# =========================

def default_target_columns(df: pd.DataFrame) -> List[str]:
    if "rt_mean" not in df.columns:
        raise ValueError("rt_mean not found in labels file.")
    return ["rt_mean"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", required=True, type=str)
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1, help="slides per batch; keep 1 for variable bags")
    ap.add_argument("--max_patches", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--freeze_blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_cols", type=str, default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    # Load labels
    labels = pd.read_csv(args.labels_csv)
    labels = standardize_columns(labels)
    labels = dedupe_columns(labels)
    case_col = find_case_id_column(labels)
    series = labels[case_col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    labels["case_id"] = series.astype(str).str.strip().str.lower()

    # Targets
    if args.target_cols.strip():
        target_cols = [c.strip().lower().replace(" ", "_") for c in args.target_cols.split(",")]
    else:
        target_cols = default_target_columns(labels)

    case_targets_df = labels.groupby("case_id")[target_cols].mean().reset_index()
    case_targets = {row["case_id"]: row[target_cols].to_numpy(dtype=np.float32) for _, row in case_targets_df.iterrows()}

    # Slides
    slide_dirs = sorted([d for d in os.listdir(args.patches_dir) if os.path.isdir(os.path.join(args.patches_dir, d))])
    slide_case_ids = [d[:12].lower() for d in slide_dirs]

    # Keep slides with labels
    keep = [cid in case_targets for cid in slide_case_ids]
    slide_dirs = [s for s, k in zip(slide_dirs, keep) if k]
    slide_case_ids = [c for c, k in zip(slide_case_ids, keep) if k]

    # Drop slides that have no patches up-front
    counts = [len(list((Path(args.patches_dir) / s).glob("*.png"))) for s in slide_dirs]
    has_png = [c > 0 for c in counts]
    dropped = sum(1 for h in has_png if not h)
    slide_dirs = [s for s, h in zip(slide_dirs, has_png) if h]
    slide_case_ids = [c for c, h in zip(slide_case_ids, has_png) if h]
    print(f"Slides with labels & patches: {len(slide_dirs)} (dropped {dropped} empty)")

    # CV
    gkf = GroupKFold(n_splits=args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    fold_summaries = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(slide_dirs, groups=slide_case_ids), 1):
        print(f"===== Fold {fold}/{args.folds} =====")
        tr_slides = [slide_dirs[i] for i in tr_idx]
        va_slides = [slide_dirs[i] for i in va_idx]
        tr_cases  = [slide_case_ids[i] for i in tr_idx]
        va_cases  = [slide_case_ids[i] for i in va_idx]

        train_ds = SlideBagDataset(tr_slides, tr_cases, case_targets, args.patches_dir, max_patches=args.max_patches, train=True)
        valid_ds = SlideBagDataset(va_slides, va_cases, case_targets, args.patches_dir, max_patches=args.max_patches, train=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=safe_collate)
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=safe_collate)

        model = WSItoRT(out_dim=len(target_cols), freeze_backbone_blocks=args.freeze_blocks).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.SmoothL1Loss(beta=1.0)
        scaler = GradScaler()

        best_val = math.inf
        best_path = os.path.join(args.out_dir, f"fold{fold}_best.pt")

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler)
            preds_by_case, targets_by_case, _ = eval_slides(model, valid_loader, device)
            if len(preds_by_case) == 0:
                print("[WARN] Validation set produced no predictions (all slides skipped). Skipping epoch eval.")
                continue
            metrics_df = compute_metrics(preds_by_case, targets_by_case, target_cols)
            val_mse_mean = float(metrics_df["mse"].mean())
            print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_mse_mean={val_mse_mean:.4f}")
            if val_mse_mean < best_val:
                best_val = val_mse_mean
                torch.save({"model": model.state_dict(), "target_cols": target_cols}, best_path)

        # Final eval
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
        fold_dir = os.path.join(args.out_dir, f"fold{fold}")
        ensure_dir(fold_dir)
        figs_dir = os.path.join(fold_dir, "figs")
        ensure_dir(figs_dir)

        # eval with figure saving (attention examples)
        preds_by_case, targets_by_case, slides_by_case = eval_slides(model, valid_loader, device, figs_dir)

        metrics_df = compute_metrics(preds_by_case, targets_by_case, target_cols)

        mean_pcc = float(metrics_df["pearson_r"].mean(skipna=True)) if not metrics_df["pearson_r"].isna().all() else float('nan')
        n_pcc30 = int((metrics_df["pearson_r"] >= 0.30).sum())
        n_pcc50 = int((metrics_df["pearson_r"] >= 0.50).sum())
        summary = {"fold": fold, "mean_pearson_r": mean_pcc, "n_features_pcc_ge_0.30": n_pcc30, "n_features_pcc_ge_0.50": n_pcc50, "val_mse_mean": float(metrics_df["mse"].mean())}
        fold_summaries.append(summary)

        metrics_df.to_csv(os.path.join(fold_dir, "metrics_per_feature.csv"), index=False)

        # Per-case aggregated predictions
        cases = sorted(targets_by_case.keys())
        agg_rows = []
        for c in cases:
            y_true = targets_by_case[c]
            y_pred = np.mean(np.stack(preds_by_case[c], axis=0), axis=0)
            row = {"case_id": c}
            for k, name in enumerate(target_cols):
                row[f"true_{name}"] = y_true[k]
                row[f"pred_{name}"] = y_pred[k]
            row["slides"] = ",".join(slides_by_case.get(c, []))
            agg_rows.append(row)
        per_case_csv = os.path.join(fold_dir, "per_case_predictions.csv")
        pd.DataFrame(agg_rows).to_csv(per_case_csv, index=False)

        # NEW (2): scatter plot true vs predicted (first target)
        if len(target_cols) > 0:
            df_pred = pd.read_csv(per_case_csv)
            tcol = f"true_{target_cols[0]}"
            pcol = f"pred_{target_cols[0]}"
            if tcol in df_pred.columns and pcol in df_pred.columns:
                t = df_pred[tcol].values
                p = df_pred[pcol].values
                lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
                plt.figure(figsize=(4.5,4.5))
                plt.scatter(t, p, alpha=0.7)
                plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
                plt.xlabel(f"True {target_cols[0]}")
                plt.ylabel(f"Predicted {target_cols[0]}")
                plt.title(f"Fold {fold}: Predicted vs True")
                plt.tight_layout()
                plt.savefig(os.path.join(figs_dir, "scatter_rt_mean.png"), dpi=300)
                plt.close()

    # Save CV summary
    cv_csv = os.path.join(args.out_dir, "cv_summary.csv")
    pd.DataFrame(fold_summaries).to_csv(cv_csv, index=False)
    print("Done. CV summary written.")

    # NEW (4): heatmap of fold-wise mean_pearson_r and val_mse_mean
    try:
        df = pd.read_csv(cv_csv)
        # Build a 2 x F matrix
        vals = np.vstack([df["mean_pearson_r"].values, df["val_mse_mean"].values])
        plt.figure(figsize=(1.2*df.shape[0]+2, 3.8))
        im = plt.imshow(vals, aspect='auto', cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks([0,1], ["mean_pearson_r","val_mse_mean"])
        plt.xticks(np.arange(df.shape[0]), [f"F{int(f)}" for f in df["fold"].values])
        plt.title("Cross-validation summary")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "cv_heatmap.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not create CV heatmap: {e}")

if __name__ == "__main__":
    main()
