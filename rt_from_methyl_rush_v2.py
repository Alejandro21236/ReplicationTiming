
#!/usr/bin/env python3
import os, sys, argparse, io, json, math, glob, subprocess, re
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np

def read_manifest_coords(path: str) -> pd.DataFrame:
    def _detect_header_row(p: str, max_lines: int = 200) -> int:
        hdr = 0
        with open(p, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                low = line.lower()
                if (("ilmn" in low or "targetid" in low or "name" in low or "probe" in low)
                    and ("chr" in low or "mapinfo" in low or "position" in low)):
                    hdr = i
                    break
        return hdr

    ext = os.path.splitext(path)[1].lower()
    sep = "," if ext == ".csv" else ("\t" if ext in [".tsv", ".txt"] else None)
    header_row = _detect_header_row(path)

    try:
        if sep is not None:
            df = pd.read_csv(path, sep=sep, engine="c", header=header_row, dtype=str)
        else:
            df = pd.read_csv(path, sep=None, engine="python", header=header_row, dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", header=header_row, dtype=str)

    df = df.dropna(axis=1, how="all").rename(columns=lambda c: str(c).strip())
    lowmap = {c.lower(): c for c in df.columns}

    pid_cands = ["ilmnid","targetid","probe","id","name","cgid","cg_id","id_ref","composite element ref"]
    chr_cands = ["chr_hg38","chromosome_hg38","chr","chromosome","chr_hg19"]
    pos_cands = ["mapinfo_hg38","mapinfo","position","pos","mapinfo_hg19","start"]

    def pick(cands):
        for k in cands:
            if k in lowmap: return lowmap[k]
        for k in lowmap:
            if any(tag in k for tag in cands): return lowmap[k]
        return None

    pid = pick(pid_cands) or df.columns[0]
    chr_col = pick(chr_cands)
    pos_col = pick(pos_cands)
    if chr_col is None or pos_col is None:
        raise ValueError("Manifest missing recognizable chromosome/position columns. Columns seen: " + ", ".join(df.columns))

    out = df[[pid, chr_col, pos_col]].copy()
    out.columns = ["probe","chr","pos"]

    def as_chr(x: str) -> str:
        x = str(x).strip()
        if x.lower().startswith("chr"): return x
        if x.isdigit(): return "chr"+x
        if x.upper() in {"X","Y","MT","M"}: return "chr"+x.upper()
        return x

    out["chr"] = out["chr"].astype(str).map(as_chr)
    out["pos"] = pd.to_numeric(out["pos"], errors="coerce")
    out = out.dropna(subset=["pos"])
    out = out[out["chr"].str.match(r"^chr([1-9]|1[0-9]|2[0-2])$")]
    out["probe"] = out["probe"].astype(str).str.strip()
    out = out[out["probe"] != ""]

    if out.empty:
        raise ValueError("Parsed manifest is empty after filtering autosomes/NA. Check build/columns.")
    return out

def _read_txt_any(path: str) -> pd.DataFrame:
    for kw in (
        dict(sep=None, engine="python", dtype=str, comment="#", skip_blank_lines=True),
        dict(sep="\t", engine="c", dtype=str),
        dict(sep=",", engine="c", dtype=str),
    ):
        try:
            df = pd.read_csv(path, **kw)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse TXT: {path}")

def _autodetect_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = {c.lower().strip(): c for c in df.columns}
    probe_keys = ["ilmnid","targetid","probe","probe_id","cgid","cg_id","name","id","id_ref","composite element ref"]
    beta_keys  = ["avg_beta","beta","beta_value","betavalue","beta-value","beta value"]
    p = next((cols[k] for k in probe_keys if k in cols), None)
    b = next((cols[k] for k in beta_keys  if k in cols), None)
    if p is None:
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().any():
                p = c; break
    if b is None:
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 100 and (s.between(0,1).mean() > 0.5):
                b = c; break
    return p, b

def get_betas_for_sample(sample_dir: str,
                         probe_col_hint: Optional[str] = None,
                         beta_col_hint: Optional[str] = None,
                         rscript_bin: Optional[str] = None,
                         idat_converter: Optional[str] = None) -> Optional[pd.Series]:
    candidates = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))
    def _prio(p):
        n = os.path.basename(p).lower()
        return -sum(k in n for k in ("beta","avg","noob","methyl"))
    candidates.sort(key=_prio)
    txt_path = candidates[0] if candidates else None

    if txt_path is None:
        ridat = next((p for p in glob.glob(os.path.join(sample_dir, "*[Rr]ed.idat*")) if os.path.isfile(p)), None)
        gidat = next((p for p in glob.glob(os.path.join(sample_dir, "*[Gg]rn.idat*")) if os.path.isfile(p)), None)
        if ridat and gidat and rscript_bin and idat_converter and os.path.exists(idat_converter):
            out_txt = os.path.join(sample_dir, "auto_beta.txt")
            try:
                subprocess.run([rscript_bin, idat_converter, "--sample_dir", sample_dir, "--out", out_txt],
                               check=True)
                if os.path.exists(out_txt):
                    txt_path = out_txt
            except subprocess.CalledProcessError:
                txt_path = None

    if txt_path is None:
        return None

    df = _read_txt_any(txt_path)
    probe_col = probe_col_hint if (probe_col_hint and probe_col_hint in df.columns) else None
    beta_col  = beta_col_hint  if (beta_col_hint  and beta_col_hint  in df.columns) else None
    if probe_col is None or beta_col is None:
        a,b = _autodetect_cols(df)
        probe_col = probe_col or a
        beta_col  = beta_col  or b
    if probe_col is None or beta_col is None:
        return None
    s = pd.to_numeric(df[beta_col], errors="coerce").astype("float32").clip(0,1)
    idx = df[probe_col].astype(str).str.strip()
    s.index = idx
    return s.dropna()

def m_from_beta(beta: pd.Series, eps=1e-6) -> pd.Series:
    b = beta.astype("float64").clip(eps, 1-eps)
    return np.log2(b/(1-b))

def make_windows(man: pd.DataFrame, win: int, step: int) -> pd.DataFrame:
    rows = []
    for chr_, sub in man.groupby("chr"):
        if sub.empty: continue
        start = int((int(sub["pos"].min()) // step) * step)
        end   = int(((int(sub["pos"].max()) // step) + 2) * step)
        s = start
        while s < end:
            e = s + win
            rows.append((chr_, s, e))
            s += step
    win_df = pd.DataFrame(rows, columns=["chr","start","end"])
    win_df["bin_id"] = win_df["chr"].astype(str)+":"+win_df["start"].astype(str)+"-"+win_df["end"].astype(str)
    return win_df

def probe_to_bin(man: pd.DataFrame, win_df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    by_chr = {c: sub.sort_values("pos") for c, sub in man.groupby("chr")}
    for chr_, w in win_df.groupby("chr"):
        if chr_ not in by_chr: continue
        sub = by_chr[chr_]
        starts = w["start"].to_numpy()
        ends   = w["end"].to_numpy()
        pos    = sub["pos"].to_numpy(dtype=np.int64)
        idx = np.searchsorted(starts, pos, side="right") - 1
        valid = (idx >= 0) & (idx < len(starts)) & (pos < ends[idx])
        for probe, i, ok in zip(sub["probe"], idx, valid):
            if ok: out[str(probe)] = str(w.iloc[i]["bin_id"])
    return out

def rolling_median(arr: np.ndarray, win_bins: int) -> np.ndarray:
    if win_bins <= 1: return arr.copy()
    s = pd.Series(arr)
    return s.rolling(window=win_bins, center=True, min_periods=max(1, win_bins//2)).median().to_numpy()

def compute_rt_proxy_for_sample(sample_dir: str, manifest: pd.DataFrame, p2b: Dict[str,str],
                                win_df: pd.DataFrame, win: int, step: int,
                                txt_probe_hint: Optional[str], txt_beta_hint: Optional[str],
                                smooth_mb: float, min_probes: int,
                                rscript_bin: Optional[str]=None, converter_path: Optional[str]=None) -> Optional[pd.DataFrame]:
    s = get_betas_for_sample(sample_dir, probe_col_hint=txt_probe_hint, beta_col_hint=txt_beta_hint,
                             rscript_bin=rscript_bin, idat_converter=converter_path)
    if s is None or s.empty:
        return None
    m = m_from_beta(s)
    df = pd.DataFrame({"probe": m.index, "M": m.values})
    df["bin_id"] = df["probe"].map(p2b).astype("string")
    df = df.dropna(subset=["bin_id"])
    agg = df.groupby("bin_id")["M"].median()
    counts = df.groupby("bin_id")["M"].size()
    good = counts[counts >= min_probes].index
    agg = agg.loc[agg.index.intersection(good)]
    if agg.empty:
        return None
    bins = win_df.set_index("bin_id").loc[agg.index][["chr","start","end"]].reset_index()
    bins["M_med"] = agg.values

    frames = []
    smooth_bins = max(3, int(round(smooth_mb * 1_000_000 / max(step,1))))
    for chr_, sub in bins.groupby("chr", sort=False):
        sub = sub.sort_values("start").reset_index(drop=True)
        x = sub["M_med"].to_numpy()
        mu = np.nanmean(x); sd = np.nanstd(x, ddof=0); z = (x - mu) / (sd if sd>0 else 1.0)
        z_sm = rolling_median(z, win_bins=smooth_bins)
        rt_proxy = -1.0 * z_sm
        tmp = sub.copy()
        tmp["RT_proxy"] = rt_proxy
        frames.append(tmp)
    out = pd.concat(frames).sort_values(["chr","start"]).reset_index(drop=True)
    return out[["chr","start","end","bin_id","RT_proxy","M_med"]]

def call_domains_quantile(per_bin_df: pd.DataFrame, q_lo=0.25, q_hi=0.75) -> pd.DataFrame:
    v = per_bin_df["RT_proxy"].to_numpy()
    lo = np.nanquantile(v, q_lo); hi = np.nanquantile(v, q_hi)
    state = np.zeros(len(v), dtype=np.int8)
    state[v <= lo] = -1
    state[v >= hi] = +1
    df = per_bin_df.copy()
    df["state"] = state
    rows = []
    cur = None
    for _, r in df.sort_values(["chr","start"]).iterrows():
        if cur is None:
            cur = [r["chr"], r["start"], r["end"], r["state"], 1, [r["RT_proxy"]]]
        else:
            if r["chr"] == cur[0] and r["state"] == cur[3] and r["start"] == cur[2]:
                cur[2] = r["end"]; cur[4] += 1; cur[5].append(r["RT_proxy"])
            else:
                rows.append(cur); cur = [r["chr"], r["start"], r["end"], r["state"], 1, [r["RT_proxy"]]]
    if cur is not None: rows.append(cur)
    dom = pd.DataFrame(rows, columns=["chr","start","end","state","n_bins","rt_list"])
    dom["len"] = dom["end"] - dom["start"]
    dom["RT_domain"] = dom["rt_list"].apply(lambda x: float(np.nanmean(x)) if len(x)>0 else np.nan)
    dom["label"] = dom["state"].map({-1:"early",0:"mid",1:"late"})
    return dom.drop(columns=["rt_list"])

def summarize_sample(per_bin: pd.DataFrame, dom: pd.DataFrame) -> Dict[str, float]:
    total_len = float((per_bin["end"] - per_bin["start"]).sum())
    rt_mean = float(((per_bin["RT_proxy"] * (per_bin["end"]-per_bin["start"])).sum() / total_len) if total_len>0 else np.nan)
    frac_late = float((dom.loc[dom["state"]==1, "len"].sum() / total_len) if total_len>0 else 0.0)
    pmd_cov = frac_late
    return {"rt_mean": rt_mean, "frac_late": frac_late, "pmd_coverage": pmd_cov}

def list_sample_dirs(root: str) -> List[str]:
    return sorted([os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def main():
    ap = argparse.ArgumentParser(description="RT-like domain proxies from methylation (TXT + on-demand IDAT). 200–400 kb, M-values, smoothing, domains.")
    ap.add_argument("--manifest", required=True, help="Illumina manifest (CSV/TSV) with probe coordinates.")
    ap.add_argument("--methyl_root", required=True, help="Root with per-sample subdirs; each contains a TXT or an IDAT pair.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--win", type=int, default=300000)
    ap.add_argument("--step", type=int, default=150000)
    ap.add_argument("--min-probes", type=int, default=10)
    ap.add_argument("--smooth-mb", type=float, default=1.0)
    ap.add_argument("--txt-probe-col", default="IlmnID")
    ap.add_argument("--txt-beta-col",  default="Avg_Beta")
    ap.add_argument("--q-lo", type=float, default=0.25)
    ap.add_argument("--q-hi", type=float, default=0.75)
    ap.add_argument("--rscript-bin", default=os.environ.get("RSCRIPT_BIN", "Rscript"),
                    help="Path to Rscript for IDAT conversion (sesame). If missing, IDAT-only samples are skipped.")
    ap.add_argument("--idat-converter", default=os.environ.get("IDAT_CONVERTER", None),
                    help="Path to sesame_idat_to_beta.R. If missing, IDAT-only samples are skipped.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    man = read_manifest_coords(args.manifest)
    win_df = make_windows(man, win=args.win, step=args.step)
    p2b = probe_to_bin(man, win_df)

    sample_dirs = list_sample_dirs(args.methyl_root)
    summaries = []
    processed = 0
    for sdir in sample_dirs:
        sample = os.path.basename(sdir)
        per_bin = compute_rt_proxy_for_sample(
            sample_dir=sdir, manifest=man, p2b=p2b, win_df=win_df,
            win=args.win, step=args.step,
            txt_probe_hint=args.txt_probe_col, txt_beta_hint=args.txt_beta_col,
            smooth_mb=args.smooth_mb, min_probes=args.min_probes,
            rscript_bin=args.rscript_bin, converter_path=args.idat_converter
        )
        if per_bin is None or per_bin.empty:
            continue
        dom = call_domains_quantile(per_bin, q_lo=args.q_lo, q_hi=args.q_hi)
        summ = summarize_sample(per_bin, dom)
        summ["sample"] = sample
        summaries.append(summ)

        out_csv = os.path.join(args.outdir, f"{sample}.rt_domains.csv")
        dom_out = dom.copy()
        dom_out.insert(0, "sample", sample)
        dom_out.to_csv(out_csv, index=False)
        per_bin_out = os.path.join(args.outdir, f"{sample}.rt_bins.tsv")
        per_bin.to_csv(per_bin_out, sep="\t", index=False)
        processed += 1

    if processed == 0:
        print("No samples processed. Check: root path, per-sample TXT presence, or provide --rscript-bin and --idat-converter for IDATs.", file=sys.stderr)
        sys.exit(2)

    df_sum = pd.DataFrame(summaries)[["sample","rt_mean","frac_late","pmd_coverage"]]
    df_sum.to_csv(os.path.join(args.outdir, "rt_summary.tsv"), sep="\t", index=False)

    meta = {
        "win": args.win, "step": args.step, "min_probes": args.min_probes,
        "smooth_mb": args.smooth_mb, "q_lo": args.q_lo, "q_hi": args.q_hi,
        "n_samples": int(df_sum.shape[0])
    }
    with open(os.path.join(args.outdir, "rt_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
