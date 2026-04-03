#!/usr/bin/env python3
# inference.py
"""
BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction
(arXiv: 2602.04512)

Long-horizon autoregressive rollout inference.

Given pretrained predictor and network-wise tokenizer decoders, predicts future
fMRI voxel trajectories for H timesteps via autoregressive circuit token rollout:
  x_hat^{T+h} = g(f_theta(x_hat^{<=T+h-1}, s^{<=T+h}))
Ground-truth stimulus tokens are provided; only circuit tokens are rolled out.

Outputs per run:
  - {rid}_pred.npy: predicted voxel trajectories [H, P_total]
  - {rid}_true.npy: ground truth voxel trajectories [H, P_total]
  - {rid}_spatial_attn.npy: K x K network coupling attention [H, nhead, K, K]
  - inference_summary.json: per-run Pearson r and global summary

Usage:
  python inference.py \\
    --predictor_ckpt path/to/predictor_best.pt \\
    --video_root path/to/video_features \\
    --audio_root path/to/audio_features \\
    --text_root path/to/text_features \\
    --z_root path/to/circuit_tokens \\
    --run_list_file path/to/test_list.txt \\
    --parcels Cont DorsAttn SalVentAttn Limbic SomMot Default Vis \\
    --tokenizer_ckpt_dir path/to/tokenizer_ckpts \\
    --tokenizer_name_template "ckpt_tokenizer_{parcel}/best.pt" \\
    --K_horizon 20 \\
    --out_dir ./predictions
"""

import os
import re
import json
import argparse
import math

import numpy as np
from tqdm import tqdm
import torch

from utils.models import InterleavedStim2BrainSpatialPredictor
from utils.data_utils import (
    parse_run_list,
    load_stimulus_and_circuit_tokens,
    load_all_decoders,
)


# ===================== Metrics =====================

def pearson_r(x, y, eps=1e-8):
    """Pearson correlation between two flat arrays."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    xm, ym = x - x.mean(), y - y.mean()
    num = (xm * ym).sum()
    den = math.sqrt((xm**2).sum()) * math.sqrt((ym**2).sum()) + eps
    return float(num / den)


def pattern_corr(X_true, X_pred, eps=1e-8):
    """Pattern correlation (p_corr): spatial correlation averaged over time.

    p_corr = (1/T) * sum_t corr(y_hat_{t,:}, y_{t,:})
    Less sensitive to drift under long-horizon rollout.
    """
    T = X_true.shape[0]
    if T == 0:
        return 0.0
    rs = []
    for t in range(T):
        xt = X_true[t].astype(np.float64)
        xp = X_pred[t].astype(np.float64)
        xt, xp = xt - xt.mean(), xp - xp.mean()
        den = math.sqrt((xt**2).sum()) * math.sqrt((xp**2).sum()) + eps
        rs.append(float((xt * xp).sum() / den))
    return float(np.mean(rs))


# ===================== Model loading =====================

def _infer_spatial_from_state_dict(sd):
    K, L = 0, 0
    pat_k = re.compile(r"^spatial_mixer\.in_projs\.(\d+)\.")
    pat_l = re.compile(r"^spatial_mixer\.layers\.(\d+)\.")
    for name in sd.keys():
        mk = pat_k.match(name)
        if mk:
            K = max(K, int(mk.group(1)) + 1)
        ml = pat_l.match(name)
        if ml:
            L = max(L, int(ml.group(1)) + 1)
    return {"spatial_K": K, "spatial_nlayers": L}


def build_model_from_ckpt(ckpt_path, Dv, Da, Dw, Dz, z_splits, device):
    """Build BrainVista predictor from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    sd = ckpt.get("state_dict", {})

    d_model = int(cfg.get("d_model", 256))
    nhead = int(cfg.get("nhead", 8))
    nlayers = int(cfg.get("nlayers", 4))
    dropout = float(cfg.get("dropout", 0.1))
    use_hrf = bool(cfg.get("use_hrf", 1))
    brain_ar = bool(cfg.get("brain_ar", 1))
    L = int(cfg.get("L", 40))
    Lb = int(cfg.get("Lb", 40))
    pred_delta_z = bool(cfg.get("pred_delta_z", 1))

    inferred = _infer_spatial_from_state_dict(sd)
    spatial_nlayers = int(cfg.get("spatial_nlayers", 0)) or inferred["spatial_nlayers"]
    spatial_nhead = int(cfg.get("spatial_nhead", 4))
    spatial_dropout = float(cfg.get("spatial_dropout", 0.1))

    print(f"[Model] d_model={d_model}, nhead={nhead}, nlayers={nlayers}, "
          f"L={L}, Lb={Lb}, spatial_nhead={spatial_nhead}, spatial_nlayers={spatial_nlayers}")

    model = InterleavedStim2BrainSpatialPredictor(
        Dv=Dv, Da=Da, Dw=Dw,
        Dz_in=Dz, Dz_out=Dz,
        d_model=d_model, nhead=nhead, nlayers=nlayers, dropout=dropout,
        use_hrf=use_hrf, brain_ar=brain_ar,
        L=L, Lb=Lb,
        z_splits=z_splits,
        spatial_nhead=spatial_nhead,
        spatial_nlayers=spatial_nlayers,
        spatial_dropout=spatial_dropout,
    )

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, {"L": L, "Lb": Lb, "pred_delta_z": pred_delta_z, "cfg": cfg}


# ===================== Inference =====================

@torch.no_grad()
def run_inference(args):
    """Run long-horizon autoregressive rollout inference."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    val_order = parse_run_list(args.run_list_file)
    if not val_order:
        raise RuntimeError(f"Empty run list: {args.run_list_file}")
    print(f"[Data] {len(val_order)} runs")

    # Probe first run for dimensions
    subj0_or_star, rid0 = val_order[0]
    subj0 = args.subjects[0] if subj0_or_star == "*" else subj0_or_star
    V0, A0, W0, Z0, _ = load_stimulus_and_circuit_tokens(args, subj0, rid0)
    Dv, Da, Dw, Dz = V0.shape[1], A0.shape[1], W0.shape[1], Z0.shape[1]
    print(f"[Dims] Dv={Dv}, Da={Da}, Dw={Dw}, D_f={Dz}")

    # Load frozen network-wise decoders
    all_decoders, total_z_dim = load_all_decoders(args, device)
    if total_z_dim != Dz:
        raise ValueError(f"D_f mismatch: circuit tokens={Dz}, decoders sum={total_z_dim}")
    z_splits = [all_decoders[p].z_dim for p in args.parcels]

    # Build predictor
    model, meta = build_model_from_ckpt(
        args.predictor_ckpt, Dv, Da, Dw, Dz, z_splits, device,
    )
    L, Lb = meta["L"], meta["Lb"]
    pred_delta_z = meta["pred_delta_z"]
    minL = max(L, Lb)
    K_horizon = args.K_horizon

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    for subj_or_star, rid in tqdm(val_order, desc="[BrainVista Inference]", ncols=100):
        subj = args.subjects[0] if subj_or_star == "*" else subj_or_star
        try:
            V_seq, A_seq, W_seq, Z_seq, T_final = load_stimulus_and_circuit_tokens(args, subj, rid)
        except Exception as e:
            print(f"[Skip] {subj}/{rid}: {e}")
            continue

        if T_final < (minL + 1):
            continue

        V_t = torch.from_numpy(np.ascontiguousarray(V_seq)).float().to(device).unsqueeze(0)
        A_t = torch.from_numpy(np.ascontiguousarray(A_seq)).float().to(device).unsqueeze(0)
        W_t = torch.from_numpy(np.ascontiguousarray(W_seq)).float().to(device).unsqueeze(0)
        Z_t = torch.from_numpy(np.ascontiguousarray(Z_seq)).float().to(device).unsqueeze(0)

        # Autoregressive rollout: ground-truth stimulus tokens provided,
        # only circuit tokens are rolled out
        t_anchor = minL - 1
        Z_hist = Z_t[:, t_anchor - Lb + 1:t_anchor + 1, :].clone()

        preds_z = []
        spatial_attns = []

        with torch.amp.autocast(device_type="cuda",
                                enabled=(device.type == "cuda" and not args.no_amp)):
            for k in range(K_horizon):
                t_curr = t_anchor + k
                if t_curr + 1 >= T_final:
                    break

                Vw = V_t[:, t_curr - L + 1:t_curr + 1, :]
                Aw = A_t[:, t_curr - L + 1:t_curr + 1, :]
                Ww = W_t[:, t_curr - L + 1:t_curr + 1, :]

                if Vw.shape[1] != L:
                    break

                out_seq, attn_dict = model(Vw, Aw, Ww, Z_hist, return_attn=True)
                out_last = out_seq[:, -1, :]
                z_prev = Z_hist[:, -1, :]
                z_pred_next = (z_prev + out_last) if pred_delta_z else out_last

                preds_z.append(z_pred_next.squeeze(0))
                if attn_dict.get("spatial_attn") is not None:
                    spatial_attns.append(
                        attn_dict["spatial_attn"][:, -1, :, :, :].cpu().numpy()
                    )

                Z_hist = torch.cat([Z_hist[:, 1:, :], z_pred_next.unsqueeze(1)], dim=1)

        if not preds_z:
            continue

        # Decode circuit tokens to voxel space via frozen network-wise decoders
        Z_pred = torch.stack(preds_z, dim=0)
        K_eff = Z_pred.shape[0]
        Z_true = Z_t[0, t_anchor + 1:t_anchor + 1 + K_eff, :]

        start_col = 0
        x_pred_parts, x_true_parts = [], []
        for parcel in args.parcels:
            dec = all_decoders[parcel]
            d_k = dec.z_dim
            x_pred_parts.append(
                dec(Z_pred[:, start_col:start_col + d_k].to(device)).cpu().numpy()
            )
            x_true_parts.append(
                dec(Z_true[:, start_col:start_col + d_k].to(device)).cpu().numpy()
            )
            start_col += d_k

        x_pred = np.concatenate(x_pred_parts, axis=1)
        x_true = np.concatenate(x_true_parts, axis=1)

        # Compute metrics
        r = pearson_r(x_true.ravel(), x_pred.ravel())
        p_corr = pattern_corr(x_true, x_pred)

        # Save predictions
        run_out = os.path.join(args.out_dir, subj)
        os.makedirs(run_out, exist_ok=True)
        np.save(os.path.join(run_out, f"{rid}_pred.npy"), x_pred.astype(np.float32))
        np.save(os.path.join(run_out, f"{rid}_true.npy"), x_true.astype(np.float32))
        if spatial_attns:
            np.save(os.path.join(run_out, f"{rid}_spatial_attn.npy"),
                    np.concatenate(spatial_attns, axis=0).astype(np.float32))

        results.append({"subj": subj, "rid": rid, "K_eff": K_eff,
                        "r": r, "p_corr": p_corr})
        print(f"  {subj}/{rid}: H={K_eff}, r={r:.4f}, p_corr={p_corr:.4f}")

    # Summary
    if results:
        mean_r = float(np.mean([r["r"] for r in results]))
        mean_pcorr = float(np.mean([r["p_corr"] for r in results]))
        print(f"\n[Summary] {len(results)} runs | mean r={mean_r:.6f} | mean p_corr={mean_pcorr:.6f}")
        with open(os.path.join(args.out_dir, "inference_summary.json"), "w") as f:
            json.dump({
                "num_runs": len(results),
                "mean_r": mean_r,
                "mean_p_corr": mean_pcorr,
                "K_horizon": K_horizon,
                "parcels": args.parcels,
                "runs": results,
            }, f, indent=2)
        print(f"[Saved] {os.path.join(args.out_dir, 'inference_summary.json')}")
    else:
        print("[Warning] No valid runs processed.")


def parse_args():
    ap = argparse.ArgumentParser(
        "BrainVista Inference: Long-horizon autoregressive rollout (arXiv: 2602.04512)"
    )
    ap.add_argument("--predictor_ckpt", required=True, help="Predictor checkpoint path")
    ap.add_argument("--video_root", required=True, help="V-JEPA 2 features root")
    ap.add_argument("--audio_root", required=True, help="Wav2Vec2-BERT features root")
    ap.add_argument("--text_root", required=True, help="Llama 3.2 features root")
    ap.add_argument("--z_root", required=True, help="Circuit tokens root")
    ap.add_argument("--run_list_file", required=True, help="Test run list file")
    ap.add_argument("--subjects", nargs="+", default=["sub-01"])
    ap.add_argument("--parcels", nargs="+", required=True,
                    help="Yeo-7 network names: Cont DorsAttn SalVentAttn Limbic SomMot Default Vis")
    ap.add_argument("--tokenizer_ckpt_dir", required=True,
                    help="Directory containing network-wise tokenizer checkpoints")
    ap.add_argument("--tokenizer_name_template", required=True,
                    help="Template for tokenizer ckpt filename, e.g., ckpt_tokenizer_{parcel}/best.pt")
    ap.add_argument("--K_horizon", type=int, default=20, help="Rollout horizon H (default: 20)")
    ap.add_argument("--out_dir", type=str, default="./predictions",
                    help="Output directory for predictions")

    # Feature processing
    ap.add_argument("--ratio_v", type=int, default=3)
    ap.add_argument("--ratio_a", type=int, default=3)
    ap.add_argument("--ratio_w", type=int, default=3)
    ap.add_argument("--collapse_v", type=str, default="flatten")
    ap.add_argument("--collapse_a", type=str, default="flatten")
    ap.add_argument("--collapse_w", type=str, default="flatten")
    ap.add_argument("--layer_pool", type=str, default="mean")
    ap.add_argument("--layer_axis", type=int, default=1)
    ap.add_argument("--select_layer_idx", type=int, default=-1)

    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
