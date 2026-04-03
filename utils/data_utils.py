# utils/data_utils.py
"""Data loading and temporal alignment utilities for BrainVista inference (arXiv: 2602.04512).

Handles stimulus-to-fMRI temporal alignment via mean-pooling within each fMRI time bin:
  tau_m^t = (1/kappa) * sum_{i=0}^{kappa-1} tau_m^{kappa*t + i}
where kappa is the stimulus-to-fMRI downsampling factor.
"""
import os
import re
import glob
import argparse
import numpy as np
from typing import List, Tuple

import torch

from .models import Dec


def parse_run_list(path: str) -> List[Tuple[str, str]]:
    """Parse run list file. Supports 'rid' or 'sub-XX rid' formats."""
    out = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t for t in s.replace(",", " ").split() if t]
            if len(toks) == 1:
                out.append(("*", toks[0]))
            else:
                out.append((toks[0], toks[1]))
    return out


def reduce_layers(X, mode="mean", axis=1, idx=-1):
    """Reduce multi-layer features to single layer."""
    if X.ndim == 2:
        return X
    if mode == "mean":
        return X.mean(axis=axis)
    if mode == "last":
        return np.take(X, -1, axis=axis)
    if mode == "select":
        return np.take(X, idx, axis=axis)
    if mode == "concat":
        T = X.shape[0]
        return X.reshape(T, int(np.prod(X.shape[1:])))
    raise ValueError(f"Unknown layer reduction mode: {mode}")


def npy(fp: str) -> np.ndarray:
    return np.load(fp, allow_pickle=False).astype(np.float32)


def safeT(*arrs: np.ndarray) -> Tuple[List[np.ndarray], int]:
    """Crop all arrays to minimum time length for alignment."""
    T = min(x.shape[0] for x in arrs)
    return [x[:T] for x in arrs], T


def _collapse_time_features(arr: np.ndarray, mode: str = "flatten") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim >= 3:
        T = arr.shape[0]
        if mode == "flatten":
            arr = arr.reshape(T, -1)
        else:
            raise ValueError(f"Unsupported collapse mode: {mode}")
    return np.ascontiguousarray(arr)


def pool_modal_to_fmri_grid_np(M: np.ndarray, T_fmri: int, ratio: int,
                                collapse: str = "flatten") -> Tuple[np.ndarray, int]:
    """Temporal alignment: mean-pool stimulus features to fMRI temporal grid.

    tau_m^t = (1/kappa) * sum_{i=0}^{kappa-1} tau_m^{kappa*t + i}
    """
    M2d = _collapse_time_features(M, mode=collapse)
    Tm, D = M2d.shape
    ratio = int(max(1, ratio))
    T_need = T_fmri * ratio
    if Tm < T_need:
        T_eff = Tm // ratio
        T_need = T_eff * ratio
        if T_need == 0:
            raise ValueError(f"Modal too short: Tm={Tm}, ratio={ratio}")
        T_fmri = T_eff
    M_cut = M2d[:T_need].reshape(T_fmri, ratio, D)
    M_pool = M_cut.mean(axis=1)
    return np.ascontiguousarray(M_pool), int(T_fmri)


def load_circuit_tokens(z_root, subj, rid, parcels):
    """Load and concatenate circuit tokens from K functional networks."""
    z_parcels = []
    for parcel_name in parcels:
        cand_paths = [
            os.path.join(z_root, subj, parcel_name, rid + ".npy"),
            os.path.join(z_root, subj, parcel_name, rid + "_z.npy"),
            os.path.join(z_root, f"z_{parcel_name}", subj, rid + ".npy"),
            os.path.join(z_root, f"z_{parcel_name}", subj, rid + "_z.npy"),
        ]
        z_fp = None
        for p in cand_paths:
            if os.path.exists(p):
                z_fp = p
                break
        if z_fp is None:
            hits = glob.glob(os.path.join(z_root, subj, "**", rid + "_z.npy"), recursive=True)
            if not hits:
                hits = glob.glob(os.path.join(z_root, subj, "**", rid + ".npy"), recursive=True)
            if hits:
                z_fp = hits[0]
        if z_fp is None:
            raise FileNotFoundError(
                f"Missing circuit tokens: network={parcel_name}, subj={subj}, rid={rid}"
            )
        z_parcels.append(npy(z_fp))
    return np.concatenate(z_parcels, axis=1)


def load_stimulus_and_circuit_tokens(args, subj, rid):
    """Load aligned stimulus features and circuit tokens for a single run.

    Returns:
        V_seq, A_seq, W_seq, Z_seq: aligned numpy arrays [T, D_*]
        T_final: number of aligned timesteps
    """
    Z_full = load_circuit_tokens(args.z_root, subj, rid, args.parcels)
    T_z = Z_full.shape[0]

    v_fp_hits = glob.glob(os.path.join(args.video_root, subj, "**", rid + ".npy"), recursive=True)
    w_fp_hits = glob.glob(os.path.join(args.text_root, subj, "**", rid + ".npy"), recursive=True)
    audio_enabled = getattr(args, "ratio_a", 0) > 0 and getattr(args, "audio_root", None) is not None
    a_fp_hits = (glob.glob(os.path.join(args.audio_root, subj, "**", rid + ".npy"), recursive=True)
                 if audio_enabled else [None])

    if not v_fp_hits or not w_fp_hits or (audio_enabled and not a_fp_hits):
        raise FileNotFoundError(f"Missing stimulus features: subj={subj}, rid={rid}")

    V_raw = reduce_layers(npy(v_fp_hits[0]), args.layer_pool, args.layer_axis, args.select_layer_idx)
    W_raw = reduce_layers(npy(w_fp_hits[0]), args.layer_pool, args.layer_axis, args.select_layer_idx)
    A_raw = (reduce_layers(npy(a_fp_hits[0]), args.layer_pool, args.layer_axis, args.select_layer_idx)
             if (audio_enabled and a_fp_hits[0])
             else np.zeros((T_z, 0), dtype=np.float32))

    # Temporal alignment via mean-pooling
    V_p, _ = pool_modal_to_fmri_grid_np(V_raw, T_z, args.ratio_v, collapse=args.collapse_v)
    A_p = (pool_modal_to_fmri_grid_np(A_raw, T_z, args.ratio_a, collapse=args.collapse_a)[0]
           if (audio_enabled and A_raw.shape[1] > 0)
           else np.zeros((T_z, 0), dtype=np.float32))

    if getattr(args, "ratio_w", 1) > 1:
        W_up = np.repeat(W_raw, args.ratio_w, axis=0)
        if W_up.shape[0] > T_z:
            W_up = W_up[:T_z]
        elif W_up.shape[0] < T_z:
            diff = T_z - W_up.shape[0]
            W_up = np.concatenate([W_up, np.tile(W_up[-1:], (diff, 1))], axis=0)
        W_p = W_up
    else:
        W_p, _ = pool_modal_to_fmri_grid_np(W_raw, T_z, args.ratio_w, collapse=args.collapse_w)

    (V_seq, A_seq, W_seq, Z_seq), T_final = safeT(V_p, A_p, W_p, Z_full)
    return V_seq, A_seq, W_seq, Z_seq, T_final


def load_all_decoders(args, device) -> Tuple[dict, int]:
    """Load frozen network-wise tokenizer decoders for all K functional networks."""
    print(f"--- Loading {len(args.parcels)} Network-wise Decoders ---")
    all_decoders = {}
    total_z_dim = 0

    for parcel in args.parcels:
        try:
            try:
                template = args.tokenizer_name_template.format(parcel=parcel)
            except (IndexError, KeyError):
                template = args.tokenizer_name_template.format(parcel)

            ckpt_path = os.path.join(args.tokenizer_ckpt_dir, template)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Not found: {ckpt_path}")

            ck_tok = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg_tok = ck_tok.get("args", ck_tok.get("cfg", {}))
            if isinstance(cfg_tok, argparse.Namespace):
                cfg_tok = vars(cfg_tok)

            D = ck_tok["Db"]
            z_dim = int(cfg_tok.get("z_dim", 128))

            dec = Dec(z_dim, D, int(cfg_tok.get("hidden", 768)),
                      float(cfg_tok.get("dropout", 0.1))).to(device)

            dec_sd = {}
            for k, v in ck_tok["dec"].items():
                for prefix in ("_orig_mod.", "module."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                dec_sd[k] = v
            dec.load_state_dict(dec_sd, strict=False)
            dec.eval()

            all_decoders[parcel] = dec
            total_z_dim += dec.z_dim
            print(f"  > {parcel}: d_k={dec.z_dim}, p_k={D}")

        except Exception as e:
            raise RuntimeError(f"Failed on network {parcel}: {e}")

    print(f"--- Total D_f: {total_z_dim} ---")
    return all_decoders, total_z_dim
