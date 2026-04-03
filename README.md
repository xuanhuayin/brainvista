# BrainVista Inference

**BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction**
(arXiv: 2602.04512)

Long-horizon autoregressive rollout inference for fMRI encoding.

## Overview

Given pretrained predictor and network-wise tokenizer decoders, this code predicts future fMRI voxel trajectories via autoregressive circuit token rollout. Ground-truth stimulus tokens are provided; only circuit tokens are rolled out.

## Requirements

```bash
pip install -r requirements.txt
```

- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `tqdm >= 4.60.0`

## Usage

1. Modify paths in `run_inference.sh` to point to your data and checkpoints.
2. Run:

```bash
bash run_inference.sh
```

## Outputs

Per run:
- `{rid}_pred.npy`: predicted voxel trajectories `[H, P_total]`
- `{rid}_true.npy`: ground truth voxel trajectories `[H, P_total]`
- `{rid}_spatial_attn.npy`: K x K network coupling attention `[H, nhead, K, K]`
- `inference_summary.json`: per-run Pearson r, pattern correlation, and global summary

## Architecture

- **Predictor**: Pre-LN causal Transformer (d_model=256, nhead=8, nlayers=4) with S2B masking and Spatial Mixer Head
- **Tokenizer decoders**: Frozen network-wise MLP decoders (one per Yeo-7 functional network)
- **Stimulus features**: V-JEPA 2 (video), Wav2Vec2-BERT (audio), Llama 3.2 (text)
