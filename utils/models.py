# utils/models.py
"""
BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction
(arXiv: 2602.04512)

Predictor model definitions:
  - Interleaved stimulus tokens (tau_s) and circuit tokens (tau_f)
  - Pre-LN causal Transformer (d_model=256, nhead=8, nlayers=4, GELU, 4x FFN)
  - S2B masking: circuit tokens cannot self-attend
  - Spatial Mixer Head: K x K self-attention over K=7 Yeo functional networks

Network-wise tokenizer decoder:
  - Two-layer MLP: circuit token tau_f -> fMRI voxels
"""
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  Network-wise tokenizer decoder                                     #
# ------------------------------------------------------------------ #

class Dec(nn.Module):
    """Network-wise tokenizer decoder: circuit token tau_f -> fMRI voxels."""
    def __init__(self, z_dim=128, D_out=512, hidden=768, dropout=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, D_out),
        )

    def forward(self, z):
        return self.net(z)


# ------------------------------------------------------------------ #
#  Spatial Mixer Head (K functional networks)                         #
# ------------------------------------------------------------------ #

class _SpatialSelfAttnBlock(nn.Module):
    """Single self-attention block for the Spatial Mixer Head."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        y, attn = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + y)
        x = self.norm2(x + self.ffn(x))
        return x, attn


class _SpatialMixerHead(nn.Module):
    """Spatial Mixer Head: per-timestep K x K self-attention over functional networks.

    Splits predicted circuit token into K network components, projects each to
    shared dimension d_z, applies multi-head self-attention to model cross-network
    interactions, then projects back. The K x K attention matrix provides
    interpretable network coupling patterns.

    Reference: Section 3.4 of BrainVista (arXiv: 2602.04512)
    """
    def __init__(self, D_f: int, z_splits: list, d_z: int,
                 nhead: int = 4, nlayers: int = 1, dropout: float = 0.1):
        super().__init__()
        assert sum(z_splits) == D_f
        self.D_f = D_f
        self.z_splits = z_splits
        self.K = len(z_splits)
        self.d_z = d_z
        self.nhead = nhead

        self.in_projs = nn.ModuleList([nn.Linear(d, d_z) for d in z_splits])
        self.layers = nn.ModuleList([
            _SpatialSelfAttnBlock(d_z, nhead, dropout) for _ in range(nlayers)
        ])
        self.out_projs = nn.ModuleList([nn.Linear(d_z, d) for d in z_splits])
        self.last_attn = None

    def forward(self, z_seq: torch.Tensor, return_attn: bool = False):
        B, T, D_f = z_seq.shape
        assert D_f == self.D_f

        z_flat = z_seq.reshape(B * T, D_f)
        chunks = torch.split(z_flat, self.z_splits, dim=-1)

        tokens = [self.in_projs[k](chunks[k]) for k in range(self.K)]
        x = torch.stack(tokens, dim=1)  # [B*T, K, d_z]

        attn_last = None
        for layer in self.layers:
            x, attn_last = layer(x)

        outs = [self.out_projs[k](x[:, k, :]) for k in range(self.K)]
        z_ref = torch.cat(outs, dim=-1).view(B, T, D_f)

        if attn_last is not None:
            self.last_attn = attn_last.view(B, T, self.nhead, self.K, self.K)

        if return_attn:
            return z_ref, self.last_attn
        return z_ref


# ------------------------------------------------------------------ #
#  Base: Interleaved Stimulus-Circuit Causal Transformer              #
# ------------------------------------------------------------------ #

class InterleavedSelfAttnPredictor(nn.Module):
    """Causal Transformer over interleaved stimulus and circuit token sequences.

    At time T with context length L, the input sequence is:
        S^T = (tau_s^{T-L+1}, tau_f^{T-L+1}, ..., tau_s^T, tau_f^T)
    Total sequence length: 2L positions.

    Architecture: Pre-LN Transformer, GELU, 4x FFN expansion, absolute position embeddings.

    Reference: Section 3.2 of BrainVista (arXiv: 2602.04512)
    """

    def __init__(self, Dv: int, Da: int, Dw: int,
                 Dz_in: int, Dz_out: int,
                 d_model: int = 256, nhead: int = 8, nlayers: int = 4,
                 dropout: float = 0.1, use_hrf: bool = True, brain_ar: bool = True,
                 L: int = 40, Lb: int = 40) -> None:
        super().__init__()
        self.Dv, self.Da, self.Dw = Dv, Da, Dw
        self.Dz_in, self.Dz_out = Dz_in, Dz_out
        self.d_model = d_model
        self.L, self.Lb = L, Lb

        if Lb < L:
            raise ValueError(f"Requires Lb >= L, got Lb={Lb}, L={L}")

        # Temporal smoothing on circuit tokens
        self.brain_ar = None
        if brain_ar:
            self.brain_ar = nn.Conv1d(Dz_in, Dz_in, kernel_size=3, padding=1,
                                      groups=Dz_in, bias=False)

        # Stimulus modality projections (V-JEPA 2, Wav2Vec2-BERT, Llama 3.2)
        self.v_proj = nn.Linear(Dv, d_model) if Dv > 0 else None
        self.a_proj = nn.Linear(Da, d_model) if Da > 0 else None
        self.w_proj = nn.Linear(Dw, d_model) if Dw > 0 else None

        # Temporal convolution for hemodynamic response modeling
        self.hrf_stim = None
        if use_hrf:
            self.hrf_stim = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2,
                                      groups=d_model, bias=False)

        # Circuit token projections
        self.z_proj = nn.Linear(Dz_in, d_model)
        self.out_proj = nn.Linear(d_model, Dz_out)

        # Type embeddings: 0 = stimulus token, 1 = circuit token
        self.type_embed = nn.Embedding(2, d_model)

        old_len = Lb - L
        S = old_len + 2 * L
        self.S = S
        self.old_len = old_len

        type_ids = [1] * old_len
        for _ in range(L):
            type_ids.extend([0, 1])
        assert len(type_ids) == S
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long), persistent=False)

        # Absolute position embeddings
        self.pos_embed = nn.Embedding(S, d_model)
        self.register_buffer("pos_ids", torch.arange(S, dtype=torch.long), persistent=False)

        # Standard causal (lower-triangular) attention mask
        causal_mask = torch.full((S, S), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Index mapping: circuit token positions in the interleaved sequence
        brain_seq_idx = []
        for i in range(Lb):
            if i < old_len:
                brain_seq_idx.append(i)
            else:
                j = i - old_len
                brain_seq_idx.append(old_len + 2 * j + 1)
        self.register_buffer("brain_seq_idx",
                             torch.tensor(brain_seq_idx, dtype=torch.long), persistent=False)

        # Pre-LN Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=False, activation="gelu", norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def _prep_stim_tokens(self, V, A, W):
        """Fuse multimodal stimulus features into shared stimulus tokens tau_s."""
        B, L, _ = V.shape
        stim = torch.zeros(B, L, self.d_model, device=V.device)
        n = 0
        if self.v_proj is not None:
            stim = stim + self.v_proj(V); n += 1
        if self.a_proj is not None and A is not None:
            stim = stim + self.a_proj(A); n += 1
        if self.w_proj is not None and W is not None:
            stim = stim + self.w_proj(W); n += 1
        if n > 0:
            stim = stim / float(n)
        if self.hrf_stim is not None:
            stim = self.hrf_stim(stim.transpose(1, 2)).transpose(1, 2)
        return stim

    def forward(self, V, A, W, Z_hist):
        """Forward pass: predict next circuit tokens from interleaved sequence.

        Args:
            V: [B, L, Dv] video features (V-JEPA 2)
            A: [B, L, Da] audio features (Wav2Vec2-BERT)
            W: [B, L, Dw] text features (Llama 3.2)
            Z_hist: [B, Lb, Dz] circuit token history
        Returns:
            out: [B, Lb, Dz_out] predicted delta or absolute circuit tokens
        """
        B, L_v, _ = V.shape
        _, Lb, _ = Z_hist.shape

        if L_v != self.L or Lb != self.Lb:
            raise ValueError(
                f"Expected V.shape[1]=={self.L}, Z_hist.shape[1]=={self.Lb}, "
                f"got L_v={L_v}, Lb={Lb}"
            )

        if self.brain_ar is not None:
            Z_sm = self.brain_ar(Z_hist.transpose(1, 2)).transpose(1, 2)
        else:
            Z_sm = Z_hist

        old_len, L = self.old_len, self.L
        Z_old = Z_sm[:, :old_len, :]
        Z_recent = Z_sm[:, old_len:, :]

        circuit_old = self.z_proj(Z_old) if old_len > 0 else None
        circuit_recent = self.z_proj(Z_recent)
        stim_recent = self._prep_stim_tokens(V, A, W)

        # Build interleaved sequence: [circuit_prefix, (stim, circuit) x L]
        tokens = []
        if old_len > 0:
            tokens.append(circuit_old)
        inter = torch.stack([stim_recent, circuit_recent], dim=2).view(B, 2 * L, self.d_model)
        tokens.append(inter)

        x = torch.cat(tokens, dim=1)
        S = x.size(1)
        assert S == self.S

        type_ids = self.type_ids.unsqueeze(0).expand(B, -1)
        pos_ids = self.pos_ids.unsqueeze(0).expand(B, -1)
        x = x + self.type_embed(type_ids) + self.pos_embed(pos_ids)

        # Causal Transformer forward
        x = x.transpose(0, 1)
        x = self.backbone(x, self.causal_mask[:S, :S])
        x = x.transpose(0, 1)

        # Extract circuit token hidden states
        h_circuit = x.index_select(dim=1, index=self.brain_seq_idx)
        return self.out_proj(h_circuit)


# ------------------------------------------------------------------ #
#  S2B Masking: Stimulus-to-Brain causal constraint                   #
# ------------------------------------------------------------------ #

class InterleavedStim2BrainPredictor(InterleavedSelfAttnPredictor):
    """Adds Stimulus-to-Brain (S2B) masking to the causal Transformer.

    Circuit tokens cannot attend to other circuit tokens at the same or
    future timesteps. This forces prediction solely from preceding stimulus
    context, preventing temporal leakage.

    Reference: Section 3.3 of BrainVista (arXiv: 2602.04512)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.causal_mask.clone()
        circuit_pos = (self.type_ids == 1).nonzero(as_tuple=False).view(-1)
        m[circuit_pos, circuit_pos] = float("-inf")
        self._buffers["causal_mask"] = m


# ------------------------------------------------------------------ #
#  Final model: S2B + Spatial Mixer Head                              #
# ------------------------------------------------------------------ #

class InterleavedStim2BrainSpatialPredictor(InterleavedStim2BrainPredictor):
    """BrainVista predictor: S2B causal Transformer + Spatial Mixer Head.

    After the S2B Transformer produces tau_hat_f [B, Lb, D_f], the Spatial
    Mixer Head splits D_f into K network components (K=7 Yeo functional
    networks), applies K x K multi-head self-attention, then maps back to D_f.

    Reference: Section 3.4 of BrainVista (arXiv: 2602.04512)
    """

    def __init__(self, *args,
                 z_splits: list = None,
                 spatial_nhead: int = 4,
                 spatial_nlayers: int = 1,
                 spatial_dropout: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.z_splits = z_splits
        self.spatial_mixer = None
        if z_splits is not None and len(z_splits) > 1 and spatial_nhead > 0:
            self.spatial_mixer = _SpatialMixerHead(
                D_f=self.Dz_out, z_splits=z_splits, d_z=self.d_model,
                nhead=spatial_nhead, nlayers=spatial_nlayers, dropout=spatial_dropout,
            )

    def forward(self, V, A, W, Z_hist, return_attn: bool = False):
        out_seq = super().forward(V, A, W, Z_hist)
        spatial_attn = None
        if self.spatial_mixer is not None:
            out_seq, spatial_attn = self.spatial_mixer(out_seq, return_attn=True)
        if return_attn:
            return out_seq, {"spatial_attn": spatial_attn}
        return out_seq
