import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_4d_robust_infer(data, med, mad, clip=5.0):
    """
    Per-channel normalization for inference.
    med, mad have shape (1, F) - normalized per feature across all channels
    """
    n, t, c, f = data.shape
    # Reshape to (n*t*c, f) to match training normalization
    flat = data.reshape(n * t * c, f)
    flat_r = (flat - med) / (1.4826 * mad + 1e-8)
    if clip is not None:
        flat_r = np.clip(flat_r, -clip, clip)
    return flat_r.reshape(n, t, c, f)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, emb_dim=32):
        super().__init__()
        self.E = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.02)

    def forward(self):
        logits = torch.relu(self.E @ self.E.t())
        A = torch.softmax(logits, dim=-1)
        I = torch.eye(A.size(0), device=A.device)
        A = A + I
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        return A


class GraphConv(nn.Module):
    def __init__(self, din, dout, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(din, dout, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dout)

    def forward(self, x, A):
        xw = self.lin(x)
        out = torch.einsum("ij,btjd->btid", A, xw)
        out = self.norm(out)
        return self.drop(F.gelu(out))


class NFGraphTemporalTransformer(nn.Module):
    def __init__(self, num_channels, num_features=9, d_model=64, gnn_layers=2,
                 nhead=4, tf_layers=2, dropout=0.1, horizon=10, adj_emb_dim=32):
        super().__init__()
        self.C = num_channels
        self.F = num_features
        self.horizon = horizon
        self.d_model = d_model
        self.temporal_dropout = 0.15  # ADDED

        # ADDED: Feature attention
        self.feature_attn = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Softmax(dim=-1)
        )

        self.in_proj = nn.Linear(num_features, d_model)
        self.adj = AdaptiveAdjacency(num_channels, emb_dim=adj_emb_dim)
        self.gnn = nn.ModuleList([GraphConv(d_model, d_model, dropout=dropout) for _ in range(gnn_layers)])

        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        B, Tin, C, F_ = x.shape
        assert C == self.C and F_ == self.F, f"Expected (B,T,{self.C},{self.F}) got {x.shape}"

        # ADDED: Feature attention
        feat_weights = self.feature_attn(x.mean(dim=(0, 1, 2)))
        x_weighted = x * feat_weights.view(1, 1, 1, -1)

        # CHANGED: use x_weighted
        h = self.in_proj(x_weighted)
        A = self.adj()
        for layer in self.gnn:
            h = h + layer(h, A)

        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * C, Tin, self.d_model)
        h = self.pos(h)
        
        # ADDED: Temporal dropout
        if self.training:
            mask = (torch.rand(h.size(0), h.size(1), 1, device=h.device) > self.temporal_dropout).float()
            h = h * mask
        
        z = self.temporal(h)

        last = z[:, -3:, :].mean(dim=1)  # Already updated
        y = self.head(last)
        y = y.reshape(B, C, self.horizon).permute(0, 2, 1).contiguous()
        return y


class Model(nn.Module):
    def __init__(self, monkey_name="beignet"):
        super().__init__()
        self.monkey_name = monkey_name
        self.init_steps = 10
        self.horizon = 10
        self.num_features = 9
        self.clip = None
    
        base = os.path.dirname(__file__)
        stats_path = os.path.join(base, f"train_data_average_std_{monkey_name}.npz")
        stats = np.load(stats_path)
        average = stats["average"]  # med
        std = stats["std"]          # mad
    
        # UPDATED: Handle new normalization format (1, F) instead of (1, C*F) or (1, C, F)
        if len(average.shape) == 2 and average.shape[1] == self.num_features:
            # New format from per-channel normalization: (1, F)
            self.average = average
            self.std = std
            # Infer channels from a different method - need to know expected channels
            # This is a problem - we need to know num_channels somehow
            # Assuming you saved it or can infer from model weights file name
            # For now, let's try to infer from the model file if it exists
            self.num_channels = None  # Will be set after loading model
        elif len(average.shape) == 3:
            # Old format: (1, C, F)
            _, self.num_channels, feat = average.shape
            if feat != self.num_features:
                raise ValueError(f"Feature dim mismatch: expected {self.num_features}, got {feat}")
            # Convert to (1, F) format by taking mean across channels
            self.average = average.mean(axis=1, keepdims=True)  # (1, F)
            self.std = std.mean(axis=1, keepdims=True)  # (1, F)
        else:
            # Very old format: (1, C*F)
            cf = int(average.shape[1])
            if cf % self.num_features != 0:
                raise ValueError(f"Stats shape {average.shape} not divisible by num_features={self.num_features}")
            self.num_channels = cf // self.num_features
            reshaped = average.reshape(1, self.num_channels, self.num_features)
            self.average = reshaped.mean(axis=1, keepdims=True)  # (1, F)
            self.std = std.reshape(1, self.num_channels, self.num_features).mean(axis=1, keepdims=True)
        
        # If num_channels not set, we need to determine it from model weights
        # This is a temporary solution - ideally save num_channels in stats
        if self.num_channels is None:
            # Try to load model to get num_channels
            wpath = os.path.join(base, f"model_{self.monkey_name}.pth")
            try:
                state_dict = torch.load(wpath, map_location='cpu', weights_only=True)
            except TypeError:
                state_dict = torch.load(wpath, map_location='cpu')
            # Get num_channels from adj.E shape
            self.num_channels = state_dict['adj.E'].shape[0]
    
        self.net = NFGraphTemporalTransformer(
            num_channels=self.num_channels,
            num_features=self.num_features,
            d_model=64,
            gnn_layers=2,
            nhead=4,
            tf_layers=2,
            dropout=0.1,
            horizon=self.horizon,
        )

    def load(self):
        base = os.path.dirname(__file__)
        wpath = os.path.join(base, f"model_{self.monkey_name}.pth")

        try:
            state_dict = torch.load(wpath, map_location=torch.device(device), weights_only=True)
        except TypeError:
            state_dict = torch.load(wpath, map_location=torch.device(device))

        self.net.load_state_dict(state_dict, strict=True)
        self.to(device)
        self.eval()

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        x = np.asarray(x)
        N, T, C_in, F_in = x.shape
        if T != 20 or F_in != self.num_features:
            raise ValueError(f"Expected (N,20,C,9), got {x.shape}")
    
        C_exp = self.num_channels
        if C_in > C_exp:
            x = x[:, :, :C_exp, :]
        elif C_in < C_exp:
            pad = np.zeros((N, T, C_exp - C_in, F_in), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=2)
    
        # UPDATED: Use new normalization (1, F) format
        med = self.average  # (1, F)
        mad = self.std      # (1, F)
        x_norm = normalize_4d_robust_infer(x, med, mad, clip=None)
    
        xin = torch.tensor(x_norm[:, :self.init_steps, :, :], dtype=torch.float32, device=device)
    
        with torch.no_grad():
            pred_future_norm = self.net(xin).detach().cpu().numpy()
    
        # UPDATED: Denormalize using (1, F) format
        med_f0 = med[0, 0]  # scalar
        scale_f0 = 1.4826 * mad[0, 0]  # scalar
    
        pred_future_raw = pred_future_norm * scale_f0 + med_f0
    
        out = np.zeros((N, 20, C_exp), dtype=np.float32)
        out[:, :self.init_steps, :] = x[:, :self.init_steps, :, 0].astype(np.float32)
        out[:, self.init_steps:, :] = pred_future_raw.astype(np.float32)
        return out