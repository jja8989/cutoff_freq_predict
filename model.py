import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ------------------------------------------------------
# 🔹 Residual Block with LayerNorm (Deterministic)
# ------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=2, dropout=0.1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.match_dim = (in_channels != out_channels)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.ln2 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(dropout)

        if self.match_dim:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = F.relu(self.ln1(out.transpose(1, 2)).transpose(1, 2))
        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out + identity)
        return out


# ------------------------------------------------------
# 🔹 Masked Global Average Pooling
# ------------------------------------------------------
def masked_global_avg_pool(x, lengths):
    # x: (B, C, L)
    B, C, L = x.shape
    mask = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1)  # (B, 1, L)
    masked_sum = (x * mask).sum(dim=2)
    valid_counts = mask.sum(dim=2).clamp(min=1)
    return masked_sum / valid_counts


# ------------------------------------------------------
# 🔹 CNN + GRU Hybrid Model
# ------------------------------------------------------
class CutoffPredictorCNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()

        # --- CNN backbone ---
        self.features = nn.Sequential(
            ResBlock1D(1, 16, kernel_size=5, dilation=1, dropout=0.05),
            nn.MaxPool1d(2),
            ResBlock1D(16, 32, kernel_size=5, dilation=2, dropout=0.05),
            nn.MaxPool1d(2),
            ResBlock1D(32, 64, kernel_size=5, dilation=4, dropout=0.1),
        )

        # --- GRU ---
        self.gru = nn.GRU(
            input_size = 64,
            hidden_size = 64,
            num_layers = 1,
            batch_first=True,
            bidirectional=True
        )

        # --- Pooling ---
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # --- Regression Head (with Fs) ---
        self.fc1 = nn.Linear(257, 128)  # ✅ 256(feature) + 1(Fs)
        self.ln = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, Fs, lengths=None):
        # x: (B, 1, L)
        x = self.features(x)  # (B, 64, L')
        L_out = x.shape[-1]

        # CNN에서 2회 pooling했으므로 길이 보정
        if lengths is not None:
            lengths_out = torch.ceil(lengths.float() / 4).long().clamp(max=L_out)
        else:
            lengths_out = torch.tensor([L_out] * x.size(0), device=x.device)

        # --- GRU ---
        x_t = x.transpose(1, 2)  # (B, L', 64)
        packed = pack_padded_sequence(x_t, lengths_out.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out_seq, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=L_out)
        out_seq = out_seq.transpose(1, 2)  # (B, 128, L')

        # --- Pooling ---
        avg = masked_global_avg_pool(out_seq, lengths_out)
        mx = self.max_pool(out_seq).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)  # (B, 256)

        # --- Fs feature 추가 ---
        x = torch.cat([x, Fs.unsqueeze(1)], dim=1)  # (B, 257)

        # --- Residual MLP ---
        x = self.fc1(x)
        x = F.relu(self.ln(x))
        h2 = F.relu(self.fc2(x)) + x
        out = self.fc_out(h2)
        return out.squeeze(-1)


