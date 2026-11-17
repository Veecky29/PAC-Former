import math
import types
import torch
from torch import nn

class RevIN(nn.Module):

    def __init__(self, num_features: int, eps=1e-05, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x

class PhaseLagAlignmentModule(nn.Module):

    def __init__(self, c_in, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw_conv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=c_in, bias=True)
        self.act = nn.GELU()

    def forward(self, x_norm_c):
        return x_norm_c + self.act(self.dw_conv(x_norm_c))

class ScaleConditionedAttention(nn.Module):

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = nn.Parameter(torch.tensor(0.0))

    def forward(self, h, bias_scale):
        b, c, d = h.shape
        q = self.q_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
        bias = 3 * torch.tanh(bias_scale / 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + torch.tanh(self.scale_factor) * bias.unsqueeze(1)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        z = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, c, d)
        return self.out_proj(z)

class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model, n_heads=8, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = ScaleConditionedAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, bias_scale):
        x = self.norm1(x + self.attn(x, bias_scale))
        x = self.norm2(x + self.ffn(x))
        return x

class PACFormer(nn.Module):

    def __init__(self, c_in=7, seq_len=96, pred_len=96, d_model=128, n_heads=8, e_layers=2, d_ff=256, dropout=0.1, kernel_size=7, d_scale=16):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN(num_features=c_in, affine=False)
        self.plm = PhaseLagAlignmentModule(c_in=c_in, kernel_size=kernel_size)
        self.scale_mlp = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d_scale))
        self.d_scale = d_scale
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x_raw):
        x_norm = self.revin(x_raw, mode='norm')
        mean = self.revin.mean
        stdev = self.revin.stdev
        x_norm_c = x_norm.permute(0, 2, 1)
        mean_c = mean.permute(0, 2, 1)
        stdev_c = stdev.permute(0, 2, 1)
        scale_stats = torch.cat([mean_c, torch.log(stdev_c + 1e-05)], dim=-1)
        e_scale = self.scale_mlp(torch.tanh(scale_stats))
        bias_scale = torch.bmm(e_scale, e_scale.transpose(1, 2)) / math.sqrt(self.d_scale)
        x_aligned = self.plm(x_norm_c)
        h = self.enc_embedding(x_aligned)
        for layer in self.layers:
            h = layer(h, bias_scale)
        y_norm_c = self.projector(h)
        y_norm = y_norm_c.permute(0, 2, 1)
        y_out = self.revin(y_norm, mode='denorm')
        return y_out

def plain_attention(self, h, bias_scale=None):
    b, c, d = h.shape
    q = self.q_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
    weights = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1))
    return self.out_proj(torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, c, d))

def no_sca_forward(self, x_raw):
    x = self.revin(x_raw, mode='norm').permute(0, 2, 1)
    x = self.plm(x)
    h = self.enc_embedding(x)
    for layer in self.layers:
        h = layer(h, None)
    return self.revin(self.projector(h).permute(0, 2, 1), mode='denorm')

def create_model(channels, horizon, config, variant='F', device='cuda'):
    if variant not in ['B', 'P', 'S', 'F']:
        raise ValueError(variant)
    model = PACFormer(c_in=channels, seq_len=96, pred_len=horizon, d_model=config['d_model'], n_heads=8, e_layers=2, d_ff=2 * config['d_model'], dropout=0.1, kernel_size=config['kernel_size'], d_scale=16).to(device)
    if variant in ['B', 'S']:
        model.plm = nn.Identity()
    if variant in ['B', 'P']:
        del model.scale_mlp
        for layer in model.layers:
            del layer.attn.scale_factor
            layer.attn.forward = types.MethodType(plain_attention, layer.attn)
        model.forward = types.MethodType(no_sca_forward, model)
    return model
