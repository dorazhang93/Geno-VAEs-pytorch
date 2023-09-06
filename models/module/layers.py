from models.types_ import *
import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn, einsum
from models.module.helpers import *
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 num_layer: int):
        super(ResidualBlock, self).__init__()
        pad_size = kernel_size // 2
        modules = []
        for i in range(num_layer):
            modules.append(nn.Sequential(
                nn.Conv1d(in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=pad_size),
                nn.ELU(),
                nn.BatchNorm1d(out_channel),
            ))
            in_channel = out_channel

        self.layer = nn.Sequential(*modules)

    def forward(self, input):
        x = self.layer(input)
        x += input
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 activation: bool = True,
                 normalize: bool = True):
        super(ConvBlock, self).__init__()

        pad_size = kernel_size // 2
        self.conv = nn.Conv1d(in_channel, out_channels=out_channel,kernel_size=kernel_size,padding=pad_size)
        self.activate = nn.ELU() if activation else None
        self.norm = nn.BatchNorm1d(out_channel) if normalize else None

    def forward(self, input):
        x= self.conv(input)
        if self.activate is not None:
            x= self.activate(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class NormConvBlock(nn.Module):
    def __init__(self,dim, dim_out = None, kernel_size = 1, stride=1,dilate=1,padding=None):
        super(NormConvBlock, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.batchNormConv = nn.Sequential(
            nn.BatchNorm1d(dim),
            GELU(),
            nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=padding,stride=stride,
                      dilation=dilate))

    def forward(self, input):
        return self.batchNormConv(input)

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super(AttentionPool,self).__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):

        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        attn = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(attn.dtype).max
            attn = attn.masked_fill(self.pool_fn(mask), mask_value)
            del mask
        attn = attn.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual,self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super(MultiheadAttention,self).__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = int((num_rel_pos_features // 6) * 6)

        self.to_rel_k = nn.Linear(self.num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        positions = self.to_rel_k(positions)

        positions = rearrange(positions, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, positions)
        rel_logits = relative_shift(rel_logits)
        attn = content_logits + rel_logits
        attn = attn.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-1], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (seq_len - target_len) // 2

        if trim == 0:
            return x

        return x[:,:, trim:(trim+target_len)]



