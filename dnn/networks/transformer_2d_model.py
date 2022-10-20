import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, hparams):
        super().__init__()
        self.attn = Attn(hparams)
        self.hparams = hparams
        self.dropout = nn.Dropout(p=hparams.decoder_dropout)
        self.layernorm_attn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.hparams.filter_size, bias=True),
                                 nn.GELU(),
                                 nn.Linear(self.hparams.filter_size, self.hparams.hidden_size, bias=True))

    def forward(self, inputs):
        x = inputs
        y = self.attn(x)
        x = self.layernorm_attn(self.dropout(y) + x)
        y = self.ffn(x)
        x = self.layernorm_ffn(self.dropout(y) + x)
        return x

class Attn(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.kd = self.hparams.total_key_depth or self.hparams.hidden_size
        self.vd = self.hparams.total_value_depth or self.hparams.hidden_size
        self.q_dense = nn.Linear(self.hparams.hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(self.hparams.hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(self.hparams.hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, self.hparams.hidden_size, bias=False)
        assert self.kd % self.hparams.attn_num_heads == 0
        assert self.vd % self.hparams.attn_num_heads == 0

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def forward(self, X):
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        # Split to shape [batch_size, attn_num_heads, len, depth / attn_num_heads]
        q = q.view(q.shape[:-1] + (self.hparams.attn_num_heads, self.kd // self.hparams.attn_num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.hparams.attn_num_heads, self.kd // self.hparams.attn_num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.hparams.attn_num_heads, self.vd // self.hparams.attn_num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.hparams.attn_num_heads) ** (-0.5)

        if self.hparams.attn_type == "global":
            bias = -1e9 * torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)
            result = self.dot_product_attention(q, k, v, bias=bias)
        elif self.hparams.attn_type == "local_1d":
            len = X.shape[1]
            blen = self.hparams.block_length
            pad = (0, 0, 0, (-len) % self.hparams.block_length) # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3) # [batch, nheads, (nblocks - 1), blen * 2, depth]
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3)
                tail_q = q[:,:,1:]
                bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4])
                result = torch.cat([first_output, tail_output], 2)
                result = result[:,:,:X.shape[1],:]
            else:
                result = first_output[:,:,:X.shape[1],:]
        else:
            raise ValueError(f'unsupported attn_type "{self.hparams.attn_type}"')

        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        return result

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# matricies have the format: [seq, batch, heads, hdim]
@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        rows = hparams['rows']
        columns = hparams['columns']
        self.num_points = rows * columns

        num_features = hparams['num_features']

        self.layers = nn.ModuleList([DecoderLayer(hparams) for _ in range(hparams.num_layers)])
        if self.hparams.distr == "dmol": # Discretized mixture of logistic, for ordinal valued inputs
            assert self.hparams.channels == 3, "Only supports 3 channels for DML"
            size = (1, self.hparams.channels)
            self.embedding_conv = nn.Conv2d(1, self.hparams.hidden_size, kernel_size=size, stride=size)
            # 10 = 1 + 2c + c(c-1)/2; if only 1 channel, then 3 total
            depth = self.hparams.num_mixtures * 10
        elif self.hparams.distr == "cat": # Categorical
            self.embeds = nn.Embedding(self.num_points * self.hparams.channels, self.hparams.hidden_size)
        else:
            raise ValueError("Only dmol or categorical distributions")

        self.output_projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(self.hparams.hidden_size * self.num_points * self.hparams.channels, num_features, bias=False),
            nn.GELU(),
            nn.LayerNorm([num_features], eps=1e-6, elementwise_affine=True),
        )

    def add_timing_signal(self, X, min_timescale=1.0, max_timescale=1.0e4):
        num_dims = len(X.shape) - 2 # 2 corresponds to batch and hidden_size dimensions
        num_timescales = self.hparams.hidden_size // (num_dims * 2)
        log_timescale_increment = np.log(max_timescale / min_timescale) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp((torch.arange(num_timescales).float() * -log_timescale_increment))
        inv_timescales = inv_timescales.to(X.device)
        total_signal = torch.zeros_like(X) # Only for debugging purposes
        for dim in range(num_dims):
            length = X.shape[dim + 1] # add 1 to exclude batch dim
            position = torch.arange(length).float().to(X.device)
            scaled_time = position.view(-1, 1) * inv_timescales.view(1, -1)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
            prepad = dim * 2 * num_timescales
            postpad = self.hparams.hidden_size - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad))
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            X += signal
            total_signal += signal
        return X

    def shift_and_pad(self, X):
        # Shift inputs over by 1 and pad
        shape = X.shape
        X = X.view(shape[0], shape[1] * shape[2], shape[3])
        X = X[:,:-1,:]
        X = F.pad(X, (0, 0, 1, 0)) # Pad second to last dimension
        X = X.view(shape)
        return X

    def _generate_square_subsequent_mask(self, sz, shift=0):
        mask = torch.ones(sz, sz)
        mask = torch.triu(mask)
        if shift > 0:
            shifted_mask = torch.where(torch.triu(mask, diagonal=shift) == 1, 0, 1)
            mask = mask * shifted_mask
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_orig(self, seq):
        # [B, L, Fin]
        batch_size, seq_len = seq.shape[:2]

        src = seq
        # [B, L, Fin] -> [L, B, Fin]
        src = src.permute((1, 0, 2))

        mask = self._generate_square_subsequent_mask(seq_len, 30).to(seq.device)
        #mask = self._generate_square_subsequent_mask(seq_len).to(seq.device)

        # [L, B, F] -> [L, B, F]
        out = self.encoder(src, mask)

        # [L, B, F] -> [B, F, L]
        out = out.permute((1, 2, 0))

        # [B, F, L] -> [B, 1, L]
        out = self.decoder_projection(out)

        # [B, 1, L] -> [B, L, 1]
        out = out.permute((0, 2, 1))

        return out

    def forward(self, X):
        X = X.permute([0, 2, 3, 1]).contiguous()
        X = X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]) # Flatten channels into width

        # Inputs -> embeddings
        if self.hparams.distr == "dmol":
            # Create a "channel" dimension for the 1x3 convolution
            # (NOTE: can apply a 1x1 convolution and not reshape, this is for consistency)
            X = X.unsqueeze(1)
            X = F.gelu(self.embedding_conv(X))
            X = X.permute([0, 2, 3, 1]) # move channels to the end
        elif self.hparams.distr == "cat":
            # Convert to indexes, and use separate embeddings for different channels
            X = (X * (self.num_points - 1)).long()
            channel_addition = (torch.tensor([0, 1, 2]) * self.num_points).to(X.device).repeat(X.shape[2] // 3).view(1, 1, -1)
            X += channel_addition
            X = self.embeds(X) * (self.hparams.hidden_size ** 0.5)

        X = self.shift_and_pad(X)
        X = self.add_timing_signal(X)
        shape = X.shape
        X = X.view(shape[0], -1, shape[3])

        for layer in self.layers:
            X = layer(X)

        print(f'X: {X.shape}')
        X = self.output_projection(X)

        return X
