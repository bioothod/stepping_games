import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, hparams):
        super().__init__()
        self.attn = Attn(hparams)
        self.hparams = hparams
        self.dropout = nn.Dropout(p=hparams.dropout)
        self.layernorm_attn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.hparams.filter_size, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hparams.filter_size, self.hparams.hidden_size, bias=True))

    def preprocess_(self, X):
        return X

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        X = self.preprocess_(X)
        y = self.attn(X)
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(self.preprocess_(X))
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X

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
        assert self.kd % self.hparams.num_heads == 0
        assert self.vd % self.hparams.num_heads == 0

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
        # Split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (self.hparams.num_heads, self.kd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.hparams.num_heads, self.kd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.hparams.num_heads, self.vd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.hparams.num_heads) ** (-0.5)

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
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# matricies have the format: [seq, batch, hdim]
@torch.jit.script
def apply_rotary_pos_emb(q, cos, sin):
    return (q * cos) + (rotate_half(q) * sin)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        rows = config['rows']
        columns = config['columns']
        num_features = config['num_features']

        input_dim = 3
        self.d_model = 3
        dim_feedforward = 32
        dropout_rate = 0.
        self.seq_len = rows * columns

        self.rotary = Rotary(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model + 2, nhead=1, dim_feedforward=dim_feedforward, dropout=dropout_rate, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=16)

        self.decoder_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.d_model + 2) * self.seq_len, num_features),
            nn.GELU(),
        )

    def _generate_square_subsequent_mask(self, sz, shift=0):
        mask = torch.ones(sz, sz)
        mask = torch.triu(mask)
        if shift > 0:
            shifted_mask = torch.where(torch.triu(mask, diagonal=shift) == 1, 0, 1)
            mask = mask * shifted_mask
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, game_states):
        # [B, C, H, W]

        batch_size, num_channels = game_states.shape[:2]

        state = game_states.view(batch_size, num_channels, self.seq_len)

        # [B, Fin, L] -> [L, B, Fin]
        src = state.permute((2, 0, 1))

        cos, sin = self.rotary(src, 0)
        #print(f'embs: {embs.shape}, src: {src.shape}, cos: {cos.shape}')

        #src = apply_rotary_pos_emb(src, cos, sin)

        cos = cos[..., :1]
        sin = sin[..., :1]
        cos = torch.tile(cos, [1, batch_size, 1])
        sin = torch.tile(sin, [1, batch_size, 1])
        #print(f'src: {src.shape}, cos: {cos.shape}, sin: {sin.shape}')
        src = torch.cat([src, cos, sin], 2)

        #mask = self._generate_square_subsequent_mask(seq_len, 30).to(seq.device)
        mask = self._generate_square_subsequent_mask(self.seq_len).to(game_states.device)

        # [L, B, F] -> [L, B, F]
        out = self.encoder(src, mask)

        # [L, B, F] -> [B, F, L]
        out = out.permute((1, 2, 0))

        # [B, F, L] -> [B, 1, L]
        out = self.decoder_projection(out)

        return out
