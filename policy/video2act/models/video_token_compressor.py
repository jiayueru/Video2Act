#!/usr/bin/env python
# coding=utf-8
"""
Video_token compressor_3D module adapted from the prior video policy for video feature compression.
This module compresses video features from [B, 17664, 3072] to [B, num_latents, dim]
with temporal attention capabilities for 3D processing.
"""

import torch
from torch import einsum, nn
import torch.nn.functional as F
try:
    from einops import rearrange, repeat
    from einops_exts import rearrange_many
    einops_available = True
except ImportError:
    einops_available = False
    # Simple fallback for basic operations
    def rearrange(tensor, pattern, **kwargs):
        # Very basic fallback - only handles specific common patterns
        if pattern == 'b T n d -> (b T) n d':
            b, T, n, d = tensor.shape
            return tensor.contiguous().view(b*T, n, d)
        elif pattern == 'b T q d -> (b T) q d':
            b, T, q, d = tensor.shape  
            return tensor.contiguous().view(b*T, q, d)
        elif pattern == '(b T) q d -> (b q) T d' and 'b' in kwargs:
            bT, q, d = tensor.shape
            b = kwargs['b']
            T = bT // b
            return tensor.contiguous().view(b, T, q, d).transpose(1, 2).contiguous().view(b*q, T, d)
        elif pattern == '(b q) T d -> (b T) q d' and 'b' in kwargs:
            bq, T, d = tensor.shape
            b = kwargs['b']
            q = bq // b
            return tensor.contiguous().view(b, q, T, d).transpose(1, 2).contiguous().view(b*T, q, d)
        elif pattern == 'b T q d -> b (T q) d':
            b, T, q, d = tensor.shape
            return tensor.contiguous().view(b, T*q, d)
        elif pattern == 'b f (h d) -> b h f d' and 'h' in kwargs:
            b, f, hd = tensor.shape
            h = kwargs['h']
            d = hd // h
            return tensor.contiguous().view(b, f, h, d).transpose(1, 2)
        elif pattern == 'b h q v -> b q (h v)':
            b, h, q, v = tensor.shape
            return tensor.transpose(1, 2).contiguous().view(b, q, h*v)
        elif pattern == 'b q (h d) -> b h q d' and 'h' in kwargs:
            b, q, hd = tensor.shape
            h = kwargs['h']
            d = hd // h
            return tensor.contiguous().view(b, q, h, d).transpose(1, 2)
        else:
            return tensor
            
    def repeat(tensor, pattern, **kwargs):
        if pattern == 'T q d -> b T q d' and 'b' in kwargs:
            b = kwargs['b']
            return tensor.unsqueeze(0).expand(b, -1, -1, -1)
        elif pattern == 'b q d -> b T q d' and 'T' in kwargs:
            T = kwargs['T']
            return tensor.unsqueeze(1).expand(-1, T, -1, -1)
        else:
            return tensor
            
    def rearrange_many(tensors, pattern, **kwargs):
        return tuple(rearrange(t, pattern, **kwargs) for t in tensors)


def feed_forward_layer(dim, mult=4, activation='gelu', dtype=torch.float32):
    """Feed forward layer with activation"""
    inner_dim = dim * mult
    activation_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
    
    return nn.Sequential(
        nn.LayerNorm(dim, dtype=dtype),
        nn.Linear(dim, inner_dim),
        activation_fn,
        nn.Linear(inner_dim, dim),
    )


class Attention(nn.Module):
    """Standard attention module for temporal processing"""
    
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            use_cross_attn=False,
            y_dim=512,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_mask=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_mask = attn_mask
        self.use_cross_attn = use_cross_attn
        
        if self.use_cross_attn:
            self.y_kv = nn.Linear(y_dim, dim * 2, bias=qkv_bias)
            self.y_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.gate = nn.Parameter(torch.zeros([self.num_heads]))

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if self.attn_mask is not None:
                self.attn_mask = self.attn_mask.to(device=x.device, dtype=q.dtype)
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=self.attn_mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        if self.use_cross_attn:
            N_y = y.shape[1]
            y_kv = self.y_kv(y).reshape(B, N_y, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            y_k, y_v = y_kv.unbind(0)
            y_k = self.y_k_norm(y_k)
            y_out = F.scaled_dot_product_attention(
                q, y_k, y_v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            y_out = y_out * self.gate.tanh().view(1, -1, 1, 1)
            x = x + y_out

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerceiverAttentionLayer(nn.Module):
    """Perceiver Attention Layer for cross-attention between latents and features"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim, dtype=dtype)
        self.norm_latents = nn.LayerNorm(dim, dtype=dtype)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents):
        """Latent vectors are cross-attending to the visual features x

        Args:
            features: Batch of visual features with shape (batch_size, n_features, dim)
            latents: Latent learnt vectors which are used to compute queries with shape (batch_size, n_latents, dim)

        Returns:
            Attention score with shape (batch_size, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # Layer normalization
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # Compute the queries from the latents, for all attention heads simultaneously
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # Keys and values for all attention heads
        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=n_heads)
        assert v.shape == torch.Size([n_batch, n_heads, n_features_latents, self.dim_head])

        q = q * self.scale

        # Attention scores
        sim = einsum('b h q d, b h f d -> b h q f', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum('b h q f, b h f v -> b h q v', alphas, v)
        out = rearrange(out, 'b h q v -> b q (h v)')

        return self.to_out(out)


class VideoTokenCompressor(nn.Module):
    """
    Video_token compressor_3D for video feature compression with temporal attention.
    
    This module compresses video features from [B, 17664, 3072] to [B, num_latents, dim],
    providing the same compression ratio as in the prior video policy: ~3.8x compression.
    Includes temporal processing capabilities for 3D understanding.
    """

    def __init__(
            self,
            dim: int = 512,  # Output dimension (same as VPP: latent_dim=512)
            depth: int = 3,  # Same as VPP: token compressor_depth=3
            condition_dim: int = 3072,  # Input video feature dimension
            dim_head: int = 64,  # Same as VPP: token compressor_dim_head=64
            heads: int = 8,  # Same as VPP: token compressor_heads=8
            num_latents: int = 224,  # Output tokens (from VPP: 224)
            num_frame: int = 1,  # Number of frames (framepack processes as single "frame")
            num_time_embeds: int = 1,  # Time embeddings
            ff_mult: int = 4,  # Feed forward multiplier
            activation: str = 'gelu',
            trainable: bool = True,
            use_temporal: bool = False,  # Enable temporal attention for 3D processing
            dtype: torch.dtype = torch.float32,  # Add dtype parameter
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents
        self.num_frame = num_frame
        self.condition_dim = condition_dim
        self.use_temporal = use_temporal

        # Project input video features to internal dimension
        self.goal_emb = nn.Sequential(
            nn.Linear(condition_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Learnable latent vectors for compression
        seq_len = num_latents // num_frame
        self.latents = nn.Parameter(torch.randn(num_frame, seq_len, dim))
        
        # Time positional embeddings
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))
        
        # Attention mask for temporal attention (if used)
        attn_mask = torch.ones((num_frame, num_frame))
        # attn_mask = torch.tril(attn_mask).bool()  # Causal mask if needed

        # Transformer layers with perceiver attention + optional temporal attention + feed forward
        self.layers = nn.ModuleList([])
        
        if self.use_temporal:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads, dtype=dtype),
                            Attention(dim, num_heads=heads, qkv_bias=True, use_cross_attn=False,
                                      y_dim=512, attn_mask=attn_mask),  # Temporal attention
                            feed_forward_layer(dim=dim, mult=ff_mult, activation=activation, dtype=dtype),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads, dtype=dtype),
                            feed_forward_layer(dim=dim, mult=ff_mult, activation=activation, dtype=dtype),
                        ]
                    )
                )

        # Final layer normalization
        self.norm = nn.LayerNorm(dim, dtype=dtype)

        self._update_trainable_state(trainable)

    def _update_trainable_state(self, trainable: bool = True):
        """Set trainable state for all parameters"""
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, framepack_features: torch.Tensor, mask: torch.BoolTensor = None, extra: torch.Tensor = None):
        """
        Compress video features using 3D perceiver resampler with temporal attention.

        Args:
            framepack_features: Video2Act features 
                - Format 1: [B, N_tokens, D] where N_tokens = T*N (needs reshaping)
                - Format 2: [B, T, N, D] (VPP-compatible format)
            mask: Optional mask for the features of shape [B, T]
            extra: Optional extra features to concatenate

        Returns:
            Compressed features of shape [B, num_latents, dim]
        """
        # Ensure input tensor has the same dtype as model parameters for mixed precision training
        if hasattr(self.goal_emb[0], 'weight') and self.goal_emb[0].weight.dtype != framepack_features.dtype:
            framepack_features = framepack_features.to(dtype=self.goal_emb[0].weight.dtype)
        
        if framepack_features.ndim == 3:
            # Input: [B, T*N, D] -> reshape to [B, T, N, D] (VPP format)
            batch_size, total_tokens, feat_dim = framepack_features.shape
            assert feat_dim == self.condition_dim, f"Expected feature dim {self.condition_dim}, got {feat_dim}"
            
            # Calculate N (spatial tokens per frame) from total tokens and num_frame
            N = total_tokens // self.num_frame
            assert total_tokens == self.num_frame * N, f"Total tokens {total_tokens} must be divisible by num_frame {self.num_frame}"
            
            # Reshape to VPP-compatible format: [B, T, N, D]
            x_f = framepack_features.view(batch_size, self.num_frame, N, feat_dim)
        elif framepack_features.ndim == 4:
            # Already in [B, T, N, D] format (VPP-compatible)
            x_f = framepack_features
            batch_size = x_f.shape[0]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {framepack_features.ndim}D with shape {framepack_features.shape}")

        assert x_f.ndim == 4
        batch_size, max_length, _, dim = x_f.shape
        assert dim == self.condition_dim

        # Apply time positional embeddings
        time_pos_emb = (
            self.time_pos_emb[:max_length].unsqueeze(0).expand(batch_size, -1, -1, -1)
        )  # [batch_size, max_length, 1, dim]
        
        if mask is not None:
            time_pos_emb = time_pos_emb * mask.unsqueeze(-1).unsqueeze(-1)

        # Project to internal dimension and add positional embeddings
        x_f = self.goal_emb(x_f)  # [B, T, N, dim]
        
        # Add extra features if provided (like in VPP 3D)
        if extra is not None:
            extra = repeat(extra, 'b q d -> b T q d', T=max_length)
            x_f = torch.cat([x_f, extra], dim=2)
        
        x_f = x_f + time_pos_emb

        # Flatten the frames: [B, T, N, D] -> [B*T, N, D]
        x_f = rearrange(x_f, 'b T n d -> (b T) n d')

        # Copy the latents for every element in the batch: [T, seq_len, dim] -> [B*T, seq_len, dim]
        x = repeat(self.latents, 'T q d -> b T q d', b=batch_size)
        x = rearrange(x, 'b T q d -> (b T) q d')

        # Apply perceiver attention, temporal attention (if enabled), and feed forward layers
        if self.use_temporal:
            for attn, temp_attn, ffw in self.layers:
                x = x + attn(x_f, x)  # Cross-attention: latents attend to video features
                # Temporal attention: reshape for temporal processing
                x = rearrange(x, '(b T) q d -> (b q) T d', b=batch_size)
                x = x + temp_attn(x)  # Temporal self-attention
                x = rearrange(x, '(b q) T d -> (b T) q d', b=batch_size)
                x = x + ffw(x)        # Self-processing
        else:
            for attn, ffw in self.layers:
                x = x + attn(x_f, x)  # Cross-attention: latents attend to video features
                x = x + ffw(x)        # Self-processing

        # Reshape back to batch format: [B*T, seq_len, dim] -> [B, T*seq_len, dim]
        x = x.reshape(batch_size, -1, x.shape[1], x.shape[2])
        x = rearrange(x, 'b T q d -> b (T q) d')
        
        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim]), f"Shape mismatch: got {x.shape}, expected [{batch_size}, {self.num_queries}, {self.dim}]"
        
        # Final normalization
        x = self.norm(x)

        return x