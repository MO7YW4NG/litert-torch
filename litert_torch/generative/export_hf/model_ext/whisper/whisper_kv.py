# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""KV-cached Whisper modules built from a HF `WhisperForConditionalGeneration`.

The HF model is split into two modules that share its weights:

* `WhisperCrossKVEncoder`: log-mel features -> encoder -> the cross-attention
  K/V of every decoder layer. Cross K/V are projected once per 30 s window
  instead of once per generated token.
* `WhisperKVDecoder`: (tokens, input_pos, self-attention KV cache, cross K/V)
  -> (logits, updated self-attention KV cache). The cache uses the BTNH layout
  ([1, S, num_heads, head_dim]) and is written at `input_pos` with
  `kv_cache.update` (dynamic_update_slice).
"""

from litert_torch.backend.composite import StableHLOCompositeBuilder
from litert_torch.generative.layers import kv_cache as kv_utils
import torch
import torch.nn.functional as F
import transformers


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    mask: torch.Tensor | None,
    use_composite: bool,
) -> torch.Tensor:
  """SDPA on BTNH tensors, optionally as an `odml.scaled_dot_product_attention`.

  Unlike `scaled_dot_product_attention_with_hlfb` in `generative.layers`, a
  missing mask means "attend to everything" (cross-attention), not causal.

  Args:
    q: Query, [B, T, N, H].
    k: Key, [B, S, N, H].
    v: Value, [B, S, N, H].
    scale: Scale applied to the attention logits.
    mask: Additive mask broadcastable to [B, N, T, S], or None for no mask.
    use_composite: Whether to wrap the op in the SDPA composite.

  Returns:
    The attention output, [B, T, N, H].
  """
  builder = None
  if use_composite:
    builder = StableHLOCompositeBuilder(
        "odml.scaled_dot_product_attention", {"scale": scale}
    )
    if mask is None:
      q, k, v = builder.mark_inputs(q, k, v)
    else:
      q, k, v, mask = builder.mark_inputs(q, k, v, mask)
  y = F.scaled_dot_product_attention(
      q.transpose(1, 2),
      k.transpose(1, 2),
      v.transpose(1, 2),
      attn_mask=mask,
      dropout_p=0.0,
      is_causal=False,
      scale=scale,
  ).transpose(1, 2)
  if builder is not None:
    y = builder.mark_outputs(y)
  return y


class WhisperCrossKVEncoder(torch.nn.Module):
  """Runs the encoder and projects every decoder layer's cross-attention K/V."""

  def __init__(self, hf_model: transformers.WhisperForConditionalGeneration):
    super().__init__()
    cfg = hf_model.config
    self.encoder = hf_model.get_encoder()
    self.dec_layers = hf_model.get_decoder().layers
    self.num_heads = cfg.decoder_attention_heads
    self.head_dim = cfg.d_model // cfg.decoder_attention_heads

  def forward(
      self, input_features: torch.Tensor
  ) -> dict[str, kv_utils.KVCache]:
    """Encodes log-mel features.

    Args:
      input_features: Log-mel features, [1, num_mel_bins, 3000].

    Returns:
      `{"cross_kv": KVCache}` with one entry per decoder layer; K and V are
      [1, 1500, num_heads, head_dim] each.
    """
    h = self.encoder(input_features=input_features).last_hidden_state
    b, s, _ = h.shape
    entries = []
    for layer in self.dec_layers:
      attn = layer.encoder_attn
      k = attn.k_proj(h).reshape(b, s, self.num_heads, self.head_dim)
      v = attn.v_proj(h).reshape(b, s, self.num_heads, self.head_dim)
      entries.append(kv_utils.KVCacheEntry(k, v))
    return {"cross_kv": kv_utils.KVCache(tuple(entries))}


class WhisperKVDecoder(torch.nn.Module):
  """Whisper decoder on T tokens with an external self-attention KV cache."""

  def __init__(
      self,
      hf_model: transformers.WhisperForConditionalGeneration,
      kv_cache_max_len: int,
      use_composite: bool = True,
  ):
    """Initializes the decoder from the HF model's weights.

    Args:
      hf_model: The HF model whose decoder weights are used.
      kv_cache_max_len: Length S of the self-attention cache.
      use_composite: Whether SDPA is wrapped in the SDPA composite.

    Raises:
      ValueError: If `kv_cache_max_len` is not in
        [1, config.max_target_positions].
    """
    super().__init__()
    cfg = hf_model.config
    if not 0 < kv_cache_max_len <= cfg.max_target_positions:
      raise ValueError(
          f"kv_cache_max_len={kv_cache_max_len} must be in"
          f" [1, max_target_positions={cfg.max_target_positions}]"
      )
    dec = hf_model.get_decoder()
    self.embed_tokens = dec.embed_tokens
    self.embed_positions = dec.embed_positions
    self.layers = dec.layers
    self.layer_norm = dec.layer_norm
    self.proj_out = hf_model.proj_out
    self.num_heads = cfg.decoder_attention_heads
    self.head_dim = cfg.d_model // cfg.decoder_attention_heads
    self.scale = self.head_dim**-0.5
    self.use_composite = use_composite
    mask = torch.full((kv_cache_max_len, kv_cache_max_len), float("-inf"))
    self.register_buffer(
        "mask_cache", mask.triu(diagonal=1)[None, None], persistent=False
    )

  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      cross_kv: kv_utils.KVCache,
  ) -> dict[str, torch.Tensor | kv_utils.KVCache]:
    """Runs the decoder on `tokens` at positions `input_pos`.

    Args:
      tokens: Token ids, int32 [1, T].
      input_pos: Positions of `tokens`, int32 [T], contiguous.
      kv_cache: Self-attention cache, [1, S, num_heads, head_dim] per entry.
      cross_kv: Cross-attention K/V from `WhisperCrossKVEncoder`.

    Returns:
      `{"logits": [1, T, vocab_size], "kv_cache": updated self-attention
      cache}`.
    """
    b, t = tokens.shape
    n, hd = self.num_heads, self.head_dim
    x = self.embed_tokens(tokens) + self.embed_positions.weight.index_select(
        0, input_pos
    )
    mask = self.mask_cache.index_select(2, input_pos)

    new_entries = []
    for i, layer in enumerate(self.layers):
      # Self-attention; the new K/V are written into the cache first.
      sa = layer.self_attn
      h = layer.self_attn_layer_norm(x)
      q = sa.q_proj(h).reshape(b, t, n, hd)
      k = sa.k_proj(h).reshape(b, t, n, hd)
      v = sa.v_proj(h).reshape(b, t, n, hd)
      entry = kv_utils.update(kv_cache.caches[i], input_pos, k, v)
      new_entries.append(entry)
      a = _sdpa(
          q, entry.k_cache, entry.v_cache, self.scale, mask, self.use_composite
      )
      x = x + sa.out_proj(a.reshape(b, t, n * hd))

      # Cross-attention over the precomputed encoder K/V; no mask.
      ca = layer.encoder_attn
      h = layer.encoder_attn_layer_norm(x)
      q = ca.q_proj(h).reshape(b, t, n, hd)
      ce = cross_kv.caches[i]
      a = _sdpa(q, ce.k_cache, ce.v_cache, self.scale, None, self.use_composite)
      x = x + ca.out_proj(a.reshape(b, t, n * hd))

      h = layer.final_layer_norm(x)
      x = x + layer.fc2(layer.activation_fn(layer.fc1(h)))

    logits = self.proj_out(self.layer_norm(x))
    return {"logits": logits, "kv_cache": kv_utils.KVCache(tuple(new_entries))}


def empty_self_kv(
    config: transformers.WhisperConfig, kv_cache_max_len: int
) -> kv_utils.KVCache:
  """Returns a zero self-attention cache for `WhisperKVDecoder`.

  Args:
    config: The Whisper model config.
    kv_cache_max_len: Length S of the cache.

  Returns:
    A `KVCache` with one [1, S, num_heads, head_dim] K/V entry per decoder
    layer.
  """
  n = config.decoder_attention_heads
  shape = (1, kv_cache_max_len, n, config.d_model // n)
  return kv_utils.KVCache(
      tuple(
          kv_utils.KVCacheEntry(torch.zeros(shape), torch.zeros(shape))
          for _ in range(config.decoder_layers)
      )
  )
