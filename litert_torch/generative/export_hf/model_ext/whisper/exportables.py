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

"""KV-cached encode + decode exportables for Whisper.

Enabled with `--whisper_kv_cache_max_len=<S>` (see `WhisperExtension`). The
converted model has two signatures:

* `encode`: `input_features` -> `cross_kv_{k,v}_{i}` [1, 1500, N, H] for every
  decoder layer i.
* `decode`: `tokens` [1, 1], `input_pos` [1], `kv_cache_{k,v}_{i}` [1, S, N, H]
  and `cross_kv_{k,v}_{i}` -> `logits` [1, 1, V] and the updated
  `kv_cache_{k,v}_{i}`.

The decoder prompt is fed through `decode` one token at a time. A separate
prefill signature is not exported: the converter folds the decoder weights
separately for each signature, so a second decoder signature stores them twice.
"""

from typing import Any

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.speech import exportables as speech_exportables
from litert_torch.generative.export_hf.model_ext.whisper import whisper as whisper_lib
from litert_torch.generative.export_hf.model_ext.whisper import whisper_kv
from litert_torch.generative.layers import kv_cache as kv_utils
import torch
import transformers

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig

KV_CACHE_MAX_LEN_FLAG = "whisper_kv_cache_max_len"


def get_kv_cache_max_len(export_config: ExportableModuleConfig | None) -> int:
  """Returns the self-attention cache length, or 0 if the KV cache is off.

  Args:
    export_config: The export config; `--whisper_kv_cache_max_len` is read from
      its `extra_kwargs`.

  Returns:
    The value of `--whisper_kv_cache_max_len`, or 0 if it is not set.

  Raises:
    ValueError: If the flag is negative.
  """
  if export_config is None:
    return 0
  kv_cache_max_len = int(
      export_config.extra_kwargs.get(KV_CACHE_MAX_LEN_FLAG) or 0
  )
  if kv_cache_max_len < 0:
    raise ValueError(
        f"--{KV_CACHE_MAX_LEN_FLAG} must be >= 0, got {kv_cache_max_len}."
    )
  return kv_cache_max_len


class LiteRTExportableModuleForWhisperKVEncode(
    speech_exportables.LiteRTExportableModuleForAsrEncode
):
  """Whisper encoder that also emits every decoder layer's cross K/V."""

  def __init__(
      self,
      asr_model: whisper_lib.Whisper,
      export_config: ExportableModuleConfig,
  ):
    super().__init__(asr_model, export_config)
    self.encoder = whisper_kv.WhisperCrossKVEncoder(asr_model.get_hf_model())

  def get_sample_inputs(
      self, model_config: transformers.PretrainedConfig, **kwargs
  ) -> dict[str, tuple[Any, dict[str, torch.export.Dim]]]:
    """Returns the encoder sample input under the name `input_features`."""
    (input_features,), dynamic_shapes = super().get_sample_inputs(
        model_config, **kwargs
    )["encode"]
    return {"encode": ({"input_features": input_features}, dynamic_shapes)}


class LiteRTExportableModuleForWhisperKVDecode(
    exportable_module_base.ExportableModuleBase
):
  """One-token Whisper decoder step with an external self-attention cache."""

  def __init__(
      self,
      asr_model: whisper_lib.Whisper,
      export_config: ExportableModuleConfig,
      encoder_output: dict[str, kv_utils.KVCache] | None = None,
  ):
    super().__init__(export_config)
    self.asr_model = asr_model
    self.kv_cache_max_len = get_kv_cache_max_len(export_config)
    self.decoder = whisper_kv.WhisperKVDecoder(
        asr_model.get_hf_model(), self.kv_cache_max_len
    )
    self._encoder_output = encoder_output

  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      cross_kv: kv_utils.KVCache,
  ) -> dict[str, torch.Tensor | kv_utils.KVCache]:
    return self.decoder(tokens, input_pos, kv_cache, cross_kv)

  def get_sample_inputs(
      self, model_config: transformers.PretrainedConfig, **kwargs
  ) -> dict[str, tuple[Any, dict[str, torch.export.Dim]]]:
    """Returns named sample inputs for the `decode` signature."""
    if self._encoder_output is None:
      raise ValueError(
          "encoder_output must be provided to WhisperKVDecode exportable."
      )
    sample_kwargs = {
        "tokens": torch.zeros((1, 1), dtype=torch.int32),
        "input_pos": torch.zeros((1,), dtype=torch.int32),
        "kv_cache": whisper_kv.empty_self_kv(
            model_config, self.kv_cache_max_len
        ),
        "cross_kv": self._encoder_output["cross_kv"],
    }
    return {"decode": (sample_kwargs, {})}
