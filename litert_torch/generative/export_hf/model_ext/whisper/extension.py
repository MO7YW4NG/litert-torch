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
"""Built-in `ModelExtension` for Whisper ASR (`whisper`).

Provides the `Whisper` speech wrapper (`whisper/whisper.py`) and attention
patches (`whisper/patch.py`). By default it uses the ASR encode + decode
exportables in `core/speech/exportables.py`. With
`--whisper_kv_cache_max_len=<S>` it exports the KV-cached encode + decode
signatures in `whisper/exportables.py` instead.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.whisper import exportables as whisper_exportables
# Registers whisper patches.
from litert_torch.generative.export_hf.model_ext.whisper import patch as _
from litert_torch.generative.export_hf.model_ext.whisper import whisper as whisper_lib


@extension_lib.register_model_extension(override=False)
class WhisperExtension(extension_lib.ModelExtension):
  """Built-in extension for OpenAI Whisper encoder-decoder ASR models."""

  model_types = ('whisper',)
  extra_kwargs_keys = (whisper_exportables.KV_CACHE_MAX_LEN_FLAG,)

  def get_speech_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config
    if whisper_exportables.get_kv_cache_max_len(export_config) > 0:
      return (
          whisper_exportables.LiteRTExportableModuleForWhisperKVEncode,
          whisper_exportables.LiteRTExportableModuleForWhisperKVDecode,
      )
    return None

  def get_speech_model_cls(self, model_type):
    """See base class."""
    del model_type
    return whisper_lib.Whisper
