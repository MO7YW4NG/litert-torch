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

"""Tests for the Whisper `ModelExtension`."""

import litert_torch.generative.export_hf  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.speech import exportables as speech_exportables
from litert_torch.generative.export_hf.model_ext import exportables as model_ext_exportables
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_torch.generative.export_hf.model_ext.whisper import exportables as whisper_exportables
from litert_torch.generative.export_hf.model_ext.whisper import extension as whisper_extension
import transformers

from absl.testing import absltest as googletest


def _export_config(**extra_kwargs):
  return exportable_module_config.ExportableModuleConfig(
      model='dummy',
      task=exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
      extra_kwargs=extra_kwargs,
  )


class WhisperExtensionTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = whisper_extension.WhisperExtension

  def get_sample_model_config(self):
    return transformers.WhisperConfig()

  def test_resolves_whisper_extension(self):
    ext = extension_lib.resolve_model_extension(transformers.WhisperConfig())
    self.assertIsInstance(ext, whisper_extension.WhisperExtension)

  def test_default_exportables_without_kv_cache_flag(self):
    self.assertEqual(
        model_ext_exportables.get_speech_exportables(
            transformers.WhisperConfig(), _export_config()
        ),
        (
            speech_exportables.LiteRTExportableModuleForAsrEncode,
            speech_exportables.LiteRTExportableModuleForAsrDecode,
        ),
    )

  def test_kv_cache_flag_selects_kv_exportables(self):
    export_config = _export_config(whisper_kv_cache_max_len=448)
    self.assertEqual(
        model_ext_exportables.get_speech_exportables(
            transformers.WhisperConfig(), export_config
        ),
        (
            whisper_exportables.LiteRTExportableModuleForWhisperKVEncode,
            whisper_exportables.LiteRTExportableModuleForWhisperKVDecode,
        ),
    )
    self.assertEqual(
        whisper_exportables.get_kv_cache_max_len(export_config), 448
    )

  def test_negative_kv_cache_flag_raises(self):
    with self.assertRaisesRegex(ValueError, 'whisper_kv_cache_max_len'):
      whisper_exportables.get_kv_cache_max_len(
          _export_config(whisper_kv_cache_max_len=-1)
      )

  def test_kv_cache_flag_zero_keeps_default_exportables(self):
    self.assertIsNone(
        whisper_extension.WhisperExtension().get_speech_exportables(
            transformers.WhisperConfig(),
            _export_config(whisper_kv_cache_max_len=0),
        )
    )


if __name__ == '__main__':
  googletest.main()
