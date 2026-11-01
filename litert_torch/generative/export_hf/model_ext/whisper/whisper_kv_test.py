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

"""Tests for the KV-cached Whisper modules and their export, on a tiny model."""

import litert_torch
import litert_torch.generative.export_hf  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.speech import asr_model
from litert_torch.generative.export_hf.model_ext.whisper import exportables
from litert_torch.generative.export_hf.model_ext.whisper import whisper as whisper_lib
from litert_torch.generative.export_hf.model_ext.whisper import whisper_kv
import numpy as np
import torch
import transformers

from absl.testing import absltest as googletest

_KV_LEN = 32


def _tiny_whisper() -> transformers.WhisperForConditionalGeneration:
  torch.manual_seed(0)
  config = transformers.WhisperConfig(
      vocab_size=96,
      num_mel_bins=80,
      d_model=64,
      encoder_layers=2,
      encoder_attention_heads=4,
      encoder_ffn_dim=128,
      decoder_layers=2,
      decoder_attention_heads=4,
      decoder_ffn_dim=128,
      max_source_positions=1500,
      max_target_positions=_KV_LEN,
      pad_token_id=0,
      bos_token_id=1,
      eos_token_id=2,
      decoder_start_token_id=1,
  )
  return transformers.WhisperForConditionalGeneration(config).eval()


class _FeatureProcessor(asr_model.AsrProcessor):
  """Log-mel features only; the tiny model has no tokenizer files."""

  def __init__(self, num_mel_bins: int):
    self._fe = transformers.WhisperFeatureExtractor(feature_size=num_mel_bins)

  def get_sampling_rate(self) -> int:
    return self._fe.sampling_rate

  def process(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
    return self._fe(
        audio, sampling_rate=self._fe.sampling_rate, return_tensors='pt'
    )

  def decode(self, tokens: list[int] | torch.Tensor) -> list[str]:
    raise NotImplementedError()


class _TinyWhisper(whisper_lib.Whisper):
  """`Whisper` wrapper around an in-memory model instead of a checkpoint."""

  def __init__(self, hf_model: transformers.WhisperForConditionalGeneration):
    asr_model.AsrModel.__init__(self)
    self._model = hf_model
    self._replace_normalizations(self._model)
    self._encoder = whisper_lib.WhisperEncoder(self._model).eval()
    self._decoder = whisper_lib.WhisperDecoder(self._model).eval()
    self._processor = _FeatureProcessor(hf_model.config.num_mel_bins)


class WhisperKVTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.feats = torch.randn(1, 80, 3000)
    self.tokens = [1, 5, 7, 11, 13, 17, 19, 23, 29, 31]

  def test_cached_decoding_matches_hf_forward(self):
    """Prefill 4 tokens, then 1 token per step; compare with one HF pass."""
    hf = _tiny_whisper()
    encoder = whisper_kv.WhisperCrossKVEncoder(hf).eval()
    decoder = whisper_kv.WhisperKVDecoder(hf, _KV_LEN, use_composite=False)
    with torch.no_grad():
      want = hf(
          input_features=self.feats,
          decoder_input_ids=torch.tensor([self.tokens]),
      ).logits[0]
      cross_kv = encoder(self.feats)['cross_kv']
      kv = whisper_kv.empty_self_kv(hf.config, _KV_LEN)
      out = decoder(
          torch.tensor([self.tokens[:4]], dtype=torch.int32),
          torch.arange(4, dtype=torch.int32),
          kv,
          cross_kv,
      )
      got = [out['logits'][0]]
      kv = out['kv_cache']
      for pos in range(4, len(self.tokens)):
        out = decoder(
            torch.tensor([[self.tokens[pos]]], dtype=torch.int32),
            torch.tensor([pos], dtype=torch.int32),
            kv,
            cross_kv,
        )
        got.append(out['logits'][0])
        kv = out['kv_cache']
    torch.testing.assert_close(torch.cat(got), want, atol=1e-4, rtol=1e-4)

  def test_kv_cache_max_len_is_checked(self):
    with self.assertRaisesRegex(ValueError, 'max_target_positions'):
      whisper_kv.WhisperKVDecoder(_tiny_whisper(), _KV_LEN + 1)

  def test_export_asr_models_matches_torch(self):
    asr = _TinyWhisper(_tiny_whisper())
    hf = asr.get_hf_model()
    export_config = exportable_module_config.ExportableModuleConfig(
        model='dummy',
        task=exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        quantization_recipe=None,
        work_dir=self.create_tempdir().full_path,
        extra_kwargs={exportables.KV_CACHE_MAX_LEN_FLAG: _KV_LEN},
    )
    artifacts = export_lib.export_asr_models(
        export_lib.SourceModelArtifacts(
            model=asr,
            model_config=hf.config,
            text_model_config=None,
            tokenizer=None,
        ),
        export_config,
        export_lib.ExportedModelArtifacts(),
    )
    edge_model = litert_torch.load(artifacts.prefill_decode_model_path)

    encoder = whisper_kv.WhisperCrossKVEncoder(hf).eval()
    decoder = whisper_kv.WhisperKVDecoder(hf, _KV_LEN).eval()
    with torch.no_grad():
      cross_kv = encoder(self.feats)['cross_kv']
    enc_out = edge_model(
        input_features=self.feats.numpy(), signature_name='encode'
    )
    n_layers = hf.config.decoder_layers
    self.assertCountEqual(
        enc_out.keys(),
        [f'cross_kv_{t}_{i}' for t in 'kv' for i in range(n_layers)],
    )
    for i, entry in enumerate(cross_kv.caches):
      np.testing.assert_allclose(
          enc_out[f'cross_kv_k_{i}'], entry.k_cache.numpy(), atol=1e-4
      )
      np.testing.assert_allclose(
          enc_out[f'cross_kv_v_{i}'], entry.v_cache.numpy(), atol=1e-4
      )

    # Feed the first tokens one at a time through both models.
    kv = whisper_kv.empty_self_kv(hf.config, _KV_LEN)
    kv_shape = kv.caches[0].k_cache.shape
    tfl_kv = {
        f'kv_cache_{t}_{i}': np.zeros(kv_shape, np.float32)
        for t in 'kv'
        for i in range(n_layers)
    }
    for pos, token in enumerate(self.tokens[:6]):
      with torch.no_grad():
        out = decoder(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([pos], dtype=torch.int32),
            kv,
            cross_kv,
        )
      kv = out['kv_cache']
      tfl_out = edge_model(
          tokens=np.array([[token]], np.int32),
          input_pos=np.array([pos], np.int32),
          **tfl_kv,
          **enc_out,
          signature_name='decode',
      )
      tfl_kv = {k: v for k, v in tfl_out.items() if k.startswith('kv_cache')}
      self.assertLen(tfl_kv, 2 * n_layers)
      np.testing.assert_allclose(
          tfl_out['logits'], out['logits'].numpy(), atol=1e-4, rtol=1e-4
      )


if __name__ == '__main__':
  googletest.main()
