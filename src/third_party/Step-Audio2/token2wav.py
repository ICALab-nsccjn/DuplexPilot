import io
import os
import time
from contextlib import nullcontext
import soundfile as sf
import torch
import torchaudio
import s3tokenizer
import onnxruntime
import numpy as np

import torchaudio.compliance.kaldi as kaldi
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml

def fade_in_out(fade_in_mel:torch.Tensor, fade_out_mel:torch.Tensor, window:torch.Tensor):
    """perform fade_in_out in tensor style
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


class Token2wav():

    def __init__(self, model_path, float16=False, profile_sink=None):
        self.float16 = float16
        self._profile_sink = profile_sink
        self._profile_context = {}

        self.audio_tokenizer = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2_25hz.onnx").cuda().eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

        with open(f"{model_path}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
            self.flow = configs['flow']
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

        self.cache = {}

        # stream conf
        self.mel_cache_len = 8  # hard-coded, 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)   # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).cuda()

        # hifigan cache
        self.hift_cache_dict = {}
        # Keep stream cache warmup length consistent with model config.
        self.stream_lookahead_len = int(
            os.environ.get(
                "LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN",
                str(getattr(self.flow, "pre_lookahead_len", 3)),
            ))

    def set_profile_sink(self, sink):
        """Install an optional diagnostic timing sink; serving semantics are unchanged."""
        self._profile_sink = sink

    def set_profile_context(self, **context):
        """Set request identity for the next caller-owned stream operation."""
        self._profile_context = dict(context)

    def _profile_range(self, name):
        try:
            from lychee_fd.runtime.apr.nvtx import range as nvtx_range
            return nvtx_range(name)
        except Exception:
            return nullcontext()

    def _record_profile(self, stage, start_ns, end_ns):
        if self._profile_sink is None:
            return
        try:
            self._profile_sink(
                stage,
                start_monotonic_ns=int(start_ns),
                end_monotonic_ns=int(end_ns),
                session_id=self._profile_context.get("session_id"),
                generation_id=self._profile_context.get("generation_id"),
                sequence_no=self._profile_context.get("sequence_no"),
                state_version=self._profile_context.get("state_version"),
                worker_id=self._profile_context.get("worker_id"),
            )
        except Exception:
            return


    def _prepare_prompt(self, prompt_wav):
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0], device='cuda')

        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).cuda()
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device='cuda')
        prompt_mels = torch.nn.functional.pad(prompt_mels, (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]), mode='replicate')
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def __call__(self, generated_speech_tokens, prompt_wav):
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10)

        # wav, _ = self.hift(speech_feat=mel)
        # output = io.BytesIO()
        # torchaudio.save(output, wav.cpu(), sample_rate=24000, format='wav')
        wav, _ = self.hift(speech_feat=mel)
        wav = wav.squeeze().cpu().numpy()
        output = io.BytesIO()
        sf.write(output, wav, 24000, format="WAV", subtype="PCM_16")

        return output.getvalue()

    def create_stream_state(self, prompt_wav):
        """Create explicit per-stream Flow/HiFT state.

        The returned object is owned by the caller.  The model instance keeps
        only immutable weights and prompt preprocessing cache; it does not
        become the owner of a logical stream's continuation cache.
        """
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]
        lookahead_len = max(1, int(self.stream_lookahead_len))
        if prompt_speech_tokens.shape[1] >= lookahead_len:
            lookahead_tail = prompt_speech_tokens[:, :lookahead_len]
        else:
            repeats = (lookahead_len + prompt_speech_tokens.shape[1] - 1) // max(
                1, prompt_speech_tokens.shape[1])
            lookahead_tail = prompt_speech_tokens.repeat(1, repeats)[:, :lookahead_len]
        return {
            "prompt_wav": str(prompt_wav),
            "flow_cache": self.flow.setup_cache(
                torch.cat([prompt_speech_tokens, lookahead_tail], dim=1),
                prompt_mels, spk_emb, n_timesteps=10),
            "hift_cache": dict(
                mel = torch.zeros(1, prompt_mels.shape[2], 0, device='cuda'),
                source = torch.zeros(1, 1, 0, device='cuda'),
                speech = torch.zeros(1, 0, device='cuda'),
            ),
        }

    def set_stream_cache(self, prompt_wav):
        """Legacy B1 compatibility wrapper for callers not yet state-aware."""
        state = self.create_stream_state(prompt_wav)
        self.stream_cache = state["flow_cache"]
        self.hift_cache_dict = state["hift_cache"]

    @staticmethod
    def _flow_step_token_tensor(tokens):
        if isinstance(tokens, torch.Tensor):
            return tokens
        return torch.tensor([list(tokens)], dtype=torch.int32)

    def begin_chunk_steps(
        self,
        generated_speech_tokens,
        prompt_wav,
        state,
        *,
        speaker=None,
        last_chunk: bool = False,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        request_id: str = "",
        generation_id: int = 0,
        sequence_no: int = 0,
        version: int = 0,
    ):
        """Expose the Flow step contract without reading private Flow state."""
        if not isinstance(state, dict) or not isinstance(state.get("flow_cache"), dict):
            raise ValueError("state must contain a caller-owned flow_cache mapping")
        flow_cache = state["flow_cache"]
        if speaker is None and prompt_wav in getattr(self, "cache", {}):
            speaker = self.cache[prompt_wav][2]
        if speaker is None:
            raise ValueError("speaker embedding is required for Flow step preparation")
        token = self._flow_step_token_tensor(generated_speech_tokens)
        reference = flow_cache.get("estimator_att_cache")
        if isinstance(reference, torch.Tensor):
            token = token.to(reference.device)
        elif isinstance(reference, (tuple, list)) and reference and isinstance(
                reference[0], torch.Tensor):
            token = token.to(reference[0].device)
        elif torch.cuda.is_available():
            token = token.cuda()
        prepared_cache = dict(flow_cache)
        step_state = self.flow.begin_chunk_steps(
            token=token,
            spk=speaker,
            cache=prepared_cache,
            last_chunk=last_chunk,
            n_timesteps=n_timesteps,
            temperature=temperature,
            request_id=request_id,
            generation_id=generation_id,
            sequence_no=sequence_no,
            version=version,
        )
        # Transfer the estimator cache to the explicit step state.  Retaining
        # the padded legacy tensors here would double request-owned GPU memory.
        flow_cache["estimator_cnn_cache"] = None
        flow_cache["estimator_att_cache"] = None
        return step_state

    def advance_chunk_step(self, state):
        """Advance one caller-owned Flow step state."""
        return self.flow.advance_chunk_step(state)

    def advance_chunk_step_batch(self, states):
        """Advance compatible caller-owned Flow states as one CFG batch."""
        return self.flow.advance_chunk_step_batch(states)

    def advance_chunk_step_variable_batch(self, states):
        """Advance an opt-in variable-attention-cache Flow batch."""
        return self.flow.advance_chunk_step_variable_batch(states)

    def advance_chunk_step_variable_mixed_batch(self, states):
        """Advance an opt-in variable-cache batch at mixed Euler steps."""
        return self.flow.advance_chunk_step_variable_mixed_batch(states)

    def advance_chunk_step_variable_mixed_batch_b4(self, states):
        """Advance the explicitly exploratory mixed-step batch with cap four."""
        return self.flow.advance_chunk_step_variable_mixed_batch_b4(states)

    def schedule_terminal_step(self, state):
        # Schedule one public terminal Flow step for a caller-owned state.
        return self.flow.schedule_terminal_step(state)

    def finish_chunk_steps(self, state):
        """Finalize Flow steps and return mel plus public estimator caches."""
        return self.flow.finish_chunk_steps(state)
    def render_chunk_pcm(self, chunk_mel, state, *, last_chunk: bool = False):
        """Render one finished Flow chunk through the caller-owned HiFT state."""
        if not isinstance(state, dict):
            raise ValueError("stream state must be a dictionary")
        hift_cache = state.get("hift_cache")
        if not isinstance(hift_cache, dict):
            raise ValueError("stream state is missing hift_cache")
        required = ("mel", "source", "speech")
        if any(key not in hift_cache for key in required):
            raise ValueError("hift_cache is incomplete")

        total_start = time.monotonic_ns()
        hift_start = total_start
        with self._profile_range("TOKEN2WAV_HIFT"):
            mel = torch.concat([hift_cache["mel"], chunk_mel], dim=2)
            speech, source = self.hift(mel, hift_cache["source"])
            if hift_cache["speech"].shape[-1] > 0:
                speech = fade_in_out(speech, hift_cache["speech"], self.speech_window)
            next_hift_cache = {
                "mel": mel[..., -self.mel_cache_len:].clone().detach(),
                "source": source[:, :, -self.source_cache_len:].clone().detach(),
                "speech": speech[:, -self.source_cache_len:].clone().detach(),
            }
        self._record_profile("TOKEN2WAV_HIFT", hift_start, time.monotonic_ns())
        state["hift_cache"] = next_hift_cache

        vocoder_start = time.monotonic_ns()
        with self._profile_range("TOKEN2WAV_VOCODER"):
            if not last_chunk:
                speech = speech[:, :-self.source_cache_len]
            wav_np = speech.cpu().numpy()
            wav_np = np.clip(wav_np, -1.0, 1.0)
            wav_int16 = (wav_np * 32767.0).astype("<i2")
            pcm_bytes = wav_int16.tobytes()
        self._record_profile("TOKEN2WAV_VOCODER", vocoder_start, time.monotonic_ns())
        self._record_profile("PCM_GENERATION", total_start, time.monotonic_ns())
        return pcm_bytes


    def stream(self, generated_speech_tokens, prompt_wav, last_chunk=False):
        """Legacy B1 wrapper that keeps the old instance-owned state contract."""
        if not hasattr(self, "stream_cache") or self.stream_cache is None:
            raise ValueError("stream_cache is not set")
        state = {
            "prompt_wav": str(prompt_wav),
            "flow_cache": self.stream_cache,
            "hift_cache": self.hift_cache_dict,
        }
        pcm_bytes = self.stream_with_state(
            generated_speech_tokens,
            prompt_wav,
            state,
            last_chunk=last_chunk,
        )
        self.stream_cache = state["flow_cache"]
        self.hift_cache_dict = state["hift_cache"]
        return pcm_bytes

    def stream_with_state(
        self,
        generated_speech_tokens,
        prompt_wav,
        state,
        last_chunk=False,
    ):
        """Run one stream chunk using caller-owned logical state.

        No mutable continuation cache is read from or written to the model
        object.  A future B2 path can pass one state per physical row and
        commit each returned state by logical stream ID.
        """
        if not isinstance(state, dict):
            raise ValueError("stream state must be a dictionary")
        if str(state.get("prompt_wav")) != str(prompt_wav):
            raise ValueError("stream state prompt does not match request")
        flow_cache = state.get("flow_cache")
        hift_cache_dict = state.get("hift_cache")
        if flow_cache is None or not isinstance(hift_cache_dict, dict):
            raise ValueError("stream state is not initialized")
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        total_start = time.monotonic_ns()
        token_start = total_start
        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')
        token_end = time.monotonic_ns()
        self._record_profile("TOKEN2WAV_TOKEN_QUEUE", token_start, token_end)

        flow_start = time.monotonic_ns()
        with self._profile_range("TOKEN2WAV_FLOW"):
            with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
                chunk_mel, flow_cache = self.flow.inference_chunk(
                    token=generated_speech_tokens,
                    spk=spk_emb,
                    cache=flow_cache,
                    last_chunk=last_chunk,
                    n_timesteps=10,
                )
            if flow_cache['estimator_att_cache'].shape[4] > (prompt_mels.shape[1] + 100):
                flow_cache['estimator_att_cache'] = torch.cat([
                    flow_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],
                    flow_cache['estimator_att_cache'][:, :, :, :, -100:],
                ], dim=4)
        self._record_profile("TOKEN2WAV_FLOW", flow_start, time.monotonic_ns())
        
        # vocoder cache
        hift_start = time.monotonic_ns()
        with self._profile_range("TOKEN2WAV_HIFT"):
            hift_cache_mel = hift_cache_dict['mel']
            hift_cache_source = hift_cache_dict['source']
            hift_cache_speech = hift_cache_dict['speech']
            mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

            speech, source = self.hift(mel, hift_cache_source)

            # overlap speech smooth
            if hift_cache_speech.shape[-1] > 0:
                speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

            # update vocoder cache
            hift_cache_dict = dict(
                mel = mel[..., -self.mel_cache_len:].clone().detach(),
                source = source[:, :, -self.source_cache_len:].clone().detach(),
                speech = speech[:, -self.source_cache_len:].clone().detach(),
            )
        self._record_profile("TOKEN2WAV_HIFT", hift_start, time.monotonic_ns())
        state["flow_cache"] = flow_cache
        state["hift_cache"] = hift_cache_dict
        vocoder_start = time.monotonic_ns()
        with self._profile_range("VOCODER"):
            if not last_chunk:
                speech = speech[:, :-self.source_cache_len]

            wav_np = speech.cpu().numpy()
            # Clip to [-1, 1] to avoid overflow, then scale to int16
            wav_np = np.clip(wav_np, -1.0, 1.0)
            wav_int16 = (wav_np * 32767.0).astype('<i2')  # 16-bit little-endian PCM
            pcm_bytes = wav_int16.tobytes()
        self._record_profile("TOKEN2WAV_VOCODER", vocoder_start, time.monotonic_ns())
        self._record_profile("PCM_GENERATION", total_start, time.monotonic_ns())
        return pcm_bytes

if __name__ == '__main__':
    token2wav = Token2wav('Step-Audio-2-mini/token2wav')

    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    audio = token2wav(tokens, 'assets/default_male.wav')
    with open('assets/give_me_a_brief_introduction_to_the_great_wall.wav', 'wb') as f:
        f.write(audio)
