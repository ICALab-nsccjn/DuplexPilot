"""Diagnostic APR bridge for the local Token2Wav checkpoint adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..token2wav_checkpoint import LocalToken2WavCheckpointAdapter
from .acoustic_backend import AcousticBackend
from .contracts import AcousticPcmRecord


class LocalToken2WavAPRBackend(AcousticBackend):
    """Expose the local checkpoint adapter through APR's state contract.

    This bridge is intentionally local and diagnostic-only. It does not wire
    the remote Token2Wav service or change the production serving selector.
    """

    def __init__(
        self,
        model: Any,
        *,
        request_id: str,
        stream_id: str,
        generation_id: int,
        prompt_wav: str,
        stream_state: Mapping[str, Any],
        sample_rate: int = 24000,
    ) -> None:
        self._adapter = LocalToken2WavCheckpointAdapter(
            model,
            request_id=request_id,
            stream_id=stream_id,
            generation_id=generation_id,
            prompt_wav=prompt_wav,
            stream_state=stream_state,
        )
        self._request_id = request_id
        self._stream_id = stream_id
        self._generation_id = generation_id
        self._sample_rate = sample_rate
        self._next_batch_sequence = 0
        self._current_batch_sequence = 0
        self._next_pcm_sequence = 0

    def capture_state(self, request_id: str) -> Mapping[str, Any]:
        if request_id != self._request_id:
            raise ValueError("request ownership mismatch")
        return {
            "token2wav_checkpoint": self._adapter.capture_state(),
            "next_batch_sequence": self._next_batch_sequence,
            "next_pcm_sequence": self._next_pcm_sequence,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("APR local backend state must be a mapping")
        if not state:
            return
        checkpoint = state.get("token2wav_checkpoint")
        if checkpoint is None:
            raise ValueError("APR local backend state is missing checkpoint")
        self._adapter.restore_state(checkpoint)
        self._next_batch_sequence = int(state["next_batch_sequence"])
        self._next_pcm_sequence = int(state["next_pcm_sequence"])

    def resume(self) -> None:
        if self._adapter.cancelled:
            raise RuntimeError("cancelled local acoustic stream cannot resume")

    def process(self, tokens: tuple[int, ...]) -> None:
        self._current_batch_sequence = self._next_batch_sequence
        self._next_batch_sequence += 1
        self._adapter.set_profile_context(
            session_id=self._request_id,
            generation_id=self._generation_id,
            sequence_no=self._current_batch_sequence,
        )
        self._adapter.process(tokens)

    def commit_pcm(self) -> Sequence[AcousticPcmRecord]:
        chunks = self._adapter.commit_output()
        records = []
        for chunk in chunks:
            records.append(
                AcousticPcmRecord(
                    request_id=self._request_id,
                    stream_id=self._stream_id,
                    generation_id=self._generation_id,
                    sequence_no=self._current_batch_sequence,
                    pcm_bytes=chunk,
                    sample_rate=self._sample_rate,
                    pcm_seq=self._next_pcm_sequence,
                )
            )
            self._next_pcm_sequence += 1
        return tuple(records)

    def cancel(self) -> None:
        self._adapter.cancel()
