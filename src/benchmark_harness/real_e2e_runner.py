"""Real-model to local acoustic handoff for APR bring-up.

The adapter deliberately owns only the public model-runner result boundary.
It does not inspect decoder, Token2Wav, vLLM, or engine private state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import time
import uuid
from typing import Any

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from lychee_fd.runtime.apr.profiling import StageProfiler


LYCHEE_AUDIO_TOKEN_START = 151696
TOKEN2WAV_CODEC_VOCAB_SIZE = 6561
TOKEN2WAV_CODEC_TOKEN_END = (
    LYCHEE_AUDIO_TOKEN_START + TOKEN2WAV_CODEC_VOCAB_SIZE
)



class RealModelAcousticHandoffError(ValueError):
    """Raised when a model-to-acoustic handoff cannot be proven safe."""


class RealModelAcousticHandoff:
    """Convert one real model step into request-owned acoustic messages."""

    def __init__(
        self,
        model_runner: Any,
        *,
        stream_ids: Mapping[str, str] | None = None,
        generation_ids: Mapping[str, int] | None = None,
        sequence_numbers: Mapping[str, int] | None = None,
        state_versions: Mapping[str, int] | None = None,
        acoustic_chunk_size: int = 1,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not hasattr(model_runner, "run_plan") or not callable(model_runner.run_plan):
            raise RealModelAcousticHandoffError(
                "model_runner must expose run_plan"
            )
        self._model_runner = model_runner
        if (
            isinstance(acoustic_chunk_size, bool)
            or not isinstance(acoustic_chunk_size, int)
            or acoustic_chunk_size <= 0
        ):
            raise RealModelAcousticHandoffError(
                "acoustic_chunk_size must be a positive integer"
            )
        self._acoustic_chunk_size = acoustic_chunk_size
        if profiler is not None and not isinstance(profiler, StageProfiler):
            raise RealModelAcousticHandoffError("profiler must be a StageProfiler")
        self._profiler = profiler
        self._stream_ids = dict(stream_ids or {})
        self._generation_ids = dict(generation_ids or {})
        self._sequence_numbers = {
            str(key): int(value) for key, value in (sequence_numbers or {}).items()
        }
        self._state_versions = {
            str(key): int(value) for key, value in (state_versions or {}).items()
        }
        self._pending_tokens: dict[str, list[int]] = {}
        self.last_model_result: Mapping[str, Any] | None = None
        self.last_output_summary: tuple[dict[str, int | str], ...] = ()

    def _profile_span(
        self,
        stage: str,
        *,
        session_id: str | None = None,
        generation_id: int | None = None,
        sequence_no: int | None = None,
        state_version: int | None = None,
    ):
        if self._profiler is None:
            return nullcontext()
        return self._profiler.span(
            stage,
            session_id=session_id,
            generation_id=generation_id,
            sequence_no=sequence_no,
            state_version=state_version,
        )

    @staticmethod
    def _require_nonnegative_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RealModelAcousticHandoffError(
                f"{name} must be a non-negative integer"
            )
        return int(value)

    def _stream_id(self, request_id: str) -> str:
        stream_id = self._stream_ids.get(request_id, f"stream-{request_id}")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise RealModelAcousticHandoffError(
                f"stream_id is missing for request {request_id}"
            )
        return stream_id

    def _tokens(self, request_id: str, output: Mapping[str, Any]) -> tuple[int, ...]:
        if "stoken_ids" in output:
            values = output["stoken_ids"]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise RealModelAcousticHandoffError(
                    f"stoken_ids is not a sequence for request {request_id}"
                )
            tokens = tuple(values)
        elif "stoken" in output:
            tokens = (output["stoken"],)
        else:
            raise RealModelAcousticHandoffError(
                f"model output has no stoken output for request {request_id}"
            )
        if not tokens or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in tokens
        ):
            raise RealModelAcousticHandoffError(
                f"stoken output is invalid for request {request_id}"
            )
        normalized = []
        for token in tokens:
            if 0 <= token < TOKEN2WAV_CODEC_VOCAB_SIZE:
                normalized.append(int(token))
            elif LYCHEE_AUDIO_TOKEN_START <= token < TOKEN2WAV_CODEC_TOKEN_END:
                normalized.append(int(token - LYCHEE_AUDIO_TOKEN_START))
        return tuple(normalized)

    def _output_to_batch(
        self,
        output: Mapping[str, Any],
        *,
        source_execution_id: str,
        created_monotonic_ns: int,
    ) -> AcousticTokenBatch:
        if not isinstance(output, Mapping):
            raise RealModelAcousticHandoffError("model output must be a mapping")
        request_id = str(output.get("request_id") or "")
        if not request_id:
            raise RealModelAcousticHandoffError(
                "model output is missing request_id"
            )
        generation_id = self._require_nonnegative_int(
            "generation_id",
            output.get("generation_id", self._generation_ids.get(request_id, 0)),
        )
        sequence_no = self._require_nonnegative_int(
            "sequence_no",
            output.get("sequence_no", self._sequence_numbers.get(request_id, 0)),
        )
        state_version = self._require_nonnegative_int(
            "state_version",
            output.get("state_version", self._state_versions.get(request_id, 0)),
        )
        output_source = str(output.get("source_execution_id") or source_execution_id)
        if not output_source:
            raise RealModelAcousticHandoffError(
                f"source execution identity is missing for request {request_id}"
            )
        return AcousticTokenBatch(
            request_id=request_id,
            stream_id=self._stream_id(request_id),
            generation_id=generation_id,
            sequence_no=sequence_no,
            stoken_ids=self._tokens(request_id, output),
            source_execution_id=output_source,
            state_version=state_version,
            created_monotonic_ns=created_monotonic_ns,
        )

    def run_plan(self, store: Any, plan: Any) -> tuple[AcousticTokenBatch, ...]:
        """Run exactly one real model step and return row-aligned handoff data."""
        with self._profile_span("MODEL_FORWARD"):
            result = self._model_runner.run_plan(
                store,
                plan,
                decode_steps=1,
                preserve_state=True,
            )
        self.last_model_result = result
        if not isinstance(result, Mapping):
            raise RealModelAcousticHandoffError(
                "real model runner result must be a mapping"
            )
        outputs = result.get("outputs")
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
            raise RealModelAcousticHandoffError(
                "real model runner result has no output sequence"
            )
        expected = tuple(str(request_id) for request_id in plan.row_to_request)
        actual = tuple(str(output.get("request_id") or "") for output in outputs)
        if actual != expected:
            raise RealModelAcousticHandoffError(
                f"model output rows diverged from plan: expected={expected} got={actual}"
            )
        source_execution_id = (
            f"real-model-{uuid.uuid4().hex}"
        )
        created_monotonic_ns = time.monotonic_ns()
        batches: list[AcousticTokenBatch] = []
        output_summary: list[dict[str, int | str]] = []
        for output in outputs:
            request_id = str(output["request_id"])
            generation_id = self._require_nonnegative_int(
                "generation_id",
                output.get("generation_id", self._generation_ids.get(request_id, 0)),
            )
            sequence_no = self._require_nonnegative_int(
                "sequence_no",
                output.get("sequence_no", self._sequence_numbers.get(request_id, 0)),
            )
            state_version = self._require_nonnegative_int(
                "state_version",
                output.get("state_version", self._state_versions.get(request_id, 0)),
            )
            with self._profile_span(
                "TOKEN_TRANSFER",
                session_id=request_id,
                generation_id=generation_id,
                sequence_no=sequence_no,
                state_version=state_version,
            ):
                normalized_tokens = self._tokens(request_id, output)
                output_summary.append(
                    {
                        "request_id": request_id,
                        "raw_token_count": (
                            len(output["stoken_ids"])
                            if isinstance(output.get("stoken_ids"), Sequence)
                            and not isinstance(output.get("stoken_ids"), (str, bytes))
                            else 1
                        ),
                        "normalized_token_count": len(normalized_tokens),
                    }
                )
                pending = self._pending_tokens.setdefault(request_id, [])
                pending.extend(normalized_tokens)
                if len(pending) < self._acoustic_chunk_size:
                    continue
                chunk = tuple(pending[:self._acoustic_chunk_size])
                del pending[:self._acoustic_chunk_size]
                aggregated = dict(output)
                aggregated["stoken_ids"] = chunk
                batches.append(
                    self._output_to_batch(
                        aggregated,
                        source_execution_id=source_execution_id,
                        created_monotonic_ns=created_monotonic_ns,
                    )
                )
        self.last_output_summary = tuple(output_summary)
        return tuple(batches)

    def flush_pending(self) -> tuple[AcousticTokenBatch, ...]:
        """Emit remaining request-owned tokens as final diagnostic chunks."""
        batches: list[AcousticTokenBatch] = []
        created_monotonic_ns = time.monotonic_ns()
        for request_id, pending in tuple(self._pending_tokens.items()):
            if not pending:
                continue
            aggregated = {
                "request_id": request_id,
                "stoken_ids": tuple(pending),
                "generation_id": self._generation_ids.get(request_id, 0),
                "sequence_no": self._sequence_numbers.get(request_id, 0),
                "state_version": self._state_versions.get(request_id, 0),
            }
            batches.append(
                self._output_to_batch(
                    aggregated,
                    source_execution_id=f"real-model-flush-{uuid.uuid4().hex}",
                    created_monotonic_ns=created_monotonic_ns,
                )
            )
            pending.clear()
        return tuple(batches)

    def acknowledge(
        self,
        batch: AcousticTokenBatch,
        *,
        committed_state_version: int,
    ) -> None:
        """Advance the adapter's public handoff cursor after acoustic commit."""
        if not isinstance(batch, AcousticTokenBatch):
            raise RealModelAcousticHandoffError(
                "acknowledge requires an AcousticTokenBatch"
            )
        next_version = self._require_nonnegative_int(
            "committed_state_version", committed_state_version
        )
        self._state_versions[batch.request_id] = next_version
        self._sequence_numbers[batch.request_id] = batch.sequence_no + 1
