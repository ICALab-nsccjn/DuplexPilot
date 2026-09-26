# Acoustic Progression Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diagnostic APR prototype after token extraction that can decouple acoustic progression while preserving the frozen RSV/DSV model path.

**Architecture:** APR consists of immutable token-batch contracts, per-request bounded queues, a versioned logical acoustic state store, and a bounded round-robin worker pool. The model scheduler and physical batch path remain unchanged.

**Tech Stack:** Existing Python runtime, `queue`, `threading`, dataclasses, JSONL traces, unittest, pinned `fdmodel-demo` environment inside `duplexpilot`.

---

## Guardrails Before Implementation

- [ ] Create the isolated branch in the remote worktree:

  ```bash
  cd /mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree
  git switch -c apr-prototype
  git rev-parse HEAD > reports/apr_prototype_base_commit.txt
  ```

- [ ] Record that `RSV`, `DSV`, `can_pack`, `Bmax`, physical row mapping,
  scheduler selection, sampler, and request ownership are read-only for this
  branch.

- [ ] Run the existing B5 W1 N=2 control before changing code and store the
  manifest under `reports/apr_prototype/baseline_n2_control/`.

  Expected gate: 3/3 clean, ownership mismatch 0, measurement schema unchanged.

## Task 1: Freeze the APR Message Contract

**Files:**
- Create: `lychee_fd/runtime/apr/contracts.py`
- Test: `tests/test_apr_contracts.py`

- [ ] Write tests for immutable token batches, request/generation identity,
  monotonic sequence numbers, and explicit rejection of missing IDs or invalid
  sequences.

- [ ] Implement these exact records:

  ```python
  @dataclass(frozen=True)
  class AcousticTokenBatch:
      request_id: str
      stream_id: str
      generation_id: int
      sequence_no: int
      stoken_ids: tuple[int, ...]
      source_execution_id: str
      state_version: int
      created_monotonic_ns: int

  @dataclass(frozen=True)
  class AcousticPcmRecord:
      request_id: str
      stream_id: str
      generation_id: int
      sequence_no: int
      pcm_bytes: bytes
      sample_rate: int
      pcm_seq: int
  ```

- [ ] Run inside the container:

  ```bash
  python -m unittest -v tests.test_apr_contracts
  ```

  Expected: all contract tests pass.

- [ ] Commit only the contract and tests:

  ```bash
  git add lychee_fd/runtime/apr/contracts.py tests/test_apr_contracts.py
  git commit -m "apr: define acoustic token and PCM contracts"
  ```

## Task 2: Add Versioned Logical Acoustic State

**Files:**
- Create: `lychee_fd/runtime/apr/state_store.py`
- Test: `tests/test_apr_state_store.py`

- [ ] Write tests for acquire/commit, stale lease rejection, generation
  cancellation, and release after terminal cleanup.

- [ ] Implement `AcousticStateStore` with these methods:

  ```python
  acquire(request_id: str, generation_id: int, expected_version: int)
  commit(lease, next_state: dict, pcm_records: list)
  cancel(request_id: str, generation_id: int)
  release(request_id: str)
  snapshot(request_id: str) -> dict
  ```

- [ ] Enforce that only one lease exists for a request, committed versions are
  exactly `previous_version + 1`, and cancelled generations cannot commit PCM.

- [ ] Run:

  ```bash
  python -m unittest -v tests.test_apr_state_store
  ```

- [ ] Commit:

  ```bash
  git add lychee_fd/runtime/apr/state_store.py tests/test_apr_state_store.py
  git commit -m "apr: add versioned logical acoustic state store"
  ```

## Task 3: Implement Per-Request Acoustic Queues

**Files:**
- Create: `lychee_fd/runtime/apr/queue.py`
- Test: `tests/test_apr_queue.py`

- [ ] Write tests for FIFO order, duplicate/gap rejection, bounded capacity,
  explicit backpressure, and generation cancellation.

- [ ] Implement `AcousticIngressQueue` with capacity 32 and methods:

  ```python
  put(batch: AcousticTokenBatch) -> None
  get() -> AcousticTokenBatch | None
  cancel_generation(generation_id: int) -> int
  depth() -> int
  close() -> None
  ```

- [ ] Make overflow raise `APRBackpressure` rather than dropping a batch.

- [ ] Run the queue tests and commit:

  ```bash
  python -m unittest -v tests.test_apr_queue
  git add lychee_fd/runtime/apr/queue.py tests/test_apr_queue.py
  git commit -m "apr: add bounded per-request acoustic ingress queues"
  ```

## Task 4: Implement Diagnostic Acoustic Scheduler

**Files:**
- Create: `lychee_fd/runtime/apr/scheduler.py`
- Test: `tests/test_apr_scheduler.py`

- [ ] Write tests showing round-robin selection across ready request IDs,
  per-request FIFO preservation, cancellation removal, and no duplicate worker
  lease.

- [ ] Implement a scheduler with configurable worker count and strict selection:

  ```python
  class AcousticScheduler:
      def register(self, request_id: str, queue: AcousticIngressQueue) -> None
      def unregister(self, request_id: str) -> None
      def next_ready(self) -> str | None
      def mark_ready(self, request_id: str) -> None
  ```

- [ ] Do not import or call the model scheduler from APR scheduler code.

- [ ] Run:

  ```bash
  python -m unittest -v tests.test_apr_scheduler
  ```

- [ ] Commit:

  ```bash
  git add lychee_fd/runtime/apr/scheduler.py tests/test_apr_scheduler.py
  git commit -m "apr: add round-robin acoustic scheduler"
  ```

## Task 5: Add One Worker-Slot Adapter

**Files:**
- Create: `lychee_fd/runtime/apr/worker.py`
- Test: `tests/test_apr_worker.py`

- [ ] Write tests using a fake decoder/Token2Wav adapter. Verify state restore,
  one token transition, atomic PCM commit, stale generation rejection, and
  worker failure isolation.

- [ ] Define the adapter interface:

  ```python
  class AcousticBackend:
      def restore(self, state: dict) -> None: ...
      def process(self, tokens: tuple[int, ...]) -> tuple[dict, list[AcousticPcmRecord]]: ...
      def capture(self) -> dict: ...
  ```

- [ ] Implement `AcousticWorker` so one task follows:

  ```text
  acquire -> restore -> process -> capture -> commit -> release
  ```

- [ ] Emit `APR_ACOUSTIC_START`, `APR_ACOUSTIC_END`, and `APR_ERROR` events
  without changing PCM contents.

- [ ] Run:

  ```bash
  python -m unittest -v tests.test_apr_worker
  ```

- [ ] Commit:

  ```bash
  git add lychee_fd/runtime/apr/worker.py tests/test_apr_worker.py
  git commit -m "apr: add isolated acoustic worker slot"
  ```

## Task 6: Integrate Only at the Post-Extraction Boundary

**Files:**
- Create: `lychee_fd/runtime/apr/runtime.py`
- Modify: `lychee_fd/app.py` at the existing token-to-acoustic handoff only
- Test: `tests/test_apr_runtime_integration.py`

- [ ] Write a fake-runtime integration test proving that the model path emits
  an `AcousticTokenBatch` and returns without invoking the acoustic backend
  synchronously.

- [ ] Implement `AprRuntime.enqueue_token_batch(...)` and
  `AprRuntime.shutdown_request(...)`.

- [ ] Keep the existing direct path behind an explicit runtime selector:

  ```text
  APR disabled -> existing B5 path byte-for-byte
  APR enabled  -> token extraction -> APR enqueue
  ```

- [ ] Do not alter `engine.step()`, row-aware forward, sampler lookup,
  `can_pack`, batch selection, Bmax, or interruption semantics.

- [ ] Run the fake integration tests and the existing ownership/unit suite
  before any real server run.

- [ ] Commit:

  ```bash
  git add lychee_fd/runtime/apr/runtime.py lychee_fd/app.py tests/test_apr_runtime_integration.py
  git commit -m "apr: connect post-extraction acoustic runtime boundary"
  ```

## Task 7: Add Observability and Fail-Closed Validation

**Files:**
- Create: `lychee_fd/runtime/apr/trace.py`
- Create: `tests/test_apr_trace.py`
- Modify: `tools/scalability_debug/normalize_lifecycle.py` only to consume APR events

- [ ] Test monotonic timestamps, request-ID preservation, queue depth, waiting
  time, scheduling delay, PCM delay, stale output, backpressure, and cleanup.

- [ ] Emit the event schema defined in `APR_ARCHITECTURE.md` as JSONL without
  sampling or synchronizing CUDA inside model execution.

- [ ] Extend the offline normalizer so missing APR stages are represented as
  `NOT_OBSERVABLE`, never fabricated.

- [ ] Run:

  ```bash
  python -m unittest -v tests.test_apr_trace tests.test_scalability_lifecycle_normalizer
  ```

- [ ] Commit the observability changes.

## Task 8: Run the Diagnostic APR Gate

**Files:**
- Create: `tools/apr/run_apr_diagnostic.py`
- Create: `reports/apr_prototype/`
- Test: `tests/test_apr_diagnostic_runner.py`

- [ ] Add one B5+APR W1 N=2 control with APR disabled and one with APR enabled.

- [ ] Run APR enabled with worker counts `P=1` and `P=2`, keeping W1 timing,
  input assets, Bmax, `MAX_NUM_SEQS`, and measurement boundary unchanged.

- [ ] Run the diagnostic matrix:

  ```text
  B5 + APR × W1 × N=2/N=4/N=8 × P=1/P=2
  3 clean attempts per cell
  ```

- [ ] Mark an attempt invalid for missing done, missing PCM, request-ID error,
  state-version error, stale commit, queue overflow without explicit fail
  closed handling, cleanup timeout, or measurement regression.

- [ ] Stop immediately on any ownership mismatch or PCM owner mismatch.

- [ ] Commit only runner/docs/tests after the diagnostic artifacts are closed.

## Task 9: Review and Handoff

- [ ] Generate `APR_DIAGNOSTIC_REPORT.md` with separate sections for P=1 and
  P=2, and explicitly distinguish decoupling evidence from scalability evidence.

- [ ] If N=4 is not 5/5 clean, do not tune APR silently. Classify the failure
  as queue, state restore, worker capacity, Token2Wav safety, or serving
  boundary incompatibility and stop.

- [ ] Run the complete relevant unit suite inside `duplexpilot`.

- [ ] Keep the frozen B5 report and raw traces immutable; APR results are a new
  diagnostic lane and cannot be merged into prior performance statistics.
