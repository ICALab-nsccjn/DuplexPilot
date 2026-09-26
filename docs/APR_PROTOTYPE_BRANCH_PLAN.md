# APR Prototype Branch Plan

## 1. Branch Identity

```text
Branch: apr-prototype
Base: scalability-debug-n4-n8 at the closed N4/N8 diagnostic commit
Purpose: APR diagnostic prototype only
```

The branch is not the paper benchmark branch. It must never rewrite or
overwrite the frozen B5 baseline, pilot-v2 evidence, or N4/N8 root-cause
report.

## 2. Branch Creation Procedure

Run in the remote worktree:

```bash
cd /mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree
git status --short
git switch -c apr-prototype
git rev-parse HEAD | tee reports/apr_prototype_base_commit.txt
```

If the worktree has unrelated uncommitted runtime changes, stop and record the
exact state before creating the branch. Do not use `git reset --hard` or discard
existing evidence.

## 3. Allowed Changes

Only these components may change in the first prototype:

```text
lychee_fd/runtime/apr/*
APR-specific tests and diagnostic runner
the post-extraction adapter call site
APR documentation and manifests
```

The adapter call site may convert an existing sampled acoustic-token result
into an APR message. It may not change the token result, model batch, sampler,
row mapping, request admission, or interruption semantics.

## 4. Forbidden Changes

The branch must not modify:

```text
RSV
DSV
can_pack
physical batch selection
Bmax
request-ID ownership
row-aware forward/logits semantics
StreamingDecoder token interpretation
Token2Wav token interpretation
Official Native behavior
```

It must not add a new model, external baseline, throughput claim, speedup
claim, or formal paper benchmark.

## 5. Commit Sequence

Use small commits with one reversible boundary each:

```text
apr: define acoustic contracts
apr: add versioned acoustic state store
apr: add bounded per-request queues
apr: add acoustic scheduler
apr: add isolated worker adapter
apr: connect post-extraction APR boundary
apr: add APR observability
apr: add diagnostic runner
apr: close N2/N4/N8 diagnostic report
```

Every commit message and `APR_CHANGELOG.md` entry must include:

```yaml
component: ...
reason: ...
semantic_change: true/false
rsv_dsv_changed: false
measurement_boundary_changed: false
```

## 6. Review Gates

### Gate A — Static boundary review

Confirm by diff that APR imports no model scheduler or batch-selection code and
that the only production call site is after token extraction.

### Gate B — Fake-backend correctness

Pass deterministic tests for FIFO, state-version commits, cancellation,
stale-output rejection, PCM ownership, and cleanup.

### Gate C — B5 N=2 regression

Run APR-disabled and APR-enabled N=2 controls. Both must retain existing
ownership and measurement behavior before N=4 is allowed.

### Gate D — N=4/N=8 diagnostic

Use the registered APR evaluation plan. A failure is evidence, not a reason to
change queue capacity, worker count, input timing, or workload during the run.

## 7. Rollback

Rollback means selecting the frozen B5 runtime selector or reverting the APR
branch commit. It does not alter the baseline worktree and does not delete raw
APR evidence.

The runtime selector must support:

```text
APR_DISABLED -> existing B5 path
APR_ENABLED  -> APR post-extraction path
```

The default remains `APR_DISABLED` until the diagnostic report passes all
correctness gates.

## 8. Merge Policy

Do not merge APR into the frozen measurement or paper benchmark branch after a
single successful run. A merge review requires:

1. APR design and implementation plan consistency;
2. N=2 regression evidence;
3. N=4/N=8 diagnostic gate evidence;
4. zero ownership/state-version errors;
5. unchanged measurement contract;
6. an explicit decision on whether APR is a new paper contribution.

Until all six are satisfied, APR remains an experimental architecture branch.
