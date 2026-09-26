# Real Audio Equivalence Proposal

## Why a signal-level contract is needed

The current CUDA acoustic backend contains stochastic Flow/HiFiGAN operations
and a `cumsum_cuda_kernel` without a deterministic implementation. Requiring
bitwise PCM equality from this backend would conflate numerical backend
reproducibility with logical state migration. A test-only CPU reference backend
can pass exact checkpoint equivalence, but it is not the production acoustic
implementation.

## Proposed production contract

The production backend must first satisfy exact Level 0 and Level 1 contracts:

```text
complete cache/state identity
complete RNG continuation state
event/flush/chunk ordering
cancel and stale-output rejection
```

After that, Level 2 uses the fixed signal-level conditions from
`ACOUSTIC_EQUIVALENCE_CONTRACT_V2.md`:

```text
same sample rate
same non-empty sample count
finite PCM
normalized RMSE <= 0.02
correlation >= 0.99
SNR >= 34 dB
```

## Scientific justification

For streaming serving, the externally relevant semantic obligations are that
the same logical generation produces the same ordered, owned, cancelable PCM
chunks with the same timing boundaries. Small floating-point differences do not
necessarily violate those obligations. Exact event/chunk checks protect the
observable stream semantics; fixed signal metrics bound numerical drift without
claiming perceptual equivalence or hiding ownership errors.

The RNG requirement is essential. The diagnostic produced bitwise equality when
the same CPU/CUDA RNG snapshot was restored and different PCM when it was not.
Thus signal-level comparison is acceptable only after stochastic continuation
state is explicit and reproducible.

## Status

This is a proposal, not an accepted production capability. The current local
checkpoint API does not yet export RNG state, and the production remote service
does not export any continuation state. APR therefore remains blocked.
