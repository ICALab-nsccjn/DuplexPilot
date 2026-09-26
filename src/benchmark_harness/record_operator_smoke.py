from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


WT = Path("/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-aware-flow-operator-e2e")
RESULTS = Path("/mnt/DuplexPilot/data/DuplexPilot/results/apr_aware_flow_operator_e2e_20260831")


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    report = """# APR Operator B=1 Branch-Integrity Smoke

## Result

`PASS`: one real A100 test completed for the frozen B=1 public Flow step,
frozen control equivalence, mel/cache contract, PCM contract, and cleanup.

Command was run inside `duplexpilot` with `CUDA_VISIBLE_DEVICES=0,1`,
`LYCHEEFD_TOKEN2WAV_DEVICE=1`, Flow cache capacity 2048, float32, and the
production Token2Wav checkpoint. The test passed in 12.12 seconds:

```text
1 passed, 8 warnings
```

The warnings are PyTorch/CuBLAS deterministic-operation warnings and an
audio-backend warning; they did not fail the contract test. No vLLM service
was started by this smoke, and it is not an online throughput measurement.
"""
    for target in (RESULTS / "APR_OPERATOR_B1_INTEGRITY_SMOKE.md", WT / "APR_OPERATOR_B1_INTEGRITY_SMOKE.md"):
        target.write_text(report, encoding="utf-8")

    manifest_path = WT / "APR_OPERATOR_SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["updated_utc"] = now
    manifest["b1_branch_integrity_smoke"] = {
        "status": "PASS",
        "test": "tests/test_flow_batch_a100_equivalence.py::test_b1_public_step_matches_frozen_control_and_pcm_contract",
        "result": "1 passed in 12.12s",
        "container": "duplexpilot",
        "cuda_visible_devices": "0,1",
        "token2wav_device": 1,
        "flow_cache_capacity": 2048,
        "dtype": "float32",
        "online_service": "not started",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(manifest_path, RESULTS / manifest_path.name)

    env_path = WT / "APR_OPERATOR_ENVIRONMENT_MANIFEST.json"
    env = json.loads(env_path.read_text(encoding="utf-8"))
    env["updated_utc"] = now
    env["b1_branch_integrity_smoke"] = "PASS: 1 real A100 contract test"
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(env_path, RESULTS / env_path.name)


if __name__ == "__main__":
    main()
