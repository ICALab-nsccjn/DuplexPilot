"""Validate the endpoint vocabulary and paired statistics used by the paper."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def median(values):
    return statistics.median(float(v) for v in values)


def main() -> None:
    run_rows = rows(ROOT / "data" / "final" / "RUN_AUDIT.csv")
    assert len(run_rows) == 96
    valid = [r for r in run_rows if r["status"].startswith("PASS")]
    assert len(valid) == 92
    required = {"save_s", "ready_s", "receive_s", "render_s", "gap_s"}
    assert required <= set(run_rows[0])

    # Every published render/gap endpoint is populated on the same valid rows;
    # missing values are not silently converted to zero.
    for r in valid:
        for key in required:
            assert r[key] != ""
        assert float(r["render_s"]) >= float(r["gap_s"])

    # Source-level aggregate values are the values printed in the appendix.
    aggregates = rows(ROOT / "data" / "final_report_aggregates.csv")
    expected = {
        ("E1", "F-cold"): (1.847, 13.317, 14.451, 13.616),
        ("E1", "Joint"): (1.387, 0.699, 1.899, 1.064),
        ("E2", "F-cold"): (1.926, 13.250, 14.400, 13.565),
        ("E2", "Joint"): (1.387, 0.700, 1.872, 1.037),
    }
    for r in aggregates:
        key = (r["environment"], r["strategy"])
        if key in expected:
            values = tuple(float(r[k]) for k in ("save_s", "ready_s", "render_s", "gap_s"))
            assert all(abs(a - b) < 1e-9 for a, b in zip(values, expected[key]))

    paired = rows(ROOT / "data" / "final" / "PAIRED_RECOMPUTED.csv")
    e2 = next(r for r in paired
              if r["environment"] == "E2-STANDALONE"
              and r["candidate"] == "L-best"
              and r["reference"] == "F-cold"
              and r["metric"] == "render_s")
    assert int(e2["evaluable_pairs"]) == 13
    assert int(e2["source_groups"]) == 12
    assert int(e2["n"]) == 12
    assert abs(float(e2["median"]) + 12.506666666666668) < 1e-9

    transfer = rows(ROOT / "data" / "final" / "transfer_pairs.csv")
    assert len(transfer) == 12
    assert {"transfer_ready_s_sync", "transfer_ready_s_optimized",
            "transfer_ready_reduction_pct"} <= set(transfer[0])
    assert all(float(r["transfer_ready_s_optimized"]) < float(r["transfer_ready_s_sync"])
               for r in transfer)
    absolute = median(float(r["transfer_ready_s_sync"])
                      - float(r["transfer_ready_s_optimized"]) for r in transfer)
    relative = median(float(r["transfer_ready_reduction_pct"]) for r in transfer)
    assert abs(absolute - 0.533333333335) < 1e-9
    assert abs(relative - 15.99975414935) < 1e-9

    print("measurement endpoint audit: PASS")
    print("cold E2 Joint-minus-Rebuild paired render median: -12.507 s (13/13; 12 source groups)")
    print(f"transfer-to-ready paired median: {absolute:.3f} s, {relative:.2f}%")


if __name__ == "__main__":
    main()
