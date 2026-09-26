"""Build the four requested paper artifacts from frozen, existing results.

The script performs no model or GPU work. It copies the source records into
data/final, computes source-level/paired summaries, and writes LaTeX tables
and plot input files used by the plotting scripts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final"
PLOT_DATA = ROOT / "plot_data"
TABLES = ROOT / "tables"
APPENDIX = ROOT / "appendix"

EVIDENCE = Path(r"E:\DuplexPilot\reports\research_evidence_review_20260925")
PERFORMANCE = Path(r"E:\DuplexPilot\reports\performance_closure_20260920")
INTERACTION = Path(r"E:\DuplexPilot\reports\interaction_external_v2_20260925")
APR = Path(r"E:\DuplexPilot\reports\apr_workload_suite\formal_matrix_v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_source(source: Path, name: str) -> Path:
    target = FINAL / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def write_csv(df: pd.DataFrame, name: str) -> Path:
    target = FINAL / name
    df.to_csv(target, index=False, float_format="%.12g")
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, PLOT_DATA / name)
    return target


def build_main_recovery() -> None:
    source = EVIDENCE / "final_originals" / "PAPER_MAIN_RESULTS.csv"
    if not source.exists():
        source = FINAL / "PAPER_MAIN_RESULTS.csv"
    df = pd.read_csv(source)
    valid = df[df["status"].astype(str).str.startswith("PASS")].copy()
    names = {
        "A-hot": "Hot",
        "F-cold": "Rebuild",
        "DuplexPilot-Optimized": "Joint",
        "KV-offload/acoustic-resident (internal)": "Hybrid",
    }
    rows = []
    for config, label in names.items():
        all_rows = df[df["configuration"] == config]
        v = valid[valid["configuration"] == config]
        row = {"policy": label, "configuration": config,
               "valid": int(len(v)), "scheduled": int(len(all_rows))}
        for env, suffix in [("E1-IAB", "e1"), ("E2-STANDALONE", "e2")]:
            ve = v[v["client_environment"] == env]
            for metric, out in [("render_s", f"t_audio_{suffix}_s"),
                                ("cpu_hold_MiB", "cpu_rss_e2_MiB"),
                                ("gpu1_net_release_MiB", "acoustic_gpu_reclaimed_e2_MiB")]:
                if env == "E2-STANDALONE" or metric == "render_s":
                    by_source = ve.groupby("source")[metric].median()
                    row[out] = float(by_source.median())
        rows.append(row)
    summary = pd.DataFrame(rows)
    write_csv(summary, "main_recovery_summary.csv")
    pair_cols = ["configuration", "source", "repeat", "client_environment",
                 "render_s", "cpu_hold_MiB", "gpu1_net_release_MiB", "status"]
    write_csv(valid[pair_cols].sort_values(
        ["configuration", "source", "repeat", "client_environment"]),
        "main_recovery_pairs.csv")
    tex = r"""\begin{table}[tbp]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \caption{Cold resumption. Latencies are source-level medians in seconds; memory columns are E2 medians in MiB. Valid/all retains every scheduled execution; unvalidated runs do not contribute latency values. GPU reclamation is the net allocated-memory decrease.}
    \label{tab:main_recovery}
    \begin{tabular}{@{}lcrrrr@{}}
        \toprule
        Policy & \shortstack{Valid/\\all} & \shortstack{$T_{\mathrm{audio}}$\\E1} & \shortstack{$T_{\mathrm{audio}}$\\E2} & \shortstack{E2 CPU\\RSS} & \shortstack{E2 acoustic GPU\\reclaimed} \\
        \midrule
"""
    for r in rows:
        tex += (f"        {r['policy']} & {r['valid']}/{r['scheduled']} & "
                f"{r['t_audio_e1_s']:.3f} & {r['t_audio_e2_s']:.3f} & "
                f"{r['cpu_rss_e2_MiB']:.1f} & "
                f"{r['acoustic_gpu_reclaimed_e2_MiB']:.1f} \\\\\n")
    tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "main_recovery.tex").write_text(tex, encoding="utf-8")


def build_throughput() -> None:
    rows = []
    for path in sorted((APR / "runs").glob("*/*/*/attempt-*/formal_primary_metrics.csv")):
        d = pd.read_csv(path)
        if len(d) != 1:
            raise ValueError(f"expected one row in {path}")
        r = d.iloc[0].to_dict()
        rows.append({
            "workload": str(r["workload"]),
            "concurrency": int(r["concurrency"]),
            "repeat": int(r["repeat"]),
            "system": str(r["system"]),
            "elapsed_s": float(r["elapsed_s"]),
            "completed_sessions": int(r["completed_sessions"]),
            "throughput_sps": float(r["session_throughput_sps"]),
            "source_file": str(path.relative_to(APR)),
        })
    runs = pd.DataFrame(rows).sort_values(
        ["workload", "concurrency", "repeat", "system"])
    write_csv(runs, "throughput_runs.csv")
    idx = ["workload", "concurrency", "repeat"]
    apr = runs[runs.system == "apr"].set_index(idx)
    aff = runs[runs.system == "original_affinity"].set_index(idx)
    nomig = runs[runs.system == "apr_no_migration"].set_index(idx)
    pairs = []
    for key in sorted(set(apr.index) & set(aff.index) & set(nomig.index)):
        w, n, rep = key
        a = float(apr.loc[key, "throughput_sps"])
        b = float(aff.loc[key, "throughput_sps"])
        c = float(nomig.loc[key, "throughput_sps"])
        pairs.append({"workload": w, "concurrency": n, "repeat": rep,
                      "apr_throughput_sps": a,
                      "affinity_throughput_sps": b,
                      "no_migration_throughput_sps": c,
                      "ratio_apr_affinity": a / b,
                      "ratio_apr_no_migration": a / c})
    pairs = pd.DataFrame(pairs)
    write_csv(pairs, "throughput_pairs.csv")
    summaries = []
    for w in "ABC":
        for n in [4, 8, 16]:
            q = pairs[(pairs.workload == w) & (pairs.concurrency == n)]
            for series, col in [("APR/Affinity", "ratio_apr_affinity"),
                                ("APR/No-migration", "ratio_apr_no_migration")]:
                summaries.append({"workload": w, "concurrency": n,
                                  "series": series, "n": int(q[col].count()),
                                  "mean_ratio": float(q[col].mean()),
                                  "median_ratio": float(q[col].median()),
                                  "min_ratio": float(q[col].min()),
                                  "max_ratio": float(q[col].max())})
    write_csv(pd.DataFrame(summaries), "throughput_summary.csv")


def build_transfer() -> None:
    raw = pd.read_csv(PERFORMANCE / "PAIRED.csv")
    cols = ["source", "repeat", "L_sync_attempt", "L_best_attempt",
            "payload_equal", "full_suffix_exact", "render_s_sync",
            "render_s_best", "render_s_delta_best_minus_sync",
            "render_s_reduction_pct"]
    out = raw[cols].copy().rename(columns={
        "render_s_sync": "transfer_ready_s_sync",
        "render_s_best": "transfer_ready_s_optimized",
        "render_s_delta_best_minus_sync":
            "transfer_ready_s_delta_optimized_minus_sync",
        "render_s_reduction_pct": "transfer_ready_reduction_pct",
    })
    write_csv(out.sort_values(["source", "repeat"]), "transfer_pairs.csv")


def build_interaction() -> None:
    raw = pd.read_csv(INTERACTION / "INTERACTION_METRICS.csv")
    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "path": "Shared Lychee / DuplexPilot online" if r["config"] == "lychee"
                    else "vLLM-Omni / MiniCPM-o4.5",
            "config": r["config"],
            "srr": f"{int(r.SRR_k)}/{int(r.SRR_n)}",
            "sir": f"{int(r.SIR_k)}/{int(r.SIR_n)}",
            "eir": f"{int(r.EIR_k)}/{int(r.EIR_n)}",
            "srir": f"{int(r.SRIR_k)}/{int(r.SRIR_n)}",
            "fsed_median_ms": float(r.FSED_median),
            "fsed_n": int(r.FSED_n),
            "fsed_min_ms": float(r.FSED_min),
            "fsed_max_ms": float(r.FSED_max),
            "ird_median_ms": float(r.IRD_median),
            "ird_n": int(r.IRD_n),
            "ird_min_ms": float(r.IRD_min),
            "ird_max_ms": float(r.IRD_max),
        })
    write_csv(pd.DataFrame(rows), "interaction_summary.csv")
    tex = r"""\begin{table}[tbp]
    \centering
    \small
    \setlength{\tabcolsep}{3.5pt}
    \caption{Natural-interaction pilot, six sources per path. Rates are $k/n$; delays are playback-layer medians in milliseconds, followed by valid event counts. These are adapted pilot results, not official benchmark scores.}
    \label{tab:interaction_metrics}
    \begin{tabular}{@{}lccccrr@{}}
        \toprule
        Path & SRR & SIR & EIR & SRIR & \shortstack{FSED\\playout} & \shortstack{IRD\\playout} \\
        \midrule
        \shortstack[l]{Shared Lychee/\\DuplexPilot online} & 7/7 & 3/3 & 4/10 & 3/3 & 4445.9 (10) & 278.2 (3) \\
        \shortstack[l]{vLLM-Omni/\\MiniCPM-o 4.5} & 6/7 & 3/3 & 9/10 & 3/3 & 635.5 (9) & 2068.3 (3) \\
        \bottomrule
    \end{tabular}
\end{table}
"""
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "interaction_metrics.tex").write_text(tex, encoding="utf-8")
    APPENDIX.mkdir(parents=True, exist_ok=True)
    appendix = r"""\paragraph{Interaction timing ranges and event definitions.}
The interaction pilot uses signed playback-layer timings. FSED is measured from
user-utterance end to reply playback onset; IRD is measured from interruption
onset to the end of preceding assistant speech. The reported median (valid $n$)
and observed range are: shared Lychee/DuplexPilot FSED $4445.9$ ms ($n=10$),
$[-141.8,7372.9]$; IRD $278.2$ ms ($n=3$), $[266.6,898.9]$; vLLM-Omni /
MiniCPM-o4.5 FSED $635.5$ ms ($n=9$), $[-224.1,3238.9]$; IRD $2068.3$ ms
($n=3$), $[564.1,3444.6]$. SRR, SIR, EIR, and SRIR retain their event
denominators as shown in Table~\ref{tab:interaction_metrics}.
"""
    (APPENDIX / "interaction_ranges_and_event_definitions.tex").write_text(
        appendix, encoding="utf-8")


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    source_map = {
        "PAPER_MAIN_RESULTS.csv": EVIDENCE / "final_originals" / "PAPER_MAIN_RESULTS.csv",
        "RUN_AUDIT.csv": EVIDENCE / "RUN_AUDIT.csv",
        "PAIRED_RECOMPUTED.csv": EVIDENCE / "PAIRED_RECOMPUTED.csv",
        "METRIC_COVERAGE.csv": EVIDENCE / "METRIC_COVERAGE.csv",
        "performance_PAIRED.csv": PERFORMANCE / "PAIRED.csv",
        "interaction_metrics_source.csv": INTERACTION / "INTERACTION_METRICS.csv",
        "interaction_event_scores_source.csv": INTERACTION / "EVENT_SCORES.csv",
        "interaction_episodes_source.json": INTERACTION / "INTERACTION_EPISODES.json",
        "common_exposure_source.json": INTERACTION / "COMMON_EXPOSURE.json",
        "source_pacing_audit_source.json": INTERACTION / "SOURCE_PACING_AUDIT.json",
        "native_interaction_markers_source.json": INTERACTION / "NATIVE_INTERACTION_MARKERS.json",
        "formal_matrix_suite_manifest.json": APR / "suite_manifest.json",
    }
    manifest = {"generated_by": "scripts/build_final_artifacts.py",
                "sources": {}, "generated": {}}
    for name, path in source_map.items():
        if path.exists():
            target = copy_source(path, name)
            manifest["sources"][name] = {"source": str(path),
                                         "sha256": sha256(target)}
    build_main_recovery()
    build_throughput()
    build_transfer()
    build_interaction()
    for path in sorted(FINAL.iterdir()):
        if path.name not in manifest["sources"]:
            manifest["generated"][path.name] = {"sha256": sha256(path)}
    (FINAL / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

