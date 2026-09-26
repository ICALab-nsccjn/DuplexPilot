import json
from pathlib import Path

from profiling.model_execution_plane.oracle import (
    ServiceCurve,
    _ready_times,
    derive_feedback_delays,
    e2e_intervals_from_trace,
    normalize_model_trace,
    simulate_central_driver,
)

root = Path('/mnt/DuplexPilot/data/DuplexPilot/results/apr_row_aware_model_plane_e2e_20260831')
curve = ServiceCurve(25_860_183, 30_717_779.5)
for label in ('n2', 'n8'):
    if label == 'n2':
        model_path = root / 'phase0_model_trace_n2_rerun/model_execution_trace.jsonl'
        acoustic_path = root / 'phase0_model_trace_n2_rerun/acoustic_trace.jsonl'
    else:
        model_path = root / 'phase0_model_trace_n8/model_execution_trace.jsonl'
        acoustic_path = root / 'phase0_model_trace_n8/acoustic_trace.jsonl'
    records = [json.loads(line) for line in model_path.open(encoding='utf-8')]
    acoustic = [json.loads(line) for line in acoustic_path.open(encoding='utf-8')]
    obs = normalize_model_trace(records)
    ready = _ready_times(obs, records)
    base = simulate_central_driver(obs, curve, ready_times_ns=ready)
    print(label, 'steps', len(obs), 'base_span_ms', round(base.makespan_ns / 1e6, 3))
    for wait_ms in (0, .5, 1, 2, 5, 10, 20, 25, 40):
        result = simulate_central_driver(obs, curve, ready_times_ns=ready, wait_budget_ns=int(wait_ms * 1e6))
        print('wait_ms', wait_ms, 'pairs', result.batch2_count, 'pair_fraction', round(result.batch2_count / len(result.batch_details), 5), 'speedup', round(base.makespan_ns / result.makespan_ns, 5), 'makespan_ms', round(result.makespan_ns / 1e6, 3))
    feedback = simulate_central_driver(obs, curve, ready_times_ns=ready, feedback_delays_ns=derive_feedback_delays(obs, records))
    print('feedback', 'pairs', feedback.batch2_count, 'speedup', round(base.makespan_ns / feedback.makespan_ns, 5), 'makespan_ms', round(feedback.makespan_ns / 1e6, 3))
    print('e2e intervals', len(e2e_intervals_from_trace(acoustic)))
