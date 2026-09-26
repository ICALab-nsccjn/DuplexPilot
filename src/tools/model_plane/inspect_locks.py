import collections
import json
from pathlib import Path

from profiling.model_execution_plane.oracle import normalize_model_trace, _ready_times

root = Path('/mnt/DuplexPilot/data/DuplexPilot/results/apr_row_aware_model_plane_e2e_20260831')
records = [json.loads(line) for line in (root / 'phase0_model_trace_n8/model_execution_trace.jsonl').open()]
observations = normalize_model_trace(records)
ready_by_oracle = _ready_times(observations, records)
locks = [row for row in records if row.get('event_type') == 'MODEL_LOCK' and row.get('request_ids')]
by = collections.defaultdict(list)
for row in locks:
    by[str(row['request_ids'][0])].append(row)
for request_id in [observations[0].request_ids[0]]:
    print('request', request_id)
    rows = by[request_id]
    index = 0
    count = 0
    previous_attempt = None
    for observation in observations:
        if observation.request_ids[0] != request_id:
            continue
        while index < len(rows) and int(rows[index]['timestamp_monotonic_ns']) < observation.timestamp_ns:
            index += 1
        if index >= len(rows):
            break
        row = rows[index]
        index += 1
        attempt = int(row['timestamp_monotonic_ns']) - int(row.get('hold_ns', 0)) - int(row.get('wait_ns', 0))
        print(count, 'step_end', round(observation.timestamp_ns / 1e6, 3), 'release', round(int(row['timestamp_monotonic_ns']) / 1e6, 3), 'wait', round(int(row.get('wait_ns', 0)) / 1e6, 3), 'hold', round(int(row.get('hold_ns', 0)) / 1e6, 3), 'attempt', round(attempt / 1e6, 3), 'delta', None if previous_attempt is None else round((attempt - previous_attempt) / 1e6, 3))
        previous_attempt = attempt
        count += 1
        if count >= 50:
            break
print('oracle_ready_deltas_for_first_request')
previous = None
count = 0
for observation, ready in zip(observations, ready_by_oracle):
    if observation.request_ids[0] != observations[0].request_ids[0]:
        continue
    if previous is not None:
        print(count, round((ready - previous) / 1e6, 3))
    previous = ready
    count += 1
    if count >= 50:
        break
