import collections
import json
from pathlib import Path
import statistics

from profiling.model_execution_plane.oracle import normalize_model_trace


root = Path('/mnt/DuplexPilot/data/DuplexPilot/results/apr_row_aware_model_plane_e2e_20260831')
trace = root / 'phase0_model_trace_n8/model_execution_trace.jsonl'
records = [json.loads(line) for line in trace.open(encoding='utf-8')]
observations = normalize_model_trace(records)
locks = [row for row in records if row.get('event_type') == 'MODEL_LOCK']
by_request = collections.defaultdict(list)
for row in locks:
    ids = row.get('request_ids') or []
    if ids:
        by_request[str(ids[0])].append(row)
seen = collections.defaultdict(int)
delays = collections.defaultdict(list)
previous = {}
first_ready = {}
matched = 0
for observation in observations:
    request_id = observation.request_ids[0]
    rows = by_request[request_id]
    index = seen[request_id]
    selected = None
    while index < len(rows):
        if int(rows[index]['timestamp_monotonic_ns']) >= observation.timestamp_ns:
            selected = rows[index]
            index += 1
            break
        index += 1
    seen[request_id] = index
    if selected is None:
        continue
    attempt = (
        int(selected['timestamp_monotonic_ns'])
        - int(selected.get('hold_ns', 0))
        - int(selected.get('wait_ns', 0))
    )
    matched += 1
    if request_id in previous:
        delays[request_id].append(max(0, attempt - previous[request_id][1]))
    else:
        first_ready[request_id] = attempt
    previous[request_id] = (observation.timestamp_ns, attempt)
values = sum(delays.values(), [])
print('matched', matched, 'requests', len(first_ready))
print('first_ready_relative_ms', [(key, round((value - min(first_ready.values())) / 1e6, 3)) for key, value in first_ready.items()])
print('delay_count', len(values))
if values:
    ordered = sorted(values)
    print('delay_ms_median_p10_p90', *(round(value / 1e6, 3) for value in (statistics.median(values), ordered[len(ordered)//10], ordered[int(len(ordered)*.9)])))
    print('delay_lt_10ms_fraction', sum(value < 10e6 for value in values) / len(values))
    print('per_request', [(key, round(statistics.median(value) / 1e6, 3), len(value)) for key, value in delays.items()])
