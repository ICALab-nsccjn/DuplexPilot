import json
from pathlib import Path

root = Path('/mnt/DuplexPilot/data/DuplexPilot/results/apr_row_aware_model_plane_e2e_20260831')
records = [json.loads(line) for line in (root / 'phase0_model_trace_n8/model_execution_trace.jsonl').open(encoding='utf-8')]
request_id = 'ddb2bf46c17540e898ac680d90d7347a'
for row in records:
    event = row.get('event_type')
    if event == 'MODEL_ENGINE_STEP_START' and row.get('requested_request_id') == request_id and 15 <= int(row.get('engine_step', -1)) <= 22:
        print('START', row)
    elif event == 'MODEL_ENGINE_STEP_END' and row.get('requested_request_id') == request_id and 15 <= int(row.get('engine_step', -1)) <= 22:
        print('END', row)
    elif event == 'MODEL_LOCK' and request_id in (row.get('request_ids') or []):
        release = int(row['timestamp_monotonic_ns'])
        if 870787000000000 <= release <= 870789500000000:
            attempt = release - int(row.get('hold_ns', 0)) - int(row.get('wait_ns', 0))
            print('LOCK', {**row, 'attempt_ns': attempt})
print('--- chronological window ---')
for row in records:
    t = int(row.get('timestamp_monotonic_ns', 0))
    if 870786800000000 <= t <= 870787200000000:
        event = row.get('event_type')
        if event in {'MODEL_ENGINE_STEP_START', 'MODEL_ENGINE_STEP_END', 'MODEL_ROW_OUTPUT', 'MODEL_LOCK'}:
            if event == 'MODEL_LOCK' and request_id not in (row.get('request_ids') or []):
                continue
            if event != 'MODEL_LOCK' and row.get('requested_request_id', row.get('request_id')) != request_id:
                continue
            print(event, row)
