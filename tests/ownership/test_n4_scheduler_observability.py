import unittest
from collections import deque
from types import SimpleNamespace


def _state(request_id, *, text_len=2, phase="listening"):
    return SimpleNamespace(
        session_id=request_id,
        text_input_ids=list(range(text_len)),
        stoken_input_ids=[1, 2],
        control_input_ids=[3, 4],
        phase=phase,
        interrupt_active=False,
        finished=False,
        row_aware_enabled=True,
    )


def _group(request_id, **state_kwargs):
    return SimpleNamespace(
        request_id=request_id,
        multihead_request_state=_state(request_id, **state_kwargs),
    )


class N4SchedulerObservabilityTests(unittest.TestCase):
    def test_admission_snapshot_preserves_queue_order_and_selection(self):
        from lychee_fd.runtime.n4_progress_trace import build_admission_event

        a = _group("A")
        b = _group("B", text_len=3)
        c = _group("C")
        running = deque([a])
        waiting = deque([b, c])
        swapped = deque()
        selected = [
            SimpleNamespace(seq_group=a),
            SimpleNamespace(seq_group=b),
        ]
        before = ([x.request_id for x in running], [x.request_id for x in waiting])

        event = build_admission_event(
            running=running,
            waiting=waiting,
            swapped=swapped,
            selected=selected,
            opportunity_id="sched-1",
        )

        self.assertEqual(before, (
            [x.request_id for x in running],
            [x.request_id for x in waiting],
        ))
        self.assertEqual(event["active_request_ids"], ["A"])
        self.assertEqual(event["pending_request_ids"], ["B", "C"])
        self.assertEqual(event["candidate_request_ids"], ["A", "B", "C"])
        self.assertEqual(event["selected_request_ids"], ["A", "B"])
        self.assertEqual(event["physical_batch_size"], 2)
        self.assertFalse(event["exact_compatible"])
        self.assertTrue(event["virtualizable_compatible"])
        self.assertTrue(event["restored_opportunity"])
        self.assertEqual(event["reason_selected"], {
            "A": "SELECTED_BY_EXISTING_SCHEDULER",
            "B": "SELECTED_BY_EXISTING_SCHEDULER",
        })
        self.assertEqual(event["reason_not_selected"], {
            "C": "NOT_SELECTED_BY_EXISTING_SCHEDULER",
        })

    def test_single_candidate_is_not_an_eligible_pair(self):
        from lychee_fd.runtime.n4_progress_trace import build_admission_event

        a = _group("A")
        event = build_admission_event(
            running=deque([a]),
            waiting=deque(),
            swapped=deque(),
            selected=[SimpleNamespace(seq_group=a)],
            opportunity_id="sched-2",
        )
        self.assertFalse(event["eligible_pair_opportunity"])
        self.assertIsNone(event["exact_compatible"])
        self.assertIsNone(event["virtualizable_compatible"])
        self.assertIsNone(event["restored_opportunity"])

    def test_physical_batch_event_preserves_runtime_rows(self):
        from lychee_fd.runtime.n4_progress_trace import build_physical_batch_event

        rows = [
            _state("A").__dict__,
            _state("B", text_len=4).__dict__,
        ]
        original_ids = [row["session_id"] for row in rows]
        event = build_physical_batch_event(rows)
        self.assertEqual([row["session_id"] for row in rows], original_ids)
        self.assertEqual(event["physical_batch_size"], 2)
        self.assertEqual(event["physical_rows"], {"0": "A", "1": "B"})
        self.assertFalse(event["exact_compatible"])
        self.assertTrue(event["virtualizable_compatible"])


if __name__ == "__main__":
    unittest.main()
