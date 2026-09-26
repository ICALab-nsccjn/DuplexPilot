import unittest
from types import SimpleNamespace

from vllm.engine.output_processor.single_step import (
    _append_multihead_side_tokens,
)
from vllm.sequence import MultiHeadRequestState


class _MutableProcessor:
    def __init__(self):
        self.prefix_cnt = 0
        self.cnt = 0
        self.has_eos = False


class AcousticHandoffRegressionTests(unittest.TestCase):
    def test_worker_processor_progress_is_committed_to_request_state(self):
        processor = _MutableProcessor()
        state = MultiHeadRequestState(
            session_id="request-A",
            stoken_processors=[processor],
            row_aware_enabled=True,
        )

        # This is the actual scheduler -> worker boundary: the payload owns a
        # deep copy, so mutations made by the worker are not mutations of the
        # persistent request state.
        worker_payload = state.to_worker_payload()
        worker_processor = worker_payload["stoken_processors"][0]
        worker_processor.prefix_cnt = 1

        seq_group = SimpleNamespace(multihead_request_state=state)
        output = SimpleNamespace(
            samples=[],
            stoken_token_ids=[158361],
            control_token_ids=[158359],
            lychee_processor_state={
                "text": [],
                "stoken": [{
                    "prefix_cnt": worker_processor.prefix_cnt,
                    "cnt": worker_processor.cnt,
                    "has_eos": worker_processor.has_eos,
                }],
                "control": [],
            },
        )

        _append_multihead_side_tokens(seq_group, output)

        self.assertEqual(state.stoken_processors[0].prefix_cnt, 1)
        next_payload = state.to_worker_payload()
        self.assertEqual(
            next_payload["stoken_processors"][0].prefix_cnt,
            1,
        )


if __name__ == "__main__":
    unittest.main()
