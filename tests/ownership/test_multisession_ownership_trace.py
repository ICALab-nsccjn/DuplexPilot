import unittest


from lychee_fd.runtime.multisession_ownership_trace import validate_ownership_trace


def _event(step, row, owner="A", *, chunk_id=None, timestamp=None, **extra):
    return {
        "event_type": "ownership",
        "run_id": "run-1",
        "session_id": owner,
        "logical_request_id": owner,
        "response_id": "response-1",
        "execution_step": step,
        "timestamp_monotonic_ns": timestamp if timestamp is not None else 1000 + step,
        "physical_batch_size": 2,
        "physical_row": row,
        "row_state_owner": owner,
        "row_side_output_owner": owner,
        "sampler_owner": owner,
        "stoken_owner": owner,
        "streaming_decoder_owner": owner,
        "token2wav_owner": owner,
        "pcm_owner": owner,
        "playback_owner": owner,
        "pcm_chunk_id": chunk_id,
        "pcm_sequence_index": step if chunk_id is not None else None,
        **extra,
    }


class MultiSessionOwnershipTraceTests(unittest.TestCase):
    def test_row_change_preserves_logical_owner(self):
        result = validate_ownership_trace(
            [
                _event(1, 0, owner="A"),
                _event(2, 1, owner="A", chunk_id="A:0"),
            ]
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["first_divergence"], None)

    def test_wrong_pcm_owner_reports_first_divergence(self):
        result = validate_ownership_trace(
            [_event(1, 0, owner="A", chunk_id="A:0", pcm_owner="B")]
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["first_divergence"]["boundary"], "pcm_owner")
        self.assertEqual(result["first_divergence"]["expected_owner"], "A")
        self.assertEqual(result["first_divergence"]["actual_owner"], "B")

    def test_duplicate_pcm_chunk_id_is_rejected(self):
        result = validate_ownership_trace(
            [
                _event(1, 0, owner="A", chunk_id="A:0"),
                _event(2, 1, owner="A", chunk_id="A:0"),
            ]
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["first_divergence"]["boundary"], "pcm_chunk_id")


if __name__ == "__main__":
    unittest.main()
