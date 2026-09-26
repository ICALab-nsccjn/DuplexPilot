import unittest

from lychee_fd.runtime.apr.contracts import (
    AcousticPcmRecord,
    AcousticTokenBatch,
    APRContractError,
    validate_sequence_chain,
)


def token_batch(**overrides):
    values = {
        "request_id": "request-a",
        "stream_id": "stream-a",
        "generation_id": 0,
        "sequence_no": 0,
        "stoken_ids": (11, 12),
        "source_execution_id": "exec-1",
        "state_version": 0,
        "created_monotonic_ns": 100,
    }
    values.update(overrides)
    return AcousticTokenBatch(**values)


class AcousticContractTests(unittest.TestCase):
    def test_token_batch_is_immutable_and_normalizes_tokens_to_tuple(self):
        batch = token_batch(stoken_ids=[11, 12])

        self.assertIsInstance(batch.stoken_ids, tuple)
        with self.assertRaises((AttributeError, TypeError)):
            batch.sequence_no = 1

    def test_missing_identity_is_rejected(self):
        for field in ("request_id", "stream_id", "source_execution_id"):
            with self.subTest(field=field):
                with self.assertRaises(APRContractError):
                    token_batch(**{field: ""})

    def test_invalid_sequence_and_generation_values_are_rejected(self):
        for field, value in (("sequence_no", -1), ("generation_id", -1), ("state_version", -1)):
            with self.subTest(field=field):
                with self.assertRaises(APRContractError):
                    token_batch(**{field: value})

    def test_sequence_chain_requires_same_request_generation_and_strict_progress(self):
        first = token_batch(sequence_no=4)
        second = token_batch(sequence_no=5, source_execution_id="exec-2", created_monotonic_ns=200)
        self.assertTrue(validate_sequence_chain(first, second))

        for bad in (
            token_batch(sequence_no=4, source_execution_id="exec-3", created_monotonic_ns=200),
            token_batch(sequence_no=6, generation_id=1, source_execution_id="exec-4", created_monotonic_ns=200),
            token_batch(sequence_no=6, request_id="request-b", source_execution_id="exec-5", created_monotonic_ns=200),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(APRContractError):
                    validate_sequence_chain(first, bad)

    def test_pcm_record_requires_owned_nonempty_pcm(self):
        record = AcousticPcmRecord(
            request_id="request-a",
            stream_id="stream-a",
            generation_id=0,
            sequence_no=0,
            pcm_bytes=b"\x00\x01",
            sample_rate=24000,
            pcm_seq=0,
        )
        self.assertEqual(record.sample_rate, 24000)
        for field, value in (("request_id", ""), ("pcm_bytes", b""), ("sample_rate", 0), ("pcm_seq", -1)):
            with self.subTest(field=field):
                values = {
                    "request_id": "request-a",
                    "stream_id": "stream-a",
                    "generation_id": 0,
                    "sequence_no": 0,
                    "pcm_bytes": b"\x00\x01",
                    "sample_rate": 24000,
                    "pcm_seq": 0,
                }
                values[field] = value
                with self.assertRaises(APRContractError):
                    AcousticPcmRecord(**values)


if __name__ == "__main__":
    unittest.main()
