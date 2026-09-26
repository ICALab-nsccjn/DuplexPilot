import json
import math
import tempfile
import unittest
from pathlib import Path

from profiling.token2wav_flow_profile import (
    FlowOperatorRecord,
    FlowProfiler,
    aggregate_operator_records,
)
from profiling.token2wav_flow_profile.profiler import _as_float


class FlowProfileTests(unittest.TestCase):
    class FakeProfilerProvider:
        def __init__(self, events):
            self.events = events
            self.calls = 0

        def profile(self, call, *, shapes):
            self.calls += 1
            result = call()
            return result, self.events

    IDENTITY = {
        "request_id": "request-7",
        "generation_id": 3,
        "batch_signature": "flow:fp16:cuda:tokens-32",
    }

    SHAPES = {
        "input_shapes": ((2, 32), (2, 128)),
        "output_shapes": ((2, 64),),
    }

    FAKE_EVENTS = (
        {
            "operator_name": "aten::matmul",
            "cpu_time_us": 120.0,
            "cuda_time_us": 95.0,
            "memory_bytes": 4096,
            "kernel_count": 3,
            "input_shapes": ((2, 32), (2, 128)),
            "output_shapes": ((2, 64),),
        },
        {
            "operator_name": "aten::gelu",
            "cpu_time_us": 40.0,
            "cuda_time_us": 12.0,
            "memory_bytes": 1024,
            "kernel_count": 1,
            "input_shapes": ((2, 64),),
            "output_shapes": ((2, 64),),
        },
    )

    def _record(self, *, name, cpu, cuda, memory, kernels):
        return FlowOperatorRecord(
            operator_name=name,
            cpu_time_us=cpu,
            cuda_time_us=cuda,
            memory_bytes=memory,
            kernel_count=kernels,
            input_shapes=((2, 32),),
            output_shapes=((2, 64),),
            request_id=self.IDENTITY["request_id"],
            generation_id=self.IDENTITY["generation_id"],
            batch_signature=self.IDENTITY["batch_signature"],
        )

    def test_disabled_profiler_is_a_noop(self):
        profiler = FlowProfiler(enabled=False)
        calls = []

        result = profiler.profile_call(
            lambda: calls.append("backend-called") or "result",
            identity=self.IDENTITY,
            shapes=self.SHAPES,
        )
        profiler.record_fake_events(self.FAKE_EVENTS, identity=self.IDENTITY)

        self.assertEqual(result, "result")
        self.assertEqual(calls, ["backend-called"])
        self.assertEqual(profiler.records, ())
        self.assertIsNone(profiler.flush())

    def test_disabled_flush_does_not_touch_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "flow.jsonl"
            profiler = FlowProfiler(enabled=False, output_path=output_path)

            self.assertIsNone(profiler.flush())
            self.assertFalse(output_path.exists())

    def test_enabled_profiler_records_identity_shapes_and_timing(self):
        profiler = FlowProfiler(enabled=True)
        profiler.record_fake_events(self.FAKE_EVENTS, identity=self.IDENTITY)

        self.assertEqual(len(profiler.records), 2)
        record = profiler.records[0]
        self.assertEqual(record.operator_name, "aten::matmul")
        self.assertEqual(record.request_id, "request-7")
        self.assertEqual(record.generation_id, 3)
        self.assertEqual(record.batch_signature, "flow:fp16:cuda:tokens-32")
        self.assertEqual(record.input_shapes, ((2, 32), (2, 128)))
        self.assertEqual(record.output_shapes, ((2, 64),))
        self.assertEqual(record.cpu_time_us, 120.0)
        self.assertEqual(record.cuda_time_us, 95.0)
        self.assertEqual(record.memory_bytes, 4096)
        self.assertEqual(record.kernel_count, 3)
        for record in profiler.records:
            self.assertEqual(record.request_id, self.IDENTITY["request_id"])
            self.assertEqual(record.generation_id, self.IDENTITY["generation_id"])
            self.assertEqual(
                record.batch_signature,
                self.IDENTITY["batch_signature"],
            )

    def test_enabled_flush_writes_schema_valid_json_and_profile_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "flow.jsonl"
            profiler = FlowProfiler(enabled=True, output_path=output_path)
            profiler.record_fake_events(self.FAKE_EVENTS, identity=self.IDENTITY)

            flushed_path = profiler.flush()

            self.assertEqual(flushed_path, output_path)
            self.assertTrue(output_path.exists())
            lines = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertGreaterEqual(len(lines), 3)
            for line in lines:
                self.assertEqual(line["schema_version"], "token2wav-flow-profile-v1")
            self.assertEqual(lines[0]["record_type"], "metadata")
            self.assertEqual(lines[0]["profile_status"], dict(profiler.profile_status))

    def test_profile_status_is_read_only_and_fake_provider_safe(self):
        profiler = FlowProfiler(enabled=True, profiler_provider=self.FakeProfilerProvider(self.FAKE_EVENTS))

        self.assertEqual(profiler.profile_status["cuda_attribution"], "not_measured")
        with self.assertRaises(TypeError):
            profiler.profile_status["cuda_attribution"] = "available"

    def test_enabled_profile_call_uses_fake_provider_and_records_result(self):
        provider = self.FakeProfilerProvider(self.FAKE_EVENTS)
        profiler = FlowProfiler(enabled=True, profiler_provider=provider)
        calls = []

        result = profiler.profile_call(
            lambda: calls.append("backend-called") or "profiled-result",
            identity=self.IDENTITY,
            shapes=self.SHAPES,
        )

        self.assertEqual(result, "profiled-result")
        self.assertEqual(calls, ["backend-called"])
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(profiler.records), len(self.FAKE_EVENTS))
        self.assertEqual(profiler.records[0].operator_name, "aten::matmul")
        self.assertEqual(profiler.records[0].cuda_time_us, 95.0)

    def test_operator_aggregation_ranks_cuda_time_kernel_count_and_memory(self):
        records = (
            self._record(name="aten::slow", cpu=100, cuda=80, memory=100, kernels=2),
            self._record(name="aten::wide", cpu=70, cuda=20, memory=900, kernels=1),
            self._record(name="aten::many", cpu=30, cuda=10, memory=50, kernels=8),
        )

        aggregate = aggregate_operator_records(records)

        by_cuda = sorted(
            aggregate.values(),
            key=lambda item: item["total_cuda_time_us"],
            reverse=True,
        )
        by_kernels = sorted(
            aggregate.values(),
            key=lambda item: item["total_kernel_count"],
            reverse=True,
        )
        by_memory = sorted(
            aggregate.values(),
            key=lambda item: item["total_memory_bytes"],
            reverse=True,
        )

        self.assertEqual(by_cuda[0]["operator_name"], "aten::slow")
        self.assertEqual(by_kernels[0]["operator_name"], "aten::many")
        self.assertEqual(by_memory[0]["operator_name"], "aten::wide")
        self.assertEqual(aggregate["aten::slow"]["total_cuda_time_us"], 80)
        self.assertTrue(aggregate["aten::slow"]["cuda_attribution_complete"])
        self.assertEqual(aggregate["aten::slow"]["cuda_known_sample_count"], 1)

    def test_mixed_cuda_attribution_is_incomplete_and_total_is_unknown(self):
        records = (
            self._record(name="aten::mixed", cpu=10, cuda=8, memory=1, kernels=1),
            self._record(name="aten::mixed", cpu=12, cuda=None, memory=1, kernels=1),
        )

        aggregate = aggregate_operator_records(records)["aten::mixed"]

        self.assertIsNone(aggregate["total_cuda_time_us"])
        self.assertFalse(aggregate["cuda_attribution_complete"])
        self.assertEqual(aggregate["cuda_known_sample_count"], 1)

    def test_invalid_operator_record_is_rejected(self):
        valid = {
            "operator_name": "aten::valid",
            "cpu_time_us": 1.0,
            "cuda_time_us": 1.0,
            "memory_bytes": 1,
            "kernel_count": 1,
            "input_shapes": ((1,),),
            "output_shapes": ((1,),),
            "request_id": "request-7",
            "generation_id": 3,
            "batch_signature": "signature",
        }
        invalid_records = (
            {"operator_name": ""},
            {"cpu_time_us": -1.0},
            {"cuda_time_us": -1.0},
            {"cpu_time_us": math.nan},
            {"cuda_time_us": math.inf},
            {"memory_bytes": -1},
            {"kernel_count": -1},
            {"input_shapes": ((1, "not-an-int"),)},
            {"output_shapes": ((1, -1),)},
            {"request_id": ""},
            {"batch_signature": ""},
        )

        for override in invalid_records:
            with self.subTest(override=override):
                payload = {**valid, **override}
                with self.assertRaises(ValueError):
                    FlowOperatorRecord(**payload)

    def test_missing_identity_fields_are_rejected(self):
        profiler = FlowProfiler(enabled=True)
        for field in ("request_id", "generation_id", "batch_signature"):
            with self.subTest(field=field):
                identity = dict(self.IDENTITY)
                del identity[field]
                with self.assertRaises(ValueError):
                    profiler.record_fake_events(self.FAKE_EVENTS, identity=identity)

    def test_memory_none_is_allowed_when_memory_is_unavailable(self):
        record = self._record(name="aten::unknown", cpu=10, cuda=None, memory=None, kernels=1)

        self.assertIsNone(record.memory_bytes)
        aggregate = aggregate_operator_records((record,))
        self.assertIsNone(aggregate["aten::unknown"]["total_memory_bytes"])

    def test_float_parser_rejects_non_finite_values(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _as_float(value)


if __name__ == "__main__":
    unittest.main()
