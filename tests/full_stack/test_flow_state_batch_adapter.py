import unittest

import torch

from profiling.token2wav_flow_profile.batching import (
    FlowStateBatchAdapter,
    IncompatibleFlowState,
)
from profiling.token2wav_flow_profile.contracts import FlowStep


class FlowStateBatchAdapterTests(unittest.TestCase):
    def _step(
        self,
        request_id,
        *,
        token_value=0,
        speaker_value=0,
        cache=None,
        tokens=None,
        speaker=None,
        model_identity="flow-v1",
    ):
        if tokens is None:
            tokens = torch.tensor(
                [[token_value, token_value + 1, token_value + 2]],
                dtype=torch.int64,
            )
        if speaker is None:
            speaker = torch.tensor(
                [[speaker_value, speaker_value + 1]],
                dtype=torch.float32,
            )
        if cache is None:
            cache = {
                "flow_h": torch.tensor(
                    [[token_value + 0.1, token_value + 0.2]],
                    dtype=torch.float32,
                ),
                "flow_k": torch.tensor(
                    [[[token_value + 1.0, token_value + 2.0]]],
                    dtype=torch.float32,
                ),
            }
        return FlowStep(
            request_id=request_id,
            generation_id=7,
            sequence_no=3,
            tokens=tokens,
            speaker=speaker,
            flow_cache=cache,
            last_chunk=False,
            n_timesteps=4,
            model_identity=model_identity,
        )

    def test_compatible_singleton_cache_states_pack_and_split_without_aliasing(self):
        first = self._step("request-a", token_value=10, speaker_value=20)
        second = self._step("request-b", token_value=30, speaker_value=40)
        adapter = FlowStateBatchAdapter()

        packed = adapter.pack((first, second))
        output_mel = packed.tokens.clone()
        output_cache = {
            key: value.clone() for key, value in packed.flow_cache.items()
        }
        split = adapter.split(packed, output_mel, output_cache)

        self.assertEqual(packed.tokens.shape, (2, 3))
        self.assertEqual(packed.speaker.shape, (2, 2))
        self.assertEqual(packed.flow_cache["flow_h"].shape, (2, 2))
        self.assertEqual(packed.flow_cache["flow_k"].shape, (2, 1, 2))
        self.assertEqual([state.request_id for state, _ in split], ["request-a", "request-b"])

        for actual, expected in zip((split[0][0], split[1][0]), (first, second)):
            self.assertTrue(torch.equal(actual.tokens, expected.tokens))
            self.assertTrue(torch.equal(actual.speaker, expected.speaker))
            for key in expected.flow_cache:
                self.assertTrue(torch.equal(actual.flow_cache[key], expected.flow_cache[key]))

        split_tokens_before = split[1][0].tokens.clone()
        split_speaker_before = split[1][0].speaker.clone()
        split_flow_h_before = split[1][0].flow_cache["flow_h"].clone()
        split_flow_k_before = split[1][0].flow_cache["flow_k"].clone()
        packed_tokens_before = packed.tokens.clone()
        packed_speaker_before = packed.speaker.clone()
        packed_flow_h_before = packed.flow_cache["flow_h"].clone()
        packed_flow_k_before = packed.flow_cache["flow_k"].clone()
        output_mel_before = output_mel.clone()
        output_cache_before = {
            key: value.clone() for key, value in output_cache.items()
        }

        # Mutating one returned state must not affect the other returned state,
        # packed state, or the caller-owned output buffers.
        split[0][0].tokens[0, 0] = -101
        split[0][0].speaker[0, 0] = -102
        split[0][0].flow_cache["flow_h"][0, 0] = -103
        split[0][0].flow_cache["flow_k"][0, 0, 0] = -104

        self.assertTrue(torch.equal(split[1][0].tokens, split_tokens_before))
        self.assertTrue(torch.equal(split[1][0].speaker, split_speaker_before))
        self.assertTrue(
            torch.equal(split[1][0].flow_cache["flow_h"], split_flow_h_before)
        )
        self.assertTrue(
            torch.equal(split[1][0].flow_cache["flow_k"], split_flow_k_before)
        )

        self.assertTrue(torch.equal(packed.tokens, packed_tokens_before))
        self.assertTrue(torch.equal(packed.speaker, packed_speaker_before))
        self.assertTrue(
            torch.equal(packed.flow_cache["flow_h"], packed_flow_h_before)
        )
        self.assertTrue(
            torch.equal(packed.flow_cache["flow_k"], packed_flow_k_before)
        )
        self.assertTrue(torch.equal(output_mel, output_mel_before))
        for key, value in output_cache_before.items():
            self.assertTrue(torch.equal(output_cache[key], value))

        # Mutating packed state after split must not affect either returned
        # state, including every tensor field and every returned cache entry.
        packed.tokens[1, 0] = -201
        packed.speaker[1, 0] = -202
        packed.flow_cache["flow_h"][1, 0] = -203
        packed.flow_cache["flow_k"][1, 0, 0] = -204

        self.assertEqual(split[0][0].tokens[0, 0].item(), -101)
        self.assertEqual(split[0][0].speaker[0, 0].item(), -102)
        self.assertEqual(split[0][0].flow_cache["flow_h"][0, 0].item(), -103)
        self.assertEqual(split[0][0].flow_cache["flow_k"][0, 0, 0].item(), -104)
        self.assertTrue(torch.equal(split[1][0].tokens, split_tokens_before))
        self.assertTrue(torch.equal(split[1][0].speaker, split_speaker_before))
        self.assertTrue(
            torch.equal(split[1][0].flow_cache["flow_h"], split_flow_h_before)
        )
        self.assertTrue(
            torch.equal(split[1][0].flow_cache["flow_k"], split_flow_k_before)
        )

    def test_pack_and_split_do_not_alias_original_input_states(self):
        first = self._step("request-a", token_value=10, speaker_value=20)
        second = self._step("request-b", token_value=30, speaker_value=40)
        packed = FlowStateBatchAdapter().pack((first, second))
        output_mel = packed.tokens.clone()
        output_cache = {
            key: value.clone() for key, value in packed.flow_cache.items()
        }
        split = FlowStateBatchAdapter().split(packed, output_mel, output_cache)

        first_before_split_mutation = (
            first.tokens.clone(),
            first.speaker.clone(),
            {key: value.clone() for key, value in first.flow_cache.items()},
        )
        second_before_split_mutation = (
            second.tokens.clone(),
            second.speaker.clone(),
            {key: value.clone() for key, value in second.flow_cache.items()},
        )

        split[0][0].tokens[0, 0] = -301
        split[0][0].speaker[0, 0] = -302
        split[0][0].flow_cache["flow_h"][0, 0] = -303
        split[0][0].flow_cache["flow_k"][0, 0, 0] = -304
        split[1][0].tokens[0, 0] = -311
        split[1][0].speaker[0, 0] = -312
        split[1][0].flow_cache["flow_h"][0, 0] = -313
        split[1][0].flow_cache["flow_k"][0, 0, 0] = -314

        self.assertTrue(torch.equal(first.tokens, first_before_split_mutation[0]))
        self.assertTrue(torch.equal(first.speaker, first_before_split_mutation[1]))
        self.assertTrue(
            torch.equal(
                first.flow_cache["flow_h"], first_before_split_mutation[2]["flow_h"]
            )
        )
        self.assertTrue(
            torch.equal(
                first.flow_cache["flow_k"], first_before_split_mutation[2]["flow_k"]
            )
        )
        self.assertTrue(torch.equal(second.tokens, second_before_split_mutation[0]))
        self.assertTrue(torch.equal(second.speaker, second_before_split_mutation[1]))
        self.assertTrue(
            torch.equal(
                second.flow_cache["flow_h"], second_before_split_mutation[2]["flow_h"]
            )
        )
        self.assertTrue(
            torch.equal(
                second.flow_cache["flow_k"], second_before_split_mutation[2]["flow_k"]
            )
        )

        packed_before_input_mutation = (
            packed.tokens.clone(),
            packed.speaker.clone(),
            {key: value.clone() for key, value in packed.flow_cache.items()},
        )
        split_before_input_mutation = (
            (split[0][0].tokens.clone(), split[0][0].speaker.clone()),
            {
                key: value.clone() for key, value in split[0][0].flow_cache.items()
            },
            (split[1][0].tokens.clone(), split[1][0].speaker.clone()),
            {
                key: value.clone() for key, value in split[1][0].flow_cache.items()
            },
        )
        output_before_input_mutation = (
            output_mel.clone(),
            {key: value.clone() for key, value in output_cache.items()},
        )

        first.tokens[0, 0] = -401
        first.speaker[0, 0] = -402
        first.flow_cache["flow_h"][0, 0] = -403
        first.flow_cache["flow_k"][0, 0, 0] = -404
        second.tokens[0, 0] = -411
        second.speaker[0, 0] = -412
        second.flow_cache["flow_h"][0, 0] = -413
        second.flow_cache["flow_k"][0, 0, 0] = -414

        self.assertTrue(torch.equal(packed.tokens, packed_before_input_mutation[0]))
        self.assertTrue(torch.equal(packed.speaker, packed_before_input_mutation[1]))
        for key, value in packed_before_input_mutation[2].items():
            self.assertTrue(torch.equal(packed.flow_cache[key], value))
        self.assertTrue(torch.equal(split[0][0].tokens, split_before_input_mutation[0][0]))
        self.assertTrue(torch.equal(split[0][0].speaker, split_before_input_mutation[0][1]))
        for key, value in split_before_input_mutation[1].items():
            self.assertTrue(torch.equal(split[0][0].flow_cache[key], value))
        self.assertTrue(torch.equal(split[1][0].tokens, split_before_input_mutation[2][0]))
        self.assertTrue(torch.equal(split[1][0].speaker, split_before_input_mutation[2][1]))
        for key, value in split_before_input_mutation[3].items():
            self.assertTrue(torch.equal(split[1][0].flow_cache[key], value))
        self.assertTrue(torch.equal(output_mel, output_before_input_mutation[0]))
        for key, value in output_before_input_mutation[1].items():
            self.assertTrue(torch.equal(output_cache[key], value))

    def test_cache_key_mismatch_is_incompatible(self):
        adapter = FlowStateBatchAdapter()
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": first.flow_cache["flow_h"].clone(),
                "different_key": first.flow_cache["flow_k"].clone(),
            },
        )

        with self.assertRaises(IncompatibleFlowState):
            adapter.pack((first, second))

    def test_cache_non_batch_shape_mismatch_is_incompatible(self):
        adapter = FlowStateBatchAdapter()
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": torch.zeros((1, 3), dtype=torch.float32),
                "flow_k": first.flow_cache["flow_k"].clone(),
            },
        )

        with self.assertRaises(IncompatibleFlowState):
            adapter.pack((first, second))

    def test_cache_dtype_mismatch_is_incompatible(self):
        adapter = FlowStateBatchAdapter()
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": first.flow_cache["flow_h"].to(torch.float64),
                "flow_k": first.flow_cache["flow_k"].clone(),
            },
        )

        with self.assertRaises(IncompatibleFlowState):
            adapter.pack((first, second))

    def test_tokens_dtype_mismatch_is_incompatible(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            tokens=first.tokens.to(torch.float32),
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_tokens_shape_mismatch_is_incompatible(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            tokens=torch.zeros((1, 4), dtype=torch.int64),
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_speaker_dtype_mismatch_is_incompatible(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            speaker=first.speaker.to(torch.float64),
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_speaker_shape_mismatch_is_incompatible(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            speaker=torch.zeros((1, 3), dtype=torch.float32),
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for device mismatch")
    def test_device_mismatch_is_incompatible(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": first.flow_cache["flow_h"].cuda(),
                "flow_k": first.flow_cache["flow_k"].cuda(),
            },
            tokens=first.tokens.cuda(),
            speaker=first.speaker.cuda(),
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_non_tensor_cache_value_is_rejected_instead_of_broadcast(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": [1.0, 2.0],
                "flow_k": first.flow_cache["flow_k"].clone(),
            },
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_scalar_cache_value_is_rejected_instead_of_broadcast(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": torch.tensor(1.0, dtype=torch.float32),
                "flow_k": first.flow_cache["flow_k"].clone(),
            },
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_singleton_cache_value_is_rejected_instead_of_broadcast(self):
        first = self._step("request-a")
        second = self._step(
            "request-b",
            cache={
                "flow_h": torch.ones((1, 1), dtype=torch.float32),
                "flow_k": first.flow_cache["flow_k"].clone(),
            },
        )

        with self.assertRaises(IncompatibleFlowState):
            FlowStateBatchAdapter().pack((first, second))

    def test_token_and_speaker_batch_are_split_in_request_order(self):
        steps = tuple(
            self._step(
                request_id,
                token_value=token_value,
                speaker_value=speaker_value,
            )
            for request_id, token_value, speaker_value in (
                ("request-a", 1, 101),
                ("request-b", 11, 111),
                ("request-c", 21, 121),
            )
        )
        adapter = FlowStateBatchAdapter()

        packed = adapter.pack(steps)
        output_mel = packed.tokens.clone()
        output_cache = {
            key: value.clone() for key, value in packed.flow_cache.items()
        }
        split = adapter.split(packed, output_mel, output_cache)

        self.assertEqual(
            [state.request_id for state, _ in split],
            ["request-a", "request-b", "request-c"],
        )
        for (actual, _), expected in zip(split, steps):
            self.assertTrue(torch.equal(actual.tokens, expected.tokens))
            self.assertTrue(torch.equal(actual.speaker, expected.speaker))
            self.assertEqual(actual.request_id, expected.request_id)
            self.assertEqual(actual.generation_id, expected.generation_id)
            self.assertEqual(actual.sequence_no, expected.sequence_no)


if __name__ == "__main__":
    unittest.main()
