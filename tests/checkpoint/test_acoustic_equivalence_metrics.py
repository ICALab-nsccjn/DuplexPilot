import unittest

import numpy as np

from lychee_fd.runtime.acoustic_equivalence import (
    PcmEquivalenceThresholds,
    compare_pcm_equivalence,
)


class AcousticEquivalenceMetricTests(unittest.TestCase):
    def test_bitwise_equal_pcm_passes_all_levels(self):
        pcm = np.array([0, 1000, -1000, 2000, -2000], dtype="<i2").tobytes()
        result = compare_pcm_equivalence(pcm, pcm, 24000, 24000)

        self.assertTrue(result.passed)
        self.assertTrue(result.bitwise_equal)
        self.assertEqual(result.sample_count, 5)
        self.assertEqual(result.normalized_rmse, 0.0)

    def test_small_signal_variation_passes_registered_thresholds(self):
        reference = np.sin(np.linspace(0, 20, 24000)) * 12000
        migrated = reference + 100
        result = compare_pcm_equivalence(
            reference.astype("<i2").tobytes(),
            migrated.astype("<i2").tobytes(),
            24000,
            24000,
        )

        self.assertTrue(result.passed)
        self.assertFalse(result.bitwise_equal)
        self.assertLessEqual(result.normalized_rmse, 0.02)
        self.assertGreaterEqual(result.correlation, 0.99)
        self.assertGreaterEqual(result.snr_db, 34.0)

    def test_large_variation_and_metadata_mismatch_fail(self):
        reference = np.full(1000, 10000, dtype="<i2").tobytes()
        migrated = np.full(1000, -10000, dtype="<i2").tobytes()
        result = compare_pcm_equivalence(reference, migrated, 24000, 16000)

        self.assertFalse(result.passed)
        self.assertFalse(result.same_sample_rate)
        self.assertTrue(result.same_signal_length)
        self.assertIn("sample_rate", result.failure_reasons)

    def test_custom_thresholds_are_applied_without_hidden_tuning(self):
        pcm = np.array([0, 100, 200, 300], dtype="<i2").tobytes()
        thresholds = PcmEquivalenceThresholds(
            normalized_rmse_max=0.0,
            correlation_min=1.0,
            snr_db_min=100.0,
        )
        result = compare_pcm_equivalence(pcm, pcm, 24000, 24000, thresholds)

        self.assertTrue(result.passed)
        self.assertEqual(result.thresholds, thresholds)


if __name__ == "__main__":
    unittest.main()
