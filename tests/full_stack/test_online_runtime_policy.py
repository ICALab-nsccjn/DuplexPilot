import unittest

from lychee_fd.runtime.policy import (
    RuntimeMode,
    RuntimePolicyError,
    resolve_runtime_policy,
)


class OnlineRuntimePolicyTests(unittest.TestCase):
    def test_unspecified_runtime_mode_defaults_to_native(self):
        policy = resolve_runtime_policy(None)

        self.assertIs(policy.mode, RuntimeMode.NATIVE)
        self.assertFalse(policy.row_aware)

    def test_dynamic_virtualized_selects_frozen_row_aware_execution(self):
        policy = resolve_runtime_policy("dynamic_virtualized")

        self.assertIs(policy.mode, RuntimeMode.DYNAMIC_VIRTUALIZED)
        self.assertTrue(policy.row_aware)
        self.assertTrue(policy.uses_shared_execution)

    def test_unknown_runtime_mode_fails_closed(self):
        with self.assertRaisesRegex(RuntimePolicyError, "unknown runtime_mode"):
            resolve_runtime_policy("silent_fallback")


if __name__ == "__main__":
    unittest.main()
