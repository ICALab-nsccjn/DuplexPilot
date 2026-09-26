import unittest

from tools.lychee_fd_life.life_runner import build_session_start_payload


class LifeRunnerRuntimeModeTests(unittest.TestCase):
    def test_runtime_mode_is_explicitly_carried_when_requested(self):
        payload = build_session_start_payload("dynamic_virtualized")
        self.assertEqual(payload["runtime_mode"], "dynamic_virtualized")

    def test_default_runner_payload_preserves_native_server_default(self):
        payload = build_session_start_payload(None)
        self.assertNotIn("runtime_mode", payload)


if __name__ == "__main__":
    unittest.main()
