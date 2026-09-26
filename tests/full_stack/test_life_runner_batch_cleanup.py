import threading
import unittest

from tools import life_runner


class _UnfinishedRunner:
    session_id = "session-1"

    def __init__(self) -> None:
        self.done = threading.Event()
        self.stop_calls = []

    def _request_stop(self, timeout_sec=None) -> None:
        self.stop_calls.append(timeout_sec)


class _FinishedRunner(_UnfinishedRunner):
    def __init__(self) -> None:
        super().__init__()
        self.done.set()


class LifeRunnerBatchCleanupTests(unittest.TestCase):
    def test_batch_deadline_requests_stop_for_unfinished_started_sessions(self):
        unfinished = _UnfinishedRunner()
        finished = _FinishedRunner()

        life_runner.request_stop_for_unfinished_runners(
            [unfinished, finished], request_timeout_sec=10.0
        )

        self.assertEqual(unfinished.stop_calls, [10.0])
        self.assertEqual(finished.stop_calls, [])


if __name__ == "__main__":
    unittest.main()
