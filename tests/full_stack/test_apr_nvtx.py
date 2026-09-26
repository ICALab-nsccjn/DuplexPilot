import unittest


class _FakeNVTX:
    def __init__(self):
        self.calls = []

    def range_push(self, name):
        self.calls.append(("push", name))

    def range_pop(self):
        self.calls.append(("pop", None))


class APRNVTXTests(unittest.TestCase):
    def test_unavailable_provider_is_a_noop(self):
        from lychee_fd.runtime.apr import nvtx

        original = nvtx._load_provider
        nvtx._load_provider = lambda: None
        try:
            with nvtx.range("APR_CHECKPOINT"):
                pass
            self.assertIsNone(nvtx.mark("APR_RESTORE"))
        finally:
            nvtx._load_provider = original

    def test_provider_receives_stable_push_and_pop_markers(self):
        from lychee_fd.runtime.apr import nvtx

        fake = _FakeNVTX()
        original = nvtx._load_provider
        nvtx._load_provider = lambda: fake
        try:
            with nvtx.range("TOKEN2WAV_FLOW"):
                pass
            nvtx.mark("VOCODER")
        finally:
            nvtx._load_provider = original
        self.assertEqual(fake.calls, [
            ("push", "TOKEN2WAV_FLOW"),
            ("pop", None),
            ("push", "VOCODER"),
            ("pop", None),
        ])


if __name__ == "__main__":
    unittest.main()
