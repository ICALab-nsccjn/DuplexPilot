import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = next(
    path
    for path in (
        _REPO_ROOT / "lychee_fd" / "app.py",
        _REPO_ROOT / "_remote_edit" / "lychee_fd" / "app.py",
    )
    if path.exists()
)


class OfficialOnlinePolicyThreadingTests(unittest.TestCase):
    def test_session_start_resolves_runtime_policy_before_constructing_state(self):
        tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
        start = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "start_realtime_session"
        )

        calls = [
            node
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_runtime_policy"
        ]
        self.assertTrue(
            calls,
            "official session-start must resolve runtime_mode before worker creation",
        )

        assignments = {
            target.id
            for node in ast.walk(start)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("runtime_mode", assignments)

    def test_official_app_bootstraps_configured_patched_vllm_source(self):
        source = APP_SOURCE.read_text(encoding="utf-8")

        self.assertIn("LYCHEEFD_VLLM_SOURCE_DIR", source)
        self.assertIn("sys.path.insert(0, VLLM_SOURCE_DIR)", source)
        self.assertIn("vllm", source)


if __name__ == "__main__":
    unittest.main()
