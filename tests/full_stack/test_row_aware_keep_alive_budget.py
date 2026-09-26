import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_SOURCE = (
    _REPO_ROOT
    / "_remote_edit"
    / "lychee_fd"
    / "vllm_integration"
    / "engine.py"
)


class RowAwareKeepAliveBudgetTests(unittest.TestCase):
    def test_persistent_row_aware_request_reserves_remaining_model_budget(self):
        tree = ast.parse(ENGINE_SOURCE.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_step_generate_stream_row_aware"
        )

        keep_alive_ifs = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "keep_request_alive"
        ]
        self.assertTrue(
            keep_alive_ifs,
            "row-aware persistent generation must have an explicit keep-alive budget branch",
        )

        body_source = ast.get_source_segment(
            ENGINE_SOURCE.read_text(encoding="utf-8"), method
        )
        self.assertIsNotNone(body_source)
        self.assertIn("self._max_model_len", body_source)
        self.assertIn("generated_input_ids.shape[1]", body_source)
        self.assertIn("text_budget = max(text_budget, remaining)", body_source)


if __name__ == "__main__":
    unittest.main()
