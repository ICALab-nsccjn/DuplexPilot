import unittest

from tools.apr.run_apr_workload_suite import (
    NOT_AVAILABLE_SYSTEM,
    build_matrix_specs,
    build_runner_command,
)


class APRWorkloadMatrixRunnerTests(unittest.TestCase):
    def test_matrix_has_one_warmup_and_counted_repeats_for_runnable_systems(self):
        specs = build_matrix_specs(
            workloads=("A",),
            concurrencies=(4,),
            systems=("original_affinity", "apr"),
            repeats=2,
        )

        self.assertEqual(
            [(spec.system, spec.warmup, spec.repeat) for spec in specs],
            [
                ("original_affinity", True, 1),
                ("original_affinity", False, 1),
                ("original_affinity", False, 2),
                ("apr", True, 1),
                ("apr", False, 1),
                ("apr", False, 2),
            ],
        )

    def test_no_rsv_dsv_ablation_is_explicitly_unavailable(self):
        specs = build_matrix_specs(
            workloads=("C",),
            concurrencies=(16,),
            systems=(NOT_AVAILABLE_SYSTEM,),
            repeats=5,
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].status, "NOT_AVAILABLE")
        self.assertFalse(specs[0].warmup)

    def test_runner_command_contains_frozen_trace_and_performance_flags(self):
        spec = build_matrix_specs(
            workloads=("B",),
            concurrencies=(8,),
            systems=("apr",),
            repeats=1,
        )[0]

        command = build_runner_command(
            python_executable="/opt/fdmodel/bin/python",
            runner_path="tools/apr/run_real_apr_e2e.py",
            out_root="reports/apr_workload_suite/runs/B/N8/apr/attempt-01",
            spec=spec,
            arrival_process="burst",
        )

        self.assertIn("--performance", command)
        self.assertIn("--workload", command)
        self.assertIn("B", command)
        self.assertIn("--arrival-process", command)
        self.assertIn("burst", command)
        self.assertIn("--warmup", command)


if __name__ == "__main__":
    unittest.main()
