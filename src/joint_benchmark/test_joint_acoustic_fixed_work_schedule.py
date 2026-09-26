from __future__ import annotations

import pytest

from tools.benchmarks.run_joint_acoustic_fixed_work import _mixed_schedule_counts


def test_mixed_fixed_work_runs_until_first_row_reaches_terminal_step():
    assert _mixed_schedule_counts((2, 4), terminal_step=10) == (6, 2)


@pytest.mark.parametrize(
    ("start_steps", "expected"),
    [
        ((0, 0), (10, 0)),
        ((8, 9), (1, 1)),
        ((10, 10), (0, 0)),
    ],
)
def test_mixed_schedule_counts_are_bounded(start_steps, expected):
    assert _mixed_schedule_counts(start_steps, terminal_step=10) == expected
