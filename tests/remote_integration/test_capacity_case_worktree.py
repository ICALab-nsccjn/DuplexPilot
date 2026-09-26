from pathlib import Path


def test_capacity_case_wrapper_defaults_to_its_own_checkout():
    wrapper = (
        Path(__file__).parents[1]
        / "tools"
        / "apr"
        / "run_capacity_slo_online_case.sh"
    )
    text = wrapper.read_text(encoding="utf-8")

    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in text
    assert (
        'WORKTREE="${DUPLEXPILOT_WORKTREE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"'
        in text
    )
    assert "apr-capacity-slo-state-v2" not in text
