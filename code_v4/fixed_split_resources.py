"""固定所有 UAV 的切分点和参考资源，作为联合优化的对照。"""
from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_v3.optimization import make_problem, reference_resources
from code_v4.experiment import EpochDecision, EpochState, run_baseline


def choose_epoch(state: EpochState) -> EpochDecision:
    next_split = state.current if state.choosing_next else None
    problem = make_problem(state.epoch, state.count, state.gain, state.current, next_split,
                           state.remaining, state.config)
    return EpochDecision(problem, reference_resources(state.config), "fixed_split_resources")


if __name__ == "__main__":
    raise SystemExit(run_baseline("fixed_split_resources", choose_epoch, __doc__, fixed_split=True))
