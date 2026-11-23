"""固定所有 UAV 的切分点，每轮优化计算频率、上行功率及上下行带宽。"""
from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_v2.uav_sfl import make_problem, solve_resources
from code_v4.experiment import EpochDecision, EpochState, run_baseline


def choose_epoch(state: EpochState) -> EpochDecision:
    next_split = state.current if state.choosing_next else None
    problem = make_problem(state.epoch, state.count, state.gain, state.current, next_split,
                           state.remaining, state.config)
    result = solve_resources(problem, state.profile, state.config, state.reference, retain_feasible=True)
    return EpochDecision(problem, result["resources"], result["status"], solve=result)


if __name__ == "__main__":
    raise SystemExit(run_baseline("fixed_split", choose_epoch, __doc__, fixed_split=True))
