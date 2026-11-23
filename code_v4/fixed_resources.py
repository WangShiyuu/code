"""固定各机资源，每个聚合周期在窗口信道下选择切分点。"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_v2.uav_sfl import (
    SPLIT_POINTS,
    audit_resources,
    evaluate_resources,
    make_problem,
    reference_resources,
)
from code_v4.experiment import EpochDecision, EpochState, run_baseline


def forecast_energy(
    state: EpochState, cuts: tuple[int, ...], gain: np.ndarray, resources: dict,
) -> float | None:
    """用固定样本数和窗口信道预测能耗，逐轮扣减各机电量。"""
    config = state.config
    epochs = config["training"]["epochs"]
    last = 2 if state.epoch == 1 else state.epoch + 2
    battery = state.remaining.copy()
    energy = 0.0
    for epoch in range(state.epoch, last + 1):
        # 预选发生在聚合前，本轮仍用原切点训练，聚合后下发新前缀。
        active = state.current if epoch == state.epoch and state.choosing_next else cuts
        following = cuts if epoch % 2 == 0 and epoch < epochs else None
        problem = make_problem(epoch, state.count, gain, active, following, battery, config)
        if not audit_resources(problem, resources, state.profile, config)["ok"]:
            return None
        metrics = evaluate_resources(problem, resources, state.profile, config)
        battery -= metrics["uav_energy_j"]
        energy += metrics["total_energy_j"]
    return energy


def choose_epoch(state: EpochState) -> EpochDecision:
    """资源始终为同一组参考值，在当前信道复核窗口排名中的候选。"""
    config = state.config
    resources = reference_resources(config)
    if state.epoch != 1 and not state.choosing_next:
        problem = make_problem(
            state.epoch, state.count, state.gain, state.current, None, state.remaining, config,
        )
        return EpochDecision(problem, resources, "fixed_resources")

    window = config["optimization"]["window_epochs"]
    gains = state.gain_history[-window:]
    window_gain = np.mean(gains, axis=0) if len(gains) else state.gain
    current = tuple(state.current)
    num_uavs = config["num_uavs"]
    ranked = []
    for cuts in product(SPLIT_POINTS, repeat=num_uavs):
        energy = forecast_energy(state, cuts, window_gain, resources)
        if energy is not None:
            ranked.append((energy, cuts != current, cuts))

    ranked.sort()
    selection = {
        "combinations_evaluated": len(SPLIT_POINTS) ** num_uavs,
        "feasible_combinations": len(ranked),
        "window_gain": window_gain,
        "candidates": [cuts for _, _, cuts in ranked],
        "forecast_energy_j": [energy for energy, _, _ in ranked],
    }
    for attempt, (_, _, cuts) in enumerate(ranked, start=1):
        active = state.current if state.choosing_next else cuts
        following = cuts if state.choosing_next else None
        problem = make_problem(
            state.epoch, state.count, state.gain, active, following, state.remaining, config,
        )
        if audit_resources(problem, resources, state.profile, config)["ok"]:
            selection["current_candidates_checked"] = attempt
            return EpochDecision(problem, resources, "fixed_resources", selection=selection)
    status = "fixed_resources_current_infeasible" if ranked else "fixed_resources_window_infeasible"
    selection["current_candidates_checked"] = len(ranked)
    return EpochDecision(None, None, status, selection=selection)


if __name__ == "__main__":
    raise SystemExit(run_baseline("fixed_resources", choose_epoch, __doc__))
