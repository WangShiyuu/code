"""在 CUDA 上训练随机切分点与随机资源基线。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_v2.uav_sfl import (
    SPLIT_POINTS, audit_resources, evaluate_resources, make_problem, maximum_frequency, uav_parameter,
)
from code_v4.experiment import EpochDecision, EpochState, run_baseline


MAX_ATTEMPTS = 2048


def draw_resources(state: EpochState, frequency_cap: np.ndarray, power_cap: np.ndarray) -> dict:
    """独立采样频率、功率和两个方向的带宽分配。"""
    rng, radio = state.rng, state.config["radio"]
    # 频率和功率分别为各机上限的 50%–100% 与 10%–100%。
    # 两个方向独立从 Dirichlet(1, …, 1) 均匀抽取带宽份额，各自用满预算。
    concentration = np.ones(len(frequency_cap))
    return {
        "frequency_hz": frequency_cap * rng.uniform(0.5, 1.0, len(frequency_cap)),
        "power_w": power_cap * rng.uniform(0.1, 1.0, len(power_cap)),
        "up_hz": radio["total_up_hz"] * rng.dirichlet(concentration),
        "down_hz": radio["total_down_hz"] * rng.dirichlet(concentration),
    }


def feasible_commitment(state: EpochState, problem: dict, resources: dict) -> bool:
    """按当前快照审计已抽出的切点在整个承诺周期内的预测可行性。"""
    if not audit_resources(problem, resources, state.profile, state.config)["ok"]:
        return False
    if state.epoch != 1 and not state.choosing_next:
        return True

    planned_epochs = state.config["training"]["epochs"]
    active = problem["next_split"] if state.choosing_next else problem["split"]
    end = state.epoch + (2 if state.choosing_next else 1)
    remaining = state.remaining - evaluate_resources(problem, resources, state.profile, state.config)["uav_energy_j"]
    for epoch in range(state.epoch + 1, end + 1):
        # 预测沿用当前样本数和信道。预测窗口末端按现切点估算前缀下发；
        # 后续实际轮次仍使用各自快照重新抽取资源，聚合时再抽下一周期切点。
        next_split = active if epoch % 2 == 0 and epoch < planned_epochs else None
        forecast = make_problem(epoch, state.count, state.gain, active, next_split, remaining, state.config)
        if not audit_resources(forecast, resources, state.profile, state.config)["ok"]:
            return False
        remaining -= evaluate_resources(forecast, resources, state.profile, state.config)["uav_energy_j"]
    return True


def choose_epoch(state: EpochState) -> EpochDecision:
    """接收首个通过审计的随机方案，耗尽采样次数后报告本轮失败。"""
    frequency_cap = maximum_frequency(state.config)
    power_cap = uav_parameter(state.config, "power_max_w")
    problem = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        current, next_split = state.current, None
        if state.epoch == 1:
            current = state.rng.choice(SPLIT_POINTS, size=len(current))
        elif state.choosing_next:
            next_split = state.rng.choice(SPLIT_POINTS, size=len(current))
        problem = make_problem(state.epoch, state.count, state.gain, current, next_split, state.remaining, state.config)
        resources = draw_resources(state, frequency_cap, power_cap)
        if feasible_commitment(state, problem, resources):
            return EpochDecision(problem, resources, "random_feasible", selection={"proposals": attempt})
    return EpochDecision(problem, None, "random_feasible_sample_not_found", selection={"proposals": MAX_ATTEMPTS})


if __name__ == "__main__":
    raise SystemExit(run_baseline("random_split_resources", choose_epoch, __doc__))
