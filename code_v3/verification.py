"""UAV-SFL 协议验证：梯度、样本释放、通信、资源约束及切分排序。"""
from __future__ import annotations

from itertools import product
import math
from typing import Any

import numpy as np

from .config import (
    BLOCKS_PER_GROUP,
    SPLIT_POINTS,
    default_config,
    load_config,
    merge_config,
    training_budget,
    uav_parameter,
)
from .optimization import (
    audit_resources,
    channel_at,
    convex_subproblem,
    evaluate_resources,
    grouped_profile,
    make_problem,
    maximum_frequency,
    patrol_geometry,
    patrol_positions,
    rebuild_reference,
    reference_resources,
    reference_scores,
    select_splits,
    solve_resources,
)
from .training import (
    aggregate_models,
    epoch_snapshot,
    forward_prefix,
    forward_suffix,
    initialize_model,
    load_batch,
    prepare_data,
    profile_splits,
    sample_release_schedule,
    sgd_step,
    split_gradients,
    train_client_epoch,
)


# 8. 有意义的协议验证：五切点梯度、尾批、聚合通信、可行性与能量下降 ---------------
def _verification_config(config: dict):
    """返回协议验证所用的小规模独立配置。"""
    c = load_config(updates={"num_uavs": 2, "data": {"kind": "synthetic", "image_size": 32,
                            "synthetic_per_uav": 5, "synthetic_eval_size": 3,
                            "initial_samples_per_uav": None, "initial_fraction": 0.6},
                            "training": {"epochs": 4, "batch_size": 2, "allow_random_init": True, "save_checkpoint": False},
                            "uav": {"kappa": 1e-28, "compute_power_max_w": 0.3375, "power_max_w": 1.0,
                                    "cycles_per_flop": 0.5, "energy_cap_j": 200000.0},
                            "optimization": {"max_iterations": 15, "split_objective": "reference_score"},
                            "output_root": config["output_root"]})
    return c


def _verify_split_gradients(model: dict, x: Any, y: Any):
    """核对五个切点的梯度与完整网络梯度，并核对单步更新。"""
    import torch
    from torch.nn import functional as F
    names = list(model["params"])
    logits = forward_suffix(model, forward_prefix(model, x, 0), 0)
    grads = dict(zip(names, torch.autograd.grad(F.cross_entropy(logits, y), list(model["params"].values()))))
    errors = []
    for split in SPLIT_POINTS:
        split_grads, _ = split_gradients(model, x, y, split)
        errors.append(max(float((grads[name] - split_grads[name]).abs().max()) for name in names))
        for name in names:
            torch.testing.assert_close(grads[name], split_grads[name], rtol=2e-5, atol=2e-6)
        stepped = sgd_step(model, split_grads, 0.01)
        for name in names:
            torch.testing.assert_close(stepped["params"][name], model["params"][name] - 0.01 * grads[name], rtol=2e-5, atol=2e-6)
    return errors


def _verify_training_and_aggregation(model: dict, data: dict, c: dict):
    """核对尾批累计步长、输入不变性和按样本数加权聚合。"""
    import torch
    names = list(model["params"])
    original = {name: value.detach().clone() for name, value in model["params"].items()}
    updated, stats = train_client_epoch(model, data, data["clients"][0][:3], 4, 1, 0, c)
    assert math.isclose(stats["step_sum"], c["training"]["epoch_step_size"], rel_tol=1e-12)
    assert any(not torch.equal(updated["params"][name], model["params"][name]) for name in names)
    assert all(torch.equal(original[name], model["params"][name]) for name in names)
    average = aggregate_models([model, updated], [1, 3])
    for name in names:
        torch.testing.assert_close(average["params"][name], model["params"][name] * 0.25 + updated["params"][name] * 0.75)


def _verify_release_and_partitions(data: dict, c: dict):
    """核对轮初样本冻结、周期与旧版释放计划以及集合互斥。"""
    snap = epoch_snapshot(data, data["start_s"], 1, c)
    assert snap["count"] == 3
    later = epoch_snapshot(data, data["release_s"][3], 2, c)
    assert later["count"] == 4 and snap["count"] == 3
    # 检查首次采样、采样间隙、末批截断，以及长任务的自动间隔。
    patrol_config = default_config()
    release, plan = sample_release_schedule(5552, patrol_config)
    initial = plan["initial_count"]
    assert np.count_nonzero(release == 0) == initial
    np.testing.assert_allclose(np.diff(release[initial - 1:]), plan["sample_interval_s"])
    assert math.isclose(release[-1], plan["target_s"])
    manual = merge_config(patrol_config, {"data": {"initial_samples_per_uav": None,
                         "sample_interval_s": 5, "samples_per_capture": 2}})
    manual_release, _ = sample_release_schedule(6, manual)
    np.testing.assert_array_equal(manual_release, [0, 5, 5, 10, 10, 15])
    np.testing.assert_array_equal(np.searchsorted(manual_release, [4.9, 5, 9.9, 10, 20], side="right"),
                                  [1, 3, 3, 5, 6])
    assert np.searchsorted(release, 9 * 600, side="right") < 0.2 * len(release)
    legacy_release, _ = sample_release_schedule(5, merge_config(c, {"data": {"rate_per_s": 1.0}}))
    np.testing.assert_array_equal(legacy_release, [1, 2, 3, 4, 5])
    groups = [set(ids) for ids in data["clients"]] + [set(data["val"]), set(data["test"])]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])


def _verify_heterogeneous_resources(profile: dict, h: np.ndarray, c: dict):
    """核对逐机画像、硬件上限、旧画像兼容性和固定功率可行性。"""
    heterogeneous = load_config(updates={"num_uavs": 2,
        "uav": {"kappa": [1e-28, 1.4e-28], "compute_power_max_w": [0.1, 0.3375],
                "power_max_w": [0.3, 0.8], "cycles_per_flop": [1.0, 0.5], "energy_cap_j": [30.0, 40.0]}})
    hetero_profile = {**profile, "client_cycles": np.array([[2.0], [1.0]]) * profile["client_cycles"]}
    # 每机画像行与自身硬件对应；共享旧画像仍可直接读取。
    legacy = {key: np.arange(1.0, 10.0) for key in
              ("client_cycles", "server_cycles", "activation_bits", "gradient_bits", "prefix_bits")}
    np.testing.assert_array_equal(grouped_profile(legacy)["client_cycles"], [1, 3, 5, 7, 9])
    legacy["client_cycles"] = np.vstack([np.arange(1.0, 10.0), np.arange(1.0, 10.0) * 2])
    np.testing.assert_array_equal(grouped_profile(legacy)["client_cycles"], [[1, 3, 5, 7, 9], [2, 6, 10, 14, 18]])
    hetero_problem = make_problem(1, 3, h, np.array([4, 6]), None, np.array([30.0, 40.0]), heterogeneous)
    hetero_resource = reference_resources(heterogeneous)
    metrics = evaluate_resources(hetero_problem, hetero_resource, hetero_profile, heterogeneous)
    np.testing.assert_allclose(metrics["loads"]["client_cycles"],
                               3 * profile["client_cycles"][[2, 3]] * [2, 1])
    for key, bounds in (("frequency_hz", maximum_frequency(heterogeneous)),
                        ("power_w", uav_parameter(heterogeneous, "power_max_w"))):
        invalid = {k: v.copy() for k, v in hetero_resource.items()}
        invalid[key][0] = bounds[0] * 1.05
        audit = audit_resources(hetero_problem, invalid, hetero_profile, heterogeneous)
        assert not audit["ok"] and math.isclose(audit["normalized_violation"], 0.05)
    phase = convex_subproblem(hetero_problem, hetero_profile, heterogeneous, fixed_power=0.3)
    assert phase["ok"] and audit_resources(hetero_problem, phase["resources"], hetero_profile, heterogeneous)["ok"]
    assert reference_scores(3, h, hetero_resource, hetero_profile, True, heterogeneous).shape == (2, 5)


def _verify_stage_costs_and_solver(profile: dict, h: np.ndarray, remaining: np.ndarray, reference: dict, c: dict):
    """核对各轮阶段时延、同步载荷和求解后的可行性与能量下降。"""
    results = []
    for epoch, next_split in ((1, None), (2, np.array([8, 2])), (4, None)):
        problem = make_problem(epoch, 3, h, np.array([0, 4]), next_split, remaining, c)
        m = evaluate_resources(problem, reference, profile, c)
        mu = profile["prefix_bits"][problem["split"] // BLOCKS_PER_GROUP] / m["up_rate_bps"] if problem["aggregate"] else np.zeros(2)
        md = profile["prefix_bits"][next_split // BLOCKS_PER_GROUP] / m["down_rate_bps"] if next_split is not None else np.zeros(2)
        aggregate_time = c["server"]["aggregation_time_s"] if problem["aggregate"] else 0.0
        correct_duration = float(np.max(m["branch_training_s"]) + np.max(mu) + np.max(md) + aggregate_time)
        assert math.isclose(m["duration_s"], correct_duration, rel_tol=1e-12)
        if next_split is not None:
            wrong_combined_max = float(np.max(m["branch_training_s"] + mu + md) + aggregate_time)
            assert correct_duration > wrong_combined_max + 1e-3
        assert (m["loads"]["download_bits"].sum() == 0) == (next_split is None)
        if next_split is not None:
            np.testing.assert_array_equal(m["loads"]["download_bits"], profile["prefix_bits"][next_split // BLOCKS_PER_GROUP])
        solution = solve_resources(problem, profile, c, reference)
        assert solution["ok"], solution
        assert audit_resources(problem, solution["resources"], profile, c)["ok"]
        accepted = [r["energy_j"] for r in solution["trace"] if r.get("accepted", True)]
        assert all(new <= old for old, new in zip(accepted, accepted[1:]))
        results.append({"epoch": epoch, "status": solution["status"], "energy_j": solution["metrics"]["total_energy_j"]})
    return results


def _verify_infeasibility_and_battery(profile: dict, h: np.ndarray, remaining: np.ndarray, reference: dict, c: dict):
    """核对不可能任务被拒绝，以及收紧电池时的约束求解。"""
    # 低电池与不可能期限必须拒绝，不能静默返回参考资源。
    problem = make_problem(1, 3, h, np.array([0, 4]), None, np.zeros(2), c)
    assert not audit_resources(problem, reference, profile, c)["ok"]
    impossible = merge_config(c, {"task": {"epoch_deadline_s": 1e-6}})
    rejected = solve_resources(problem, profile, impossible)
    assert not rejected["ok"]
    # 仅收紧电量，让初始参考资源失效，验证电池约束确实进入凸求解。
    p = make_problem(1, 3, h, np.array([0, 4]), None, remaining, c)
    reference_energy = evaluate_resources(p, reference, profile, c)["uav_energy_j"]
    tight = make_problem(1, 3, h, np.array([0, 4]), None, reference_energy * 0.6, c)
    assert not audit_resources(tight, reference, profile, c)["ok"]
    tight_solution = solve_resources(tight, profile, c)
    assert tight_solution["ok"], tight_solution
    assert audit_resources(tight, tight_solution["resources"], profile, c)["ok"]


def _verify_reference_recovery_and_ranking(profile: dict, h: np.ndarray, remaining: np.ndarray, c: dict):
    """核对严格参考失败、显式参考恢复和全部切分组合排序。"""
    # 使用一个过慢的固定参考，严格 P1 应拒绝，显式重建可恢复参考可行性。
    slow = reference_resources(c)
    slow["frequency_hz"][:] = 1000.0
    selection = select_splits(2, 3, np.array([0, 4]), [h], h, remaining, slow, profile, c)
    assert not selection["ok"] and selection["status"] == "no_reference_feasible_split"
    rebuilt = rebuild_reference(2, 3, np.array([0, 4]), h, remaining, profile, c)
    assert rebuilt["ok"]
    selection = select_splits(2, 3, np.array([0, 4]), [h], h, remaining, rebuilt["resources"], profile, c)
    assert selection["ok"]
    assert selection["combinations_evaluated"] == len(SPLIT_POINTS) ** c["num_uavs"]
    assert {tuple(row["vector"]) for row in selection["trace"]} == set(product(SPLIT_POINTS, repeat=c["num_uavs"]))
    ranked = [tuple(v) for v in selection["candidates"]]
    feasible_rows = [row for row in selection["trace"] if row["feasible"]]
    assert selection["scores_j"].shape == (c["num_uavs"], len(SPLIT_POINTS))
    expected = sorted(feasible_rows, key=lambda row: (row["score_j"], row["vector"] != (0, 4), row["vector"]))
    assert ranked == [row["vector"] for row in expected]


def _verify_budget_and_channel(c: dict):
    """核对训练预算取整和信道计算的确定性。"""
    assert training_budget(default_config())["conditional_epochs"] == 100
    # 相同原始轨迹时刻必须产生完全相同的信道，不依赖调用顺序。
    np.testing.assert_array_equal(channel_at(20, c)["gain"], channel_at(20, c)["gain"])


def _verify_absolute_initial_samples(config: dict) -> dict:
    """独立核对默认500优先于比例，以及首次新增和全量释放的快照边界。"""
    release, plan = sample_release_schedule(5552, config)
    assert plan["initial_count"] == 500
    assert np.count_nonzero(release == 0) == 500
    assert release[499] == 0 < release[500]
    data = {"release_s": release, "capacity": 5552,
            "clients": [list(range(5552)) for _ in range(4)]}
    initial = epoch_snapshot(data, 0.0, 1, config)
    before = epoch_snapshot(data, float(np.nextafter(release[500], -np.inf)), 2, config)
    after = epoch_snapshot(data, float(release[500]), 2, config)
    final = epoch_snapshot(data, float(release[-1]), 120, config)
    assert initial["count"] == before["count"] == 500
    assert after["count"] == 501 and final["count"] == 5552
    assert all(len(indices) == 500 for indices in initial["indices"])
    full_release, full_plan = sample_release_schedule(500, config)
    np.testing.assert_array_equal(full_release, np.zeros(500))
    assert full_plan["initial_count"] == 500
    # 旧恒速模式不提前释放500个样本；启动等待仍由第500个到达时刻决定。
    rate_config = merge_config(config, {"data": {"rate_per_s": 2.0}})
    rate_release, rate_plan = sample_release_schedule(501, rate_config)
    assert rate_plan["initial_count"] == 500 and rate_release[499] == 250.0
    return {"initial_count": initial["count"], "capacity": 5552,
            "first_new_sample_s": float(release[500]), "full_release_s": float(release[-1])}


def _verify_initial_capacity_and_small_data(config: dict):
    """不足500的正式容量应拒绝；显式回退比例的小合成数据仍能准备。"""
    for capacity in (499, 0, -1, 1.5, True):
        try:
            sample_release_schedule(capacity, config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法或小于默认初始量的容量未被拒绝: {capacity!r}")
    small = load_config(updates={"data": {"kind": "synthetic", "initial_samples_per_uav": None,
        "initial_fraction": 0.6, "synthetic_per_uav": 5, "synthetic_eval_size": 3},
        "training": {"epochs": 4}})
    data = prepare_data(small)
    assert data["capacity"] == 5 and data["release_plan"]["initial_count"] == 3
    assert epoch_snapshot(data, data["start_s"], 1, small)["count"] == 3
    assert default_config(2)["trajectory"]["kind"] == "circle"


def _verify_serpentine_lengths_and_period(config: dict, geometry: dict) -> dict:
    """以独立解析值及黑箱积分核对四机闭合长度和共同周期。"""
    # 六段480m横扫、五段84m连接、420m闭合；12个R10圆角替换直角。
    expected_length = 3480.0 + 60.0 * math.pi
    # 直线峰值10、转弯5、切向加速度2：24次速度过渡额外耗时15s。
    expected_period = 348.0 + 12.0 * math.pi + 15.0
    assert math.isclose(geometry["raw_length_m"], 3720.0, rel_tol=1e-12)
    assert math.isclose(geometry["length_m"], expected_length, rel_tol=1e-12)
    assert math.isclose(geometry["period_s"], expected_period, rel_tol=1e-12)
    period = geometry["period_s"]
    for t in (0.0, 0.017 * period, 0.41 * period, period - 1e-4):
        np.testing.assert_allclose(patrol_positions(t, config), patrol_positions(t + period, config),
                                   rtol=0, atol=1e-8)
    np.testing.assert_allclose(patrol_positions(-1e-4, config),
                               patrol_positions(period - 1e-4, config), rtol=0, atol=1e-8)
    times = np.linspace(0.0, period, 4097)
    positions = np.stack([patrol_positions(float(t), config) for t in times])
    assert positions.shape == (4097, 4, 3) and np.isfinite(positions).all()
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=2).sum(axis=0)
    # 有限采样的折线略短于圆弧；容差只用于此独立数值积分。
    np.testing.assert_allclose(lengths, expected_length, rtol=0, atol=0.15)
    np.testing.assert_allclose(lengths, lengths[0], rtol=0, atol=0.02)
    return {"raw_length_m": 3720.0, "rounded_length_m": expected_length,
            "period_s": expected_period, "sampled_lengths_m": lengths.tolist()}


def _verify_serpentine_rotation_and_phase(config: dict, geometry: dict):
    """核对象限与同高边界、旋转关系，以及错相只打破同步的信道序列。"""
    period = geometry["period_s"]
    zero_phase = merge_config(config, {"trajectory": {"phase_offsets_fraction": [0.0] * 4}})
    phase = config["trajectory"]["phase_offsets_fraction"]
    assert phase == [0.0, 0.25, 0.5, 0.75]
    gains = []
    for t in np.linspace(0.0, period, 129):
        positions = patrol_positions(float(t), config)
        assert np.all(np.abs(positions[:, :2]) <= 500.0 + 1e-8)
        np.testing.assert_allclose(positions[:, 2], 100.0, rtol=0, atol=1e-10)
        base = patrol_positions(float(t), zero_phase)[0, :2]
        x, y = base
        rotations = np.array([[x, y], [-y, x], [-x, -y], [y, -x]])
        np.testing.assert_allclose(patrol_positions(float(t), zero_phase)[:, :2],
                                   rotations, rtol=0, atol=1e-8)
        for uid in range(4):
            shifted = patrol_positions(float(t + phase[uid] * period), zero_phase)
            np.testing.assert_allclose(positions[uid], shifted[uid], rtol=0, atol=1e-8)
        inverse_xy = np.array([[positions[0, 0], positions[0, 1]],
                               [positions[1, 1], -positions[1, 0]],
                               [-positions[2, 0], -positions[2, 1]],
                               [-positions[3, 1], positions[3, 0]]])
        assert np.all((inverse_xy[:, 0] >= 10.0 - 1e-8) & (inverse_xy[:, 0] <= 490.0 + 1e-8))
        assert np.all((inverse_xy[:, 1] >= 40.0 - 1e-8) & (inverse_xy[:, 1] <= 460.0 + 1e-8))
        for a in range(4):
            for b in range(a):
                assert np.linalg.norm(positions[a] - positions[b]) >= 50.0 - 1e-8
        gains.append(channel_at(float(t), config)["gain"])
    gains = np.stack(gains)
    for a in range(4):
        for b in range(a):
            assert not np.allclose(gains[:, a], gains[:, b], rtol=1e-7, atol=0)
    same_phase_gain = channel_at(0.173 * period, zero_phase)["gain"]
    np.testing.assert_allclose(same_phase_gain, same_phase_gain[0], rtol=1e-12, atol=0)


def _verify_serpentine_rounded_motion(config: dict, geometry: dict):
    """在全部线弧接点和周期接缝两侧检查位置与速度连续、转弯半径和速度。"""
    eps = 1e-4
    segments = geometry["segments"]
    np.testing.assert_allclose(segments[-1]["end_xy_m"], segments[0]["start_xy_m"], atol=1e-10)
    for uid, phase_s in enumerate(geometry["phase_offsets_s"]):
        for join_s in [0.0] + [segment["end_time_s"] for segment in segments]:
            t = join_s - phase_s
            before = patrol_positions(t - eps, config)[uid]
            at = patrol_positions(t, config)[uid]
            after = patrol_positions(t + eps, config)[uid]
            left_velocity, right_velocity = (at - before) / eps, (after - at) / eps
            assert np.isfinite(left_velocity).all() and np.isfinite(right_velocity).all()
            assert np.linalg.norm(left_velocity) <= 10.0 + 1e-5
            assert np.linalg.norm(right_velocity) <= 10.0 + 1e-5
            assert np.linalg.norm(left_velocity - right_velocity) < 0.01
    zero_phase = merge_config(config, {"trajectory": {"phase_offsets_fraction": [0.0] * 4}})
    arcs = [segment for segment in segments if "radius_m" in segment]
    assert len(arcs) == 12
    for arc in arcs:
        t = (arc["start_time_s"] + arc["end_time_s"]) / 2
        position = patrol_positions(t, zero_phase)[0, :2]
        velocity = (patrol_positions(t + eps, zero_phase)[0, :2]
                    - patrol_positions(t - eps, zero_phase)[0, :2]) / (2 * eps)
        radial = position - arc["center_xy_m"]
        assert math.isclose(float(np.linalg.norm(radial)), 10.0, rel_tol=1e-9)
        assert math.isclose(float(np.linalg.norm(velocity)), 5.0, rel_tol=1e-7)
        assert abs(float(np.dot(radial, velocity))) < 1e-5


def _verify_default_patrol_and_initial_samples() -> tuple[dict, dict]:
    """验证新默认场景；独立于用于原21项检查的两机圆轨迹配置。"""
    config = load_config()
    assert config["trajectory"]["kind"] == "serpentine_equal"
    geometry = patrol_geometry(config)
    trajectory_report = _verify_serpentine_lengths_and_period(config, geometry)
    _verify_serpentine_rotation_and_phase(config, geometry)
    _verify_serpentine_rounded_motion(config, geometry)
    release_report = _verify_absolute_initial_samples(config)
    _verify_initial_capacity_and_small_data(config)
    return trajectory_report, release_report


def verify_implementation(config: dict) -> dict:
    """依次执行小型 CPU 协议验证；失败抛 AssertionError，不作为实验准确率。"""
    import torch
    c = _verification_config(config)
    torch.set_num_threads(c["training"]["torch_threads"])
    model, _ = initialize_model(c)
    data = prepare_data(c)
    x, y = load_batch(data, data["clients"][0][:2], "cpu")
    errors = _verify_split_gradients(model, x, y)
    _verify_training_and_aggregation(model, data, c)
    _verify_release_and_partitions(data, c)
    profile = profile_splits(model, c)
    remaining = np.full(2, c["uav"]["energy_cap_j"])
    h = channel_at(data["start_s"], c)["gain"]
    reference = reference_resources(c)
    _verify_heterogeneous_resources(profile, h, c)
    results = _verify_stage_costs_and_solver(profile, h, remaining, reference, c)
    _verify_infeasibility_and_battery(profile, h, remaining, reference, c)
    _verify_reference_recovery_and_ranking(profile, h, remaining, c)
    _verify_budget_and_channel(c)
    trajectory_report, release_report = _verify_default_patrol_and_initial_samples()
    return {"passed": True, "verification_config": c, "split_gradient_max_abs_errors": errors,
            "checks": ["five_split_gradients_and_updates", "exhaustive_split_ranking", "input_immutability", "tail_batch_step_sum",
                       "weighted_aggregation", "frozen_epoch_snapshot", "periodic_capture_release", "legacy_rate_release", "disjoint_partitions",
                       "separate_stage_maxima", "new_prefix_payload", "final_zero_download",
                       "convex_original_feasibility", "accepted_energy_monotonicity", "infeasible_rejection",
                       "binding_battery_constraint", "strict_reference_failure_and_explicit_rebuild", "budget_rounding",
                       "per_uav_cycles_and_limits", "per_uav_phase_one", "legacy_and_grouped_profiles",
                       "serpentine_equal_length_period_and_wrap", "serpentine_rotation_phase_and_quadrants",
                       "serpentine_rounded_motion_continuity", "absolute_initial_500_snapshot_boundaries",
                       "initial_capacity_and_small_synthetic_compatibility"],
            "solver_cases": results, "trajectory_verification": trajectory_report,
            "initial_sample_verification": release_report}
