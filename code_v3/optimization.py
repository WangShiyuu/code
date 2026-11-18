"""UAV-SFL 的信道、物理成本、切分决策与连续资源优化。

本模块只处理 NumPy 数组和字典，不导入训练模块或初始化神经网络。
CVXPY 在实际求解时按需导入；供 Windows 进程池调用的函数保留在模块顶层。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
from functools import lru_cache, partial
from itertools import product
import time
from typing import Any
import warnings

import numpy as np

from .config import BLOCKS, BLOCKS_PER_GROUP, SPLIT_POINTS, uav_parameter


def patrol_radii(config: dict) -> np.ndarray:
    trajectory = config["trajectory"]
    return np.linspace(trajectory["radius_min_m"], trajectory["radius_max_m"], config["num_uavs"])


def _patrol_line(start: np.ndarray, end: np.ndarray, speed: float,
                 turn_speed: float, acceleration: float) -> dict:
    """构造两端为转弯速度、可升至巡航速度的直线段及解析速度剖面。"""
    length = float(np.linalg.norm(end - start))
    ramp_distance = (speed ** 2 - turn_speed ** 2) / (2 * acceleration)
    if length <= 0 or length + 1e-10 < 2 * ramp_distance:
        raise ValueError("圆角后的每段直线必须为正且足以完成加速与减速")
    ramp_time = (speed - turn_speed) / acceleration
    cruise_distance = max(0.0, length - 2 * ramp_distance)
    cruise_time = cruise_distance / speed
    return {"kind": "line", "start_xy_m": start.tolist(), "end_xy_m": end.tolist(),
            "length_m": length, "duration_s": 2 * ramp_time + cruise_time,
            "direction_xy": ((end - start) / length).tolist(),
            "start_speed_m_s": turn_speed, "peak_speed_m_s": speed,
            "acceleration_m_s2": acceleration, "accel_distance_m": ramp_distance,
            "cruise_distance_m": cruise_distance, "accel_time_s": ramp_time,
            "cruise_time_s": cruise_time}


@lru_cache(maxsize=32)
def _serpentine_template(region_size: float, rows: int, edge: float, cross: float,
                          radius: float, speed: float, turn_speed: float,
                          acceleration: float) -> dict:
    """按轨迹几何数值缓存 NE 模板；缓存仅内部只读，公开接口返回独立副本。"""
    half = region_size / 2
    ys = np.linspace(cross, half - cross, rows)
    vertices = [np.array([edge, ys[0]])]
    for index, y in enumerate(ys):
        x = half - edge if index % 2 == 0 else edge
        vertices.append(np.array([x, y]))
        if index + 1 < rows:
            vertices.append(np.array([x, ys[index + 1]]))
    entries, exits, centers, signs = [], [], [], []
    raw_length = 0.0
    for index, vertex in enumerate(vertices):
        previous, following = vertices[index - 1], vertices[(index + 1) % len(vertices)]
        incoming, outgoing = vertex - previous, following - vertex
        raw_length += float(np.linalg.norm(outgoing))
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        if not np.isclose(np.dot(incoming, outgoing), 0.0, atol=1e-12):
            raise ValueError("蛇形闭合路线的相邻边必须垂直")
        entries.append(vertex - radius * incoming)
        exits.append(vertex + radius * outgoing)
        centers.append(entries[-1] + radius * outgoing)
        signs.append(float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]))
    segments, elapsed = [], 0.0
    # 周期起点为第一个圆角的出切点，默认 (20,40)，不是已切除的 (10,40)。
    for index in range(len(vertices)):
        corner = (index + 1) % len(vertices)
        line = _patrol_line(exits[index], entries[corner], speed, turn_speed, acceleration)
        arc = {"kind": "arc", "start_xy_m": entries[corner].tolist(),
               "end_xy_m": exits[corner].tolist(), "center_xy_m": centers[corner].tolist(),
               "radius_m": radius, "length_m": np.pi * radius / 2,
               "duration_s": np.pi * radius / (2 * turn_speed),
               "start_angle_rad": float(np.arctan2(*(entries[corner] - centers[corner])[::-1])),
               "sweep_angle_rad": signs[corner] * np.pi / 2, "speed_m_s": turn_speed}
        for segment in (line, arc):
            segment["start_time_s"] = elapsed
            elapsed += segment["duration_s"]
            segment["end_time_s"] = elapsed
            segments.append(segment)
    return {"kind": "serpentine_equal", "segments": segments,
            "vertices_xy_m": [v.tolist() for v in vertices],
            "start_xy_m": exits[0].tolist(), "raw_length_m": raw_length,
            "length_m": sum(s["length_m"] for s in segments), "period_s": elapsed,
            "turn_count": len(vertices)}


def _patrol_template(config: dict) -> dict:
    """读取几何值组成缓存键；位置、服务器、时间和其他配置不会污染缓存。"""
    t = config["trajectory"]
    rows = t.get("scan_rows", 6)
    if config["num_uavs"] != 4 or isinstance(rows, bool) or not isinstance(rows, (int, np.integer)) or rows < 2 or rows % 2:
        raise ValueError("serpentine_equal 要求四架 UAV 且 scan_rows 为至少 2 的偶数")
    values = [float(t.get(key, default)) for key, default in (
        ("region_size_m", 1000.0), ("edge_margin_m", 10.0), ("cross_margin_m", 40.0),
        ("turn_radius_m", 10.0), ("speed_m_s", 10.0), ("turn_speed_m_s", 5.0),
        ("tangential_acceleration_m_s2", 2.0))]
    size, edge, cross, radius, speed, turn_speed, acceleration = values
    if not np.isfinite(values).all() or min(size, radius, speed, turn_speed, acceleration) <= 0 or min(edge, cross) < 0:
        raise ValueError("蛇形区域、圆角、速度和加速度必须有限为正，边距必须有限非负")
    if turn_speed > speed:
        raise ValueError("turn_speed_m_s 不能大于 speed_m_s")
    straight_lengths = [size / 2 - 2 * edge - 2 * radius,
                        (size / 2 - 2 * cross) / (rows - 1) - 2 * radius,
                        size / 2 - 2 * cross - 2 * radius]
    if min(straight_lengths) <= 0:
        raise ValueError("扫描区域及行距不足以容纳圆角")
    return _serpentine_template(size, int(rows), edge, cross, radius, speed, turn_speed, acceleration)


def _patrol_frame(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回与服务器坐标独立的区域中心，以及四机的周期时间相位比例。"""
    t = config["trajectory"]
    center = np.asarray(t.get("center_xy_m", [0.0, 0.0]), dtype=float)
    phases = np.asarray(t.get("phase_offsets_fraction", [0.0, 0.25, 0.5, 0.75]), dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy_m 必须是有限二维坐标")
    if phases.shape != (4,) or not np.isfinite(phases).all() or np.any((phases < 0) | (phases >= 1)):
        raise ValueError("phase_offsets_fraction 必须为四个 [0,1) 内的有限数")
    return center, phases


def patrol_geometry(config: dict) -> dict:
    """返回独立轨迹几何；蛇形含模板线/弧段和共同周期，圆形含各机周长/周期。

    蛇形 segments 为相对区域中心的 NE 模板，周期从默认 (20,40) 向右出发。
    四机分别旋转 0/90/180/270 度，并按 phase_offsets_s 做时间错相。
    长度/覆盖属于给定轨迹模型；圆形分支保留旧版给定轨迹的数值行为。
    """
    kind = config["trajectory"].get("kind", "circle")
    if kind == "circle":
        radii = patrol_radii(config)
        lengths = 2 * np.pi * radii
        return {"kind": kind, "segments": [], "radii_m": radii,
                "length_m": lengths, "period_s": lengths / config["trajectory"]["speed_m_s"]}
    if kind != "serpentine_equal":
        raise ValueError(f"未知轨迹 kind: {kind}")
    result = copy.deepcopy(_patrol_template(config))
    center, phases = _patrol_frame(config)
    result.update(center_xy_m=center.tolist(), rotation_rad=(np.arange(4) * np.pi / 2).tolist(),
                  phase_offsets_s=(phases * result["period_s"]).tolist())
    return result


def _patrol_segment_position(segment: dict, time_s: float) -> np.ndarray:
    """按线段的加减速或圆弧的匀速，解析求出段内位置。"""
    elapsed = min(max(time_s, 0.0), segment["duration_s"])
    if segment["kind"] == "arc":
        angle = segment["start_angle_rad"] + segment["sweep_angle_rad"] * elapsed / segment["duration_s"]
        return np.asarray(segment["center_xy_m"]) + segment["radius_m"] * np.array([np.cos(angle), np.sin(angle)])
    v0, acceleration = segment["start_speed_m_s"], segment["acceleration_m_s2"]
    if elapsed < segment["accel_time_s"]:
        distance = v0 * elapsed + acceleration * elapsed ** 2 / 2
    elif elapsed < segment["accel_time_s"] + segment["cruise_time_s"]:
        distance = segment["accel_distance_m"] + segment["peak_speed_m_s"] * (elapsed - segment["accel_time_s"])
    else:
        remaining = segment["duration_s"] - elapsed
        distance = segment["length_m"] - v0 * remaining - acceleration * remaining ** 2 / 2
    return np.asarray(segment["start_xy_m"]) + distance * np.asarray(segment["direction_xy"])


def patrol_positions(physical_s: float, config: dict) -> np.ndarray:
    """返回标量物理时刻的 XYZ[N,3]；周期首尾及所有线/弧接点位置、速度连续。"""
    physical_s = float(physical_s)
    if not np.isfinite(physical_s):
        raise ValueError("physical_s 必须有限")
    n, t = config["num_uavs"], config["trajectory"]
    kind = t.get("kind", "circle")
    if kind == "circle":
        radius = patrol_radii(config)
        phase = 2 * np.pi * np.arange(n) / n
        angle = phase + t["speed_m_s"] * physical_s / radius
        return np.column_stack((radius * np.cos(angle) + t["center_offset_m"] * np.cos(phase),
                                radius * np.sin(angle) + t["center_offset_m"] * np.sin(phase),
                                np.full(n, t["altitude_m"])))
    if kind != "serpentine_equal":
        raise ValueError(f"未知轨迹 kind: {kind}")
    template = _patrol_template(config)
    center, phases = _patrol_frame(config)
    segments, period = template["segments"], template["period_s"]
    end_times = [segment["end_time_s"] for segment in segments]
    xyz = np.empty((4, 3), dtype=float)
    for uid, fraction in enumerate(phases):
        local_time = (physical_s + fraction * period) % period
        index = min(int(np.searchsorted(end_times, local_time, side="right")), len(segments) - 1)
        xy = _patrol_segment_position(segments[index], local_time - segments[index]["start_time_s"])
        x, y = xy
        rotated = ((x, y), (-y, x), (-x, -y), (y, -x))[uid]
        xyz[uid, :2], xyz[uid, 2] = np.asarray(rotated) + center, t["altitude_m"]
    return xyz


def grouped_profile(profile: dict) -> dict:
    """统一实测/旧版画像到五个组边界，并保留各列对应的残差块数。"""
    metadata = profile.get("metadata", {})
    points = metadata.get("split_points")
    if points is None:
        # 旧版九列画像按每个块后的位置排列，新版五列按组排列。
        points = range(len(BLOCKS) + 1) if np.shape(profile["client_cycles"])[-1:] == (9,) else SPLIT_POINTS
    points = tuple(points)
    if points not in (SPLIT_POINTS, tuple(range(len(BLOCKS) + 1))):
        raise ValueError("画像的切点顺序与 ResNet18 组边界不一致")
    columns = [points.index(cut) for cut in SPLIT_POINTS]
    result = dict(profile)
    fields = ("client_cycles", "server_cycles", "activation_bits", "gradient_bits", "prefix_bits")
    for key in fields + tuple(k for k in ("client_flops", "server_flops") if k in profile):
        values = np.asarray(profile[key], dtype=float)
        valid_shape = values.shape == (len(points),)
        if key == "client_cycles":
            valid_shape = values.ndim in (1, 2) and values.shape[-1] == len(points)
        if not valid_shape or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"画像 {key} 的长度必须与切点数一致且数值为有限正数")
        result[key] = values[..., columns]
    result["metadata"] = {**metadata, "split_points": list(SPLIT_POINTS)}
    return result


def channel_at(physical_s: float, config: dict) -> dict:
    """式(6)—(8)：在给定巡逻轨迹上计算确定性大尺度平均增益，无小尺度衰落。

    轨迹位置连续；实验主循环仍在轮初冻结信道，不在整轮内部积分。
    """
    t, c = config["trajectory"], config["channel"]
    xyz = patrol_positions(physical_s, config)
    delta = xyz - np.asarray(t["server_xyz_m"])
    distance = np.linalg.norm(delta, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(delta[:, 2] / distance, -1, 1)))
    probability = 1 / (1 + c["los_a"] * np.exp(-c["los_b"] * (elevation - c["los_a"])))
    gain = probability * c["beta_l"] * distance ** (-c["alpha_l"]) + (1 - probability) * c["beta_n"] * distance ** (-c["alpha_n"])
    result = {"xyz_m": xyz, "distance_m": distance, "elevation_deg": elevation, "los_probability": probability, "gain": gain}
    if t.get("kind", "circle") == "serpentine_equal":
        template = _patrol_template(config)
        _, phases = _patrol_frame(config)
        result.update(trajectory_kind="serpentine_equal", patrol_period_s=template["period_s"],
                      patrol_length_m=template["length_m"], phase_offsets_s=phases * template["period_s"])
    return result


def client_cycle_table(profile: dict, num_uavs: int) -> np.ndarray:
    """旧画像共享五列，新画像每机一行。"""
    values = np.asarray(profile["client_cycles"], dtype=float)
    if values.shape not in ((len(SPLIT_POINTS),), (num_uavs, len(SPLIT_POINTS))):
        raise ValueError("客户端周期画像必须为 [5] 或 [num_uavs,5]")
    return np.broadcast_to(values, (num_uavs, len(SPLIT_POINTS)))


def maximum_frequency(config: dict) -> np.ndarray:
    """式(9)：由计算功率上限恢复 f_max，返回 Hz。"""
    return (uav_parameter(config, "compute_power_max_w") / uav_parameter(config, "kappa")) ** (1 / 3)


def reference_resources(config: dict) -> dict:
    """返回独立参考配置；其可行性必须由 audit_resources 判定。"""
    n = config["num_uavs"]
    return {"frequency_hz": maximum_frequency(config) * config["reference"]["frequency_fraction"],
            "power_w": uav_parameter(config, "power_max_w") * config["reference"]["power_fraction"],
            "up_hz": np.full(n, config["radio"]["total_up_hz"] / n),
            "down_hz": np.full(n, config["radio"]["total_down_hz"] / n)}


def shannon_rate(bandwidth_hz: Any, power_w: Any, gain: Any, noise_psd: float) -> np.ndarray:
    """式(8)，返回 bit/s；带宽内噪声=N0*B。"""
    b, p, h = np.asarray(bandwidth_hz), np.asarray(power_w), np.asarray(gain)
    if np.any(b <= 0) or np.any(p <= 0) or np.any(h <= 0):
        raise ValueError("速率输入必须严格为正")
    return b * np.log1p(p * h / (noise_psd * b)) / np.log(2)


def make_problem(epoch: int, count: int, gain: Any, split: Any, next_split: Any,
                 remaining_j: Any, config: dict) -> dict:
    """构造当前 epoch 的独立 Problem，供成本计算与资源优化共用。

    count 为每机本轮冻结样本数 D_k；gain、split、remaining_j 均为长度 N。
    split 是本轮执行的切点，next_split 只决定聚合后新前缀的下发负载。
    非最终偶数轮必须传 next_split；奇数轮和最终轮必须传 None。这样可以在
    P2 求解前确定所有通信量，避免求解后才切换模型造成时延/能耗漏算。
    返回时复制输入数组，后续扣电量或切换切点不会修改这份问题快照。
    """
    n, epochs = config["num_uavs"], config["training"]["epochs"]
    aggregate, final = epoch % 2 == 0, epoch == epochs
    if not 1 <= epoch <= epochs or count < 1:
        raise ValueError("epoch/count 越界")
    s = np.asarray(split, dtype=int)
    q = None if next_split is None else np.asarray(next_split, dtype=int)
    if not np.array_equal(np.asarray(split), s) or (q is not None and not np.array_equal(np.asarray(next_split), q)):
        raise ValueError("切分编号必须为整数")
    if s.shape != (n,) or not np.isin(s, SPLIT_POINTS).all():
        raise ValueError("当前切分向量非法")
    if aggregate and not final and (q is None or q.shape != (n,) or not np.isin(q, SPLIT_POINTS).all()):
        raise ValueError("非最终聚合轮需要下一周期切分向量")
    if (not aggregate or final) and q is not None:
        raise ValueError("非聚合轮或最终轮不应传 next_split")
    gain, remaining = np.asarray(gain, dtype=float), np.asarray(remaining_j, dtype=float)
    if gain.shape != (n,) or remaining.shape != (n,) or not np.isfinite(gain).all() or np.any(gain <= 0) or not np.isfinite(remaining).all():
        raise ValueError("信道或剩余电量形状/数值不合法")
    return {"epoch": epoch, "count": count, "gain": gain.copy(), "split": s.copy(),
            "next_split": None if q is None else q.copy(), "remaining_j": remaining.copy(),
            "aggregate": aggregate, "final": final}


def loads(problem: dict, profile: dict) -> dict:
    """返回当前轮各机周期数、训练通信量、同步通信量；最终下发严格为零。"""
    s, d = problem["split"] // BLOCKS_PER_GROUP, problem["count"]
    zero = np.zeros(len(s))
    # 计算/激活/梯度画像按“单样本”给出，乘 D_k 后才是整轮负载。
    # prefix_bits 已是一份完整前缀的大小，只在聚合时传一次，不再乘 D_k。
    # upload 用当前 s；download 用 next_split，二者可能跨越不同残差块边界。
    return {"client_cycles": d * client_cycle_table(profile, len(s))[np.arange(len(s)), s],
            "server_cycles": d * profile["server_cycles"][s],
            "activation_bits": d * profile["activation_bits"][s], "gradient_bits": d * profile["gradient_bits"][s],
            "upload_bits": profile["prefix_bits"][s] if problem["aggregate"] else zero,
            "download_bits": profile["prefix_bits"][problem["next_split"] // BLOCKS_PER_GROUP] if problem["next_split"] is not None else zero}


def evaluate_resources(problem: dict, resources: dict, profile: dict, config: dict) -> dict:
    """用原式(10)—(17)计算真实模型成本，返回可直接记录的 metrics 字典。

    Resources 的 frequency_hz/power_w/up_hz/down_hz 均为长度 N 的物理量数组。
    duration_s、total_energy_j、server_energy_j 为标量；uav_energy_j、各链路
    速率及 components 中各能耗项为长度 N 的数组；stage_times 为阶段标量。
    此处的“真实”指原始数学模型值，用来核验凸上界解，不是硬件功耗实测。
    """
    l = loads(problem, profile)
    f, p = resources["frequency_hz"], resources["power_w"]
    e, u, c = config["server"], config["uav"], config["channel"]
    ru = shannon_rate(resources["up_hz"], p, problem["gain"], c["noise_psd_w_hz"])
    rd = shannon_rate(resources["down_hz"], e["downlink_power_w"], problem["gain"], c["noise_psd_w_hz"])
    tc, te = l["client_cycles"] / f, l["server_cycles"] / e["frequency_hz"]
    ta, tg, tu, td = l["activation_bits"] / ru, l["gradient_bits"] / rd, l["upload_bits"] / ru, l["download_bits"] / rd
    components = {"uav_compute_j": uav_parameter(config, "kappa") * l["client_cycles"] * f ** 2,
                  "activation_upload_j": p * ta, "prefix_upload_j": p * tu,
                  "server_compute_j": e["kappa"] * e["frequency_hz"] ** 2 * l["server_cycles"],
                  "gradient_download_j": e["downlink_power_w"] * tg,
                  "prefix_download_j": e["downlink_power_w"] * td}
    # UAV 电池只扣本地计算和两类上行；下行发射能耗属于服务器。
    # 聚合处理能耗为服务器侧一次性开销，不分摊进各 UAV 的训练电池。
    uav_energy = components["uav_compute_j"] + components["activation_upload_j"] + components["prefix_upload_j"]
    agg_time = e["aggregation_time_s"] if problem["aggregate"] else 0.0
    agg_energy = e["aggregation_energy_j"] if problem["aggregate"] else 0.0
    server_energy = float(sum(np.sum(components[k]) for k in ("server_compute_j", "gradient_download_j", "prefix_download_j")) + agg_energy)
    # 同步屏障使每阶段都要等待该阶段最慢分支，且各阶段最慢的 UAV 可以不同。
    # 因而要 sum(max(stage_n))；不能把四个阶段相加后只取一次 max。
    stage_times = {"split_training_s": float(np.max(tc + te + ta + tg)), "prefix_upload_s": float(np.max(tu)),
                   "aggregation_s": agg_time, "prefix_download_s": float(np.max(td))}
    return {"duration_s": sum(stage_times.values()), "total_energy_j": float(uav_energy.sum() + server_energy),
            "uav_energy_j": uav_energy, "server_energy_j": server_energy, "components": components,
            "aggregation_energy_j": agg_energy, "stage_times": stage_times,
            "branch_training_s": tc + te + ta + tg, "up_rate_bps": ru, "down_rate_bps": rd,
            "compute_power_w": uav_parameter(config, "kappa") * f ** 3, "loads": l}


def audit_resources(problem: dict, resources: dict, profile: dict, config: dict) -> dict:
    """原始约束核验，返回 ok、无量纲最大违约与带单位的物理残差。

    残差按“实际值 - 上界”定义：正数表示违约，负数表示尚有余量。为让 Hz、W、
    J、s 可以共用数值容差，分别除以对应上界，再取正部最大值。无效资源数组
    会直接返回 ok=False/reason，不计算可能含除零的速率；调用方先读 ok。
    求解器报告 optimal/optimal_inaccurate 后，也必须通过这一原式检查。
    """
    n, tol = config["num_uavs"], config["optimization"]["feasibility_tolerance"]
    if any(np.shape(v) != (n,) or not np.isfinite(v).all() or np.any(np.asarray(v) <= 0) for v in resources.values()):
        return {"ok": False, "reason": "nonpositive_or_invalid_resources"}
    m = evaluate_resources(problem, resources, profile, config)
    residuals = {"deadline_s": m["duration_s"] - config["task"]["epoch_deadline_s"],
                 "battery_j": m["uav_energy_j"] - problem["remaining_j"],
                 "frequency_hz": resources["frequency_hz"] - maximum_frequency(config),
                 "power_w": resources["power_w"] - uav_parameter(config, "power_max_w"),
                 "up_hz": float(np.sum(resources["up_hz"])) - config["radio"]["total_up_hz"],
                 "down_hz": float(np.sum(resources["down_hz"])) - config["radio"]["total_down_hz"]}
    normalized = [residuals["deadline_s"] / config["task"]["epoch_deadline_s"],
                  float(np.max(residuals["battery_j"] / np.maximum(1.0, problem["remaining_j"]))),
                  float(np.max(residuals["frequency_hz"] / maximum_frequency(config))),
                  float(np.max(residuals["power_w"] / uav_parameter(config, "power_max_w"))),
                  residuals["up_hz"] / config["radio"]["total_up_hz"], residuals["down_hz"] / config["radio"]["total_down_hz"]]
    violation = max(0.0, *normalized)
    return {"ok": bool(violation <= tol and np.isfinite(m["total_energy_j"])),
            "normalized_violation": violation, "residuals": residuals}


def normalized_resource_variables(problem: dict, num_uavs: int,
                                  tangent: dict | None, fixed_power: float | None) -> dict:
    """创建频率、功率、带宽比例与归一化阶段时长变量。"""
    import cvxpy as cp
    # f=f_actual/f_max，p=p_actual/p_max，bu/bd=B_actual/B_total。
    f = cp.Variable(num_uavs, name="frequency_fraction")
    bu, bd = cp.Variable(num_uavs, name="up_fraction"), cp.Variable(num_uavs, name="down_fraction")
    p = cp.Variable(num_uavs, name="power_fraction") if tangent is not None else np.full(num_uavs, fixed_power)
    # 奇数轮仅训练；中间聚合轮还上传和下发；末轮没有下发。
    tau = cp.Variable(1 + int(problem["aggregate"]) + int(problem["next_split"] is not None),
                      nonneg=True, name="stage_s")
    return {"f": f, "bu": bu, "bd": bd, "p": p, "tau": tau}


def convex_rate_expressions(problem: dict, config: dict, variables: dict) -> dict:
    """用 Shannon perspective 构建 Mbit/s 速率及其倒数表达式。"""
    import cvxpy as cp
    bu, bd, p = variables["bu"], variables["bd"], variables["p"]
    radio, e = config["radio"], config["server"]
    noise, gain = config["channel"]["noise_psd_w_hz"], problem["gain"]
    pmax = uav_parameter(config, "power_max_w")
    # -rel_entr(b,b+a*p)=b*log(1+a*p/b)，对带宽和功率联合凹。
    ru = -(radio["total_up_hz"] / 1e6) * cp.rel_entr(bu, bu + cp.multiply(gain * pmax / (noise * radio["total_up_hz"]), p)) / np.log(2)
    rd = -(radio["total_down_hz"] / 1e6) * cp.rel_entr(bd, bd + gain * e["downlink_power_w"] / (noise * radio["total_down_hz"])) / np.log(2)
    invru, invrd = cp.inv_pos(ru), cp.inv_pos(rd)
    return {"ru": ru, "rd": rd, "invru": invru, "invrd": invrd}


def convex_energy_expressions(problem: dict, profile: dict, config: dict, load_data: dict,
                              variables: dict, rates: dict, tangent: dict | None,
                              fixed_power: float | None) -> dict:
    """构建计算/通信能量、变量边界，以及 Phase-I 或相切上界的尺度。"""
    import cvxpy as cp
    n, floor = config["num_uavs"], config["optimization"]["positive_floor"]
    f, bu, bd, p = (variables[key] for key in ("f", "bu", "bd", "p"))
    ru, invru, invrd = rates["ru"], rates["invru"], rates["invrd"]
    fmax = maximum_frequency(config)
    pmax, kappa = uav_parameter(config, "power_max_w"), uav_parameter(config, "kappa")
    ec = cp.multiply(kappa * load_data["client_cycles"] * fmax ** 2, cp.square(f))
    # 特征通信与模型同步共享资源，能量先合并载荷，时延仍分阶段约束。
    up_mbit = (load_data["activation_bits"] + load_data["upload_bits"]) / 1e6
    down_mbit = (load_data["gradient_bits"] + load_data["download_bits"]) / 1e6
    constraints = [f >= floor, f <= 1, bu >= floor, bd >= floor, cp.sum(bu) <= 1, cp.sum(bd) <= 1]
    if tangent is not None:
        m0 = evaluate_resources(problem, tangent, profile, config)
        ru0, p0 = m0["up_rate_bps"] / 1e6, tangent["power_w"]
        # 当前点 psi=1，E_up0/psi 与原上行能量相切；正凹函数的倒数为凸。
        psi = 2 * cp.sqrt(cp.multiply(1 / ru0, ru)) - cp.multiply(pmax / p0, p)
        eup0 = up_mbit * p0 / ru0
        eup = cp.multiply(eup0, cp.inv_pos(psi))
        constraints += [p >= floor, p <= 1, psi >= floor]
        scale, slack = max(1e-6, m0["total_energy_j"] / n), 0.0
    else:
        eup = cp.multiply(up_mbit * pmax * fixed_power, invru)
        scale = max(1e-6, float(np.sum(kappa * load_data["client_cycles"] * fmax ** 2)) / n)
        slack = cp.Variable(nonneg=True, name="phase_one_slack")
    edown = cp.multiply(config["server"]["downlink_power_w"] * down_mbit, invrd)
    return {"compute": ec, "uplink": eup, "downlink": edown, "constraints": constraints,
            "scale": scale, "slack": slack}


def build_convex_program(problem: dict, load_data: dict, config: dict, variables: dict,
                         rates: dict, energy: dict, tangent: dict | None) -> Any:
    """按原顺序加入各阶段时延、电量约束，返回能量最小化凸程序。"""
    import cvxpy as cp
    f, tau = variables["f"], variables["tau"]
    invru, invrd = rates["invru"], rates["invrd"]
    e, deadline = config["server"], config["task"]["epoch_deadline_s"]
    ec, eup, edown = energy["compute"], energy["uplink"], energy["downlink"]
    scale, slack = energy["scale"], energy["slack"]
    train_time = cp.multiply(load_data["client_cycles"] / maximum_frequency(config), cp.inv_pos(f)) + load_data["server_cycles"] / e["frequency_hz"]
    train_time += cp.multiply(load_data["activation_bits"] / 1e6, invru) + cp.multiply(load_data["gradient_bits"] / 1e6, invrd)
    constraints = list(energy["constraints"])
    constraints += [train_time / deadline <= tau[0]]
    if problem["aggregate"]:
        constraints += [cp.multiply(load_data["upload_bits"] / 1e6 / deadline, invru) <= tau[1]]
    if problem["next_split"] is not None:
        constraints += [cp.multiply(load_data["download_bits"] / 1e6 / deadline, invrd) <= tau[2]]
    agg_time = e["aggregation_time_s"] if problem["aggregate"] else 0.0
    battery_scale = np.maximum(scale, np.abs(problem["remaining_j"]))
    # Phase-I 放宽归一化时限和电量；后续仍以 audit_resources 判定是否可行。
    constraints += [cp.sum(tau) + agg_time / deadline <= 1 + slack,
                    cp.multiply(1 / battery_scale, ec + eup) <= problem["remaining_j"] / battery_scale + slack]
    # 后缀计算和聚合能量为常数，此处省略，原式成本统计时完整加回。
    objective = cp.sum(ec + eup + edown) / scale
    objective = objective if tangent is not None else slack + 1e-8 * objective
    return cp.Problem(cp.Minimize(objective), constraints)


def solve_convex_program(program: Any, solver: str) -> dict:
    """检查 DCP/求解器并执行求解，记录求解警告和原有计时范围。"""
    import cvxpy as cp
    if not program.is_dcp():
        raise RuntimeError("凸子问题没有通过 CVXPY DCP 校验")
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"求解器 {solver} 不可用；当前 {cp.installed_solvers()}")
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            options = {"max_iter": 300, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9} if solver == "CLARABEL" else {}
            program.solve(solver=solver, **options)
        notes = [str(w.message) for w in caught]
    except cp.error.SolverError as exc:
        return {"ok": False, "status": "solver_error", "message": str(exc), "wall_s": time.perf_counter() - start}
    return {"ok": False, "status": program.status, "warnings": notes, "dcp": True,
            "wall_s": time.perf_counter() - start}


def recover_convex_resources(variables: dict, config: dict, tangent: dict | None) -> dict:
    """将求解后的比例变量恢复为实际频率、功率和上下行带宽。"""
    p, radio = variables["p"], config["radio"]
    power_fraction = np.asarray(p.value if tangent is not None else p)
    return {"frequency_hz": np.asarray(variables["f"].value) * maximum_frequency(config),
            "power_w": power_fraction * uav_parameter(config, "power_max_w"),
            "up_hz": np.asarray(variables["bu"].value) * radio["total_up_hz"],
            "down_hz": np.asarray(variables["bd"].value) * radio["total_down_hz"]}


def convex_subproblem(problem: dict, profile: dict, config: dict,
                      tangent: dict | None = None, fixed_power: float | None = None) -> dict:
    """求解一次凸问题并返回资源/求解状态。

    tangent != None：式(31)，用相切点归一化后的 psi=2*sqrt(R/R0)-p/p0，
    上行上界=E_up0/psi，与式(29)—(31)完全等价，避免 y 的数量级病态。
    fixed_power != None：固定发射功率的凸 Phase-I，最小化时延/电量松弛；
    只用于寻找原问题可行点，找不到不构成全局不可行证明。

    tangent 为已经验证可行的 Resources；fixed_power 是 p_max 的比例，两者
    必须且只能提供一个。输出 ok 指原约束也已通过，而不只是求解器有数值解。
    成功时 resources 恢复为 Hz/W，epigraph_s 恢复为秒；status/warnings 保存
    求解器状态。失败结果可能没有 resources/audit，调用方需先检查 ok。
    """
    import cvxpy as cp
    if (tangent is None) == (fixed_power is None):
        raise ValueError("必须且只能指定 tangent 或 fixed_power")
    load_data = loads(problem, profile)
    variables = normalized_resource_variables(problem, config["num_uavs"], tangent, fixed_power)
    rates = convex_rate_expressions(problem, config, variables)
    energy = convex_energy_expressions(problem, profile, config, load_data, variables,
                                       rates, tangent, fixed_power)
    program = build_convex_program(problem, load_data, config, variables, rates, energy, tangent)
    result = solve_convex_program(program, config["optimization"]["solver"])
    if result["status"] not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or any(
            variables[key].value is None for key in ("f", "bu", "bd")):
        return result
    resources = recover_convex_resources(variables, config, tangent)
    audit = audit_resources(problem, resources, profile, config)
    result.update({"ok": audit["ok"], "resources": resources, "audit": audit,
                   "epigraph_s": np.asarray(variables["tau"].value) * config["task"]["epoch_deadline_s"],
                   "phase_slack": float(energy["slack"].value) if tangent is None else 0.0})
    return result


def feasible_initialization(problem: dict, profile: dict, config: dict, previous: dict | None = None) -> dict:
    """返回 {ok, resources, attempts}；失败还包含明确的 status。

    尝试顺序为上一可行配置、初始参考配置、最大频率/功率等分带宽，最后在若干
    固定功率比例下解凸 Phase-I。每个候选均使用当前 Problem 重新审计，不能
    因它上一轮可行就直接复用。attempts 记录全部尝试，便于区分初始化失败与
    后续凸迭代失败。此有限初始化搜索不能提供原非凸问题的不可行证明。
    """
    attempts = []
    for name, candidate in (("previous", previous), ("reference", reference_resources(config))):
        if candidate is None:
            continue
        audit = audit_resources(problem, candidate, profile, config)
        attempts.append({"method": name, "audit": audit})
        if audit["ok"]:
            return {"ok": True, "resources": copy.deepcopy(candidate), "attempts": attempts}
    maximum = reference_resources(config)
    maximum["frequency_hz"][:] = maximum_frequency(config)
    maximum["power_w"][:] = uav_parameter(config, "power_max_w")
    audit = audit_resources(problem, maximum, profile, config)
    attempts.append({"method": "maximum_equal_bandwidth", "audit": audit})
    if audit["ok"]:
        return {"ok": True, "resources": maximum, "attempts": attempts}
    for fraction in (1.0, 0.3, 0.1, 0.03, 0.01, 0.001):
        result = convex_subproblem(problem, profile, config, fixed_power=max(fraction, config["optimization"]["positive_floor"]))
        attempts.append({"method": "fixed_power_phase_one", "power_fraction": fraction,
                         **{k: v for k, v in result.items() if k != "resources"}})
        if result["ok"]:
            return {"ok": True, "resources": result["resources"], "attempts": attempts}
    return {"ok": False, "resources": None, "attempts": attempts, "status": "feasible_initialization_not_found"}


def solve_resources(problem: dict, profile: dict, config: dict, previous: dict | None = None,
                    *, retain_feasible: bool = False) -> dict:
    """式(30)/(31)交替求解，返回 SolveResult，接受步真实可行且真实目标不增。

    trace[0] 是已验证初值，此后每项记录真实能耗、相对下降、原约束审计及
    是否接受；并非所有日志项都会被执行。ok=True 时 resources/metrics 可用于
    本轮执行，但 status='iteration_limit_feasible' 只表示达到上限仍然可行。
    ok=False 时返回 resources=None，由上层尝试下一候选或报告任务未完成。
    retain_feasible 用于窗口能耗比较。迭代失败时保留最近通过原约束审计的点并记录原因。
    """
    start = time.perf_counter()
    initialization = feasible_initialization(problem, profile, config, previous)
    trace = []
    def result(ok, status, resources=None):
        return {"ok": ok, "status": status, "resources": resources,
                "metrics": evaluate_resources(problem, resources, profile, config) if resources is not None else None,
                "trace": trace, "initialization": initialization["attempts"], "wall_s": time.perf_counter() - start}
    if not initialization["ok"]:
        return result(False, "feasible_initialization_not_found")
    current = initialization["resources"]
    energy = evaluate_resources(problem, current, profile, config)["total_energy_j"]
    trace.append({"iteration": 0, "energy_j": energy, "audit": audit_resources(problem, current, profile, config)})
    for iteration in range(1, config["optimization"]["max_iterations"] + 1):
        proposed = convex_subproblem(problem, profile, config, tangent=current)
        if not proposed["ok"]:
            trace.append({"iteration": iteration, "accepted": False, "subproblem": proposed})
            if retain_feasible:
                return result(True, "subproblem_failed_previous_feasible", current)
            return result(False, "convex_subproblem_failed")
        candidate = proposed["resources"]
        # 停止条件和接受条件均依据原始总能量，而不是随切线更新的凸上界目标。
        # 分母防止能量接近零时出现除零；每步还独立通过了原始约束核验。
        metrics = evaluate_resources(problem, candidate, profile, config)
        new_energy = metrics["total_energy_j"]
        relative = (energy - new_energy) / max(abs(energy), 1e-12)
        entry = {"iteration": iteration, "energy_j": new_energy, "relative_decrease": relative,
                 "solver_status": proposed["status"], "audit": proposed["audit"],
                 "solver_wall_s": proposed["wall_s"], "warnings": proposed["warnings"]}
        if new_energy > energy:
            entry["accepted"] = False
            trace.append(entry)
            # 数值误差范围内保留前一个已验证的可行点；不能把升高的点写入执行状态。
            if -relative <= config["optimization"]["relative_tolerance"]:
                return result(True, "numerical_stationary_previous_feasible", current)
            if retain_feasible:
                return result(True, "objective_increased_previous_feasible", current)
            return result(False, "true_objective_increased")
        entry["accepted"] = True
        trace.append(entry)
        current, energy = candidate, new_energy
        if relative <= config["optimization"]["relative_tolerance"]:
            return result(True, "converged", current)
    return result(True, "iteration_limit_feasible", current)


def reference_scores(count: int, gain: np.ndarray, resources: dict, profile: dict,
                     first_cycle: bool, config: dict) -> np.ndarray:
    """式(26)：返回 [N,5] 参考能耗评分矩阵，元素是某机某候选的周期评分 J。

    count 是决策时刻每机样本数，gain 是已完成 epoch 窗口的平均信道。
    预测周期固定使用 m=2*count。首周期模型已预部署，chi=0；后续周期
    chi=1，计入新前缀下发。与新切点无关的旧前缀上传和聚合常数在评分中
    省略，但 screen_split 的可行性检查仍包括这些成本。
    """
    u, e, noise = config["uav"], config["server"], config["channel"]["noise_psd_w_hz"]
    ru = shannon_rate(resources["up_hz"], resources["power_w"], gain, noise)[:, None]
    rd = shannon_rate(resources["down_hz"], e["downlink_power_w"], gain, noise)[:, None]
    m, chi = 2 * count, int(not first_cycle)
    compute = m * (uav_parameter(config, "kappa")[:, None] * resources["frequency_hz"][:, None] ** 2
                   * client_cycle_table(profile, config["num_uavs"])
                   + e["kappa"] * e["frequency_hz"] ** 2 * profile["server_cycles"])
    upload = resources["power_w"][:, None] * (m * profile["activation_bits"] + profile["prefix_bits"]) / ru
    download = e["downlink_power_w"] * (m * profile["gradient_bits"] + chi * profile["prefix_bits"]) / rd
    return compute + upload + download


def screen_split(candidate: np.ndarray, current: np.ndarray, epoch: int, count: int,
                 gain: np.ndarray, remaining_j: np.ndarray, reference: dict,
                 profile: dict, config: dict) -> dict:
    """式(25)：返回参考预测范围的可行性、各轮时长、累计每机能耗及拒绝原因。

    首周期检查 epoch=1,2；在 epoch=2 预选下一周期时，检查当前未执行的
    聚合轮 2 和未来训练轮 3,4。整个范围固定当前 count 和窗口信道，属于
    参考预测，不读取未来真实 D(t)/h(t)。电池约束按范围内累计能耗检查。
    """
    first = epoch == 1
    start = 1 if first else epoch + 1
    horizon = []
    if not first:
        horizon.append(make_problem(epoch, count, gain, current, candidate, remaining_j, config))
    for k in range(start, start + 2):
        q = candidate if k % 2 == 0 and k < config["training"]["epochs"] else None
        horizon.append(make_problem(k, count, gain, candidate, q, remaining_j, config))
    metrics = [evaluate_resources(p, reference, profile, config) for p in horizon]
    total_uav = sum(m["uav_energy_j"] for m in metrics)
    times = [m["duration_s"] for m in metrics]
    tol = config["optimization"]["feasibility_tolerance"]
    time_ok = max(times) <= config["task"]["epoch_deadline_s"] * (1 + tol)
    energy_ok = bool(np.all(total_uav <= remaining_j + tol * np.maximum(1, remaining_j)))
    return {"ok": bool(time_ok and energy_ok), "duration_s": times, "uav_energy_j": total_uav,
            "reason": "feasible" if time_ok and energy_ok else ("reference_deadline" if not time_ok else "reference_battery")}


def forecast_split_energy(cuts, *, config, profile, epoch, count, current, gain, remaining, reference):
    """在相同预测信道和样本数下逐轮优化并扣电，累计原公式的系统能耗。"""
    # 即使忽略全部通信，最快 CPU 也无法按时完成的切点可直接排除。
    tol = config["optimization"]["feasibility_tolerance"]
    max_clock = maximum_frequency(config) * (1 + tol)
    compute_seconds = count * (client_cycle_table(profile, config["num_uavs"]) / max_clock[:, None]
                               + profile["server_cycles"] / config["server"]["frequency_hz"])
    used_cuts = [cuts] if epoch == 1 else [current, cuts]
    if any(np.any(compute_seconds[np.arange(config["num_uavs"]), np.asarray(vector) // BLOCKS_PER_GROUP]
                  > config["task"]["epoch_deadline_s"] * (1 + tol)) for vector in used_cuts):
        return cuts, None, "compute_time_exceeds_deadline"
    battery = np.array(remaining, dtype=float)
    total = 0.0
    statuses = []
    last = 2 if epoch == 1 else epoch + 2
    for k in range(epoch, last + 1):
        active = current if k == epoch and epoch != 1 else cuts
        next_cuts = cuts if k % 2 == 0 and k < config["training"]["epochs"] else None
        problem = make_problem(k, count, gain, active, next_cuts, battery, config)
        result = solve_resources(problem, profile, config, reference, retain_feasible=True)
        if not result["ok"]:
            return cuts, None, f"epoch_{k}: {result['status']}"
        total += result["metrics"]["total_energy_j"]
        battery -= result["metrics"]["uav_energy_j"]
        statuses.append(result["status"])
    return cuts, total, ", ".join(statuses)


def select_splits(epoch: int, count: int, current: np.ndarray, gain_history: list[np.ndarray],
                  initial_gain: np.ndarray, remaining_j: np.ndarray, reference: dict,
                  profile: dict, config: dict) -> dict:
    """穷举组边界组合，以式(25)筛选并按式(26)的参考总能耗排列 P2 候选。

    scores_j 的列按 SPLIT_POINTS 排列。总分相同时优先沿用当前组合，
    其余同分组合按切点向量的字典序排列。
    """
    o, n = config["optimization"], config["num_uavs"]
    window = 1 if o["strategy"] == "no_window" else o["window_epochs"]
    gain = np.mean(gain_history[-window:], axis=0) if gain_history else initial_gain.copy()
    scores = reference_scores(count, gain, reference, profile, epoch == 1, config)
    current_key = tuple(current)
    if o["split_objective"] == "optimized_window":
        evaluate = partial(forecast_split_energy, config=config, profile=profile, epoch=epoch, count=count,
                           current=current, gain=gain, remaining=remaining_j, reference=reference)
        combinations = product(SPLIT_POINTS, repeat=n)
        if o["split_workers"] == 1:
            rows = list(map(evaluate, combinations))
        else:
            with ProcessPoolExecutor(max_workers=o["split_workers"]) as pool:
                rows = list(pool.map(evaluate, combinations, chunksize=4))
        ranked = sorted((energy, cuts != current_key, cuts) for cuts, energy, _ in rows if energy is not None)
        return {"ok": bool(ranked), "status": "optimized_window_complete" if ranked else "no_window_solution_found",
                "candidates": [np.array(cuts) for _, _, cuts in ranked], "scores_j": scores, "window_gain": gain,
                "trace": [{"vector": cuts, "forecast_energy_j": energy, "feasible": energy is not None,
                           "reason": status} for cuts, energy, status in rows],
                "combinations_evaluated": len(rows), "feasible_combinations": len(ranked)}
    uavs = np.arange(n)
    trace, feasible = [], []
    for cuts in product(SPLIT_POINTS, repeat=n):
        candidate = np.array(cuts, dtype=int)
        score = float(scores[uavs, candidate // BLOCKS_PER_GROUP].sum())
        audit = screen_split(candidate, current, epoch, count, gain, remaining_j, reference, profile, config)
        trace.append({"vector": cuts, "score_j": score, "feasible": audit["ok"], "reason": audit["reason"]})
        if audit["ok"]:
            feasible.append((score, cuts != current_key, cuts))
    feasible.sort()
    return {"ok": bool(feasible), "status": "exhaustive_complete" if feasible else "no_reference_feasible_split",
            "candidates": [np.array(cuts, dtype=int) for _, _, cuts in feasible], "scores_j": scores,
            "window_gain": gain, "trace": trace, "combinations_evaluated": len(trace),
            "feasible_combinations": len(feasible)}


def rebuild_reference(epoch: int, count: int, current: np.ndarray, window_gain: np.ndarray,
                      remaining_j: np.ndarray, profile: dict, config: dict) -> dict:
    """可选工程扩展（文稿默认关闭）：重建固定参考资源后重做 P1。

    用保留当前切分的聚合轮找可行配置，并将电量按参考 horizon 均分作为保守
    初始化预算。仍须通过完整式(25)筛选及当前真实状态 P2，不保证任务能完成。
    """
    first = epoch == 1
    aggregation_epoch = 2 if first else epoch
    q = current if aggregation_epoch < config["training"]["epochs"] else None
    horizon_size = 2 if first else 3
    problem = make_problem(aggregation_epoch, count, window_gain, current, q, remaining_j / horizon_size, config)
    initialized = feasible_initialization(problem, profile, config)
    if not initialized["ok"]:
        return initialized
    screen = screen_split(current, current, epoch, count, window_gain, remaining_j,
                          initialized["resources"], profile, config)
    return {**initialized, "ok": screen["ok"], "reference_screen": screen}


def select_epoch_splits(epoch, count, current, gain_history, gain, remaining, reference, profile, config):
    """选择本轮候选；必要时重建参考资源并重新筛选，返回候选和参考资源。"""
    selection = select_splits(epoch, count, current, gain_history, gain, remaining,
                              reference, profile, config)
    if (not selection["ok"] and config["optimization"]["split_objective"] == "reference_score"
            and config["optimization"]["reference_recovery"] == "rebuild"):
        original_failure = selection
        recovery = rebuild_reference(epoch, count, current, selection["window_gain"], remaining, profile, config)
        if recovery["ok"]:
            reference = recovery["resources"]
            selection = select_splits(epoch, count, current, gain_history, gain, remaining,
                                      reference, profile, config)
        selection = {**selection, "reference_recovery": recovery, "original_reference_status": original_failure["status"]}

    return selection, reference


def solve_epoch_candidates(epoch, count, current, gain, remaining, reference, profile, config, selection):
    """依次尝试当前信道下的候选；成功后返回切点、问题和全部求解记录。"""
    strategy = config["optimization"]["strategy"]
    dynamic = strategy in ("dynamic_opt", "no_window")
    epochs = config["training"]["epochs"]
    choosing_current = dynamic and epoch == 1
    choosing_next = epoch % 2 == 0 and epoch < epochs
    queue = selection["candidates"] if selection else [current]
    attempts, solved, problem = [], None, None
    # 第三步：在真实当前信道下尝试候选并求 P2。参考筛选负责产生队列，
    # 不能代替实际时限/电池审计；任何候选都不可行时，在更新模型前停止。
    for cuts in queue:
        trial_current = cuts if choosing_current else current
        q = cuts if choosing_next else None
        problem = make_problem(epoch, count, gain, trial_current, q, remaining, config)
        if strategy == "fixed_fixed":
            resource = reference_resources(config)
            audit = audit_resources(problem, resource, profile, config)
            candidate = {"ok": audit["ok"], "status": "fixed_resources" if audit["ok"] else "fixed_resources_infeasible",
                         "resources": resource, "metrics": evaluate_resources(problem, resource, profile, config),
                         "trace": [], "audit": audit, "wall_s": 0.0}
        else:
            candidate = solve_resources(problem, profile, config, reference,
                                        retain_feasible=dynamic and config["optimization"]["split_objective"] == "optimized_window")
        attempts.append({"split": trial_current, "next_split": q, "solve": candidate})
        if candidate["ok"]:
            current = trial_current
            solved = candidate
            break

    return current, problem, solved, attempts
