"""作用：动态 UAV-SFL 的块级资源优化、原约束核验与物理成本计算。

使用方式（在 code_v1 目录）：
    python -X utf8 -B train_sfl.py --help
    from resource_solver import BlockProblem, solve_block, evaluate_block

本模块由动态训练入口调用，不单独读取数据或启动神经网络。输入载荷用 bit，
计算量用周期，预算用 s，CSI 为线性功率增益。输出资源为 Hz、W、Hz、Hz。
每块的输入 CSI 冻结；计算和通信成本始终用原 Shannon 公式重算。

求解实现正文式（33）至（35）：更新 omega/beta，求近端凸子问题，然后
在原能耗目标上做 Armijo 回溯。采用 CVXPY 的相对熵表示 Shannon 速率，
由 Clarabel 求指数锥问题。没有活动负载的专用变量会被删除。电量目前仅
统计，因而不加入尚未标定的电池上限，也无须启用式（33）的 Q 能耗上界。
失败返回明确的 failed，调用方不能把它替换为参考资源继续提交训练。
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockProblem:
    """一个训练块或聚合块的已知输入；各数组按 uav_ids 的顺序排列。"""

    kind: str
    uav_ids: list[int]
    channel_gains: list[float]
    client_cycles: list[float]
    server_cycles: list[float]
    up_bits: list[float]
    down_bits: list[float]
    budget_s: float
    aggregation_time_s: float = 0.0

    def __post_init__(self):
        """校验块类型、数组维度、非负负载和正预算；聚合块禁止夹带训练周期。"""
        if self.kind not in ("training", "aggregation"):
            raise ValueError("块类型必须为 training 或 aggregation")
        n = len(self.uav_ids)
        if not n or len(set(self.uav_ids)) != n:
            raise ValueError("uav_ids 必须非空且互不重复")
        for name in ("channel_gains", "client_cycles", "server_cycles", "up_bits", "down_bits"):
            values = getattr(self, name)
            if len(values) != n or any(not math.isfinite(float(v)) or v < 0 for v in values):
                raise ValueError(f"{name} 必须为长度 N 的有限非负数组")
        if not math.isfinite(self.budget_s) or self.budget_s <= 0:
            raise ValueError("budget_s 必须为有限正数")
        if not math.isfinite(self.aggregation_time_s) or self.aggregation_time_s < 0:
            raise ValueError("aggregation_time_s 必须非负")
        if self.kind == "aggregation" and any(self.client_cycles + self.server_cycles):
            raise ValueError("聚合块不能含有训练计算负载")


def reference_resources(config, count):
    """返回参考预演资源；该函数不是优化失败时的执行回退。"""
    return dict(frequency_hz=[float(config["uav"]["f_ref_hz"])] * count,
                upload_power_w=[float(config["uav"]["p_up_ref_w"])] * count,
                uplink_bandwidth_hz=[float(config["radio"]["total_up_hz"]) / count] * count,
                downlink_bandwidth_hz=[float(config["radio"]["total_down_hz"]) / count] * count)


def _rate(bandwidth, power, gain, noise):
    """用稳定的log1p计算Shannon速率，输入Hz、W和线性增益，输出bit/s。"""
    if bandwidth <= 0 or power <= 0 or gain <= 0:
        return 0.0
    return bandwidth * math.log1p(power * gain / (noise * bandwidth)) / math.log(2.0)


def _capacity_expression(cp, bandwidth_mhz, power_w, gain_to_noise):
    """Shannon 速率的精确相对熵表达，单位 Mbit/s，不使用速率线性近似。"""
    # 将指数锥参数缩到单位附近；相对熵的齐次变换不改变Shannon公式，
    # 输出速率始终恢复为原始 Mbit/s。
    scale = max(1.0, gain_to_noise)
    return -scale * cp.rel_entr(bandwidth_mhz / scale,
                                (bandwidth_mhz + gain_to_noise * power_w) / scale) / math.log(2.0)


def evaluate_block(problem, config, resources):
    """按原物理公式核算资源，不读取或重新采样信道，也不推进模拟时钟。

    per_uav 给出四阶段总量；训练画像的前向/反向细分由执行层再按画像拆分。
    energy_j 包含服务器训练计算及固定聚合能耗，objective_j 排除资源无关项。
    """
    n = len(problem.uav_ids)
    keys = ("frequency_hz", "upload_power_w", "uplink_bandwidth_hz", "downlink_bandwidth_hz")
    if any(key not in resources or len(resources[key]) != n for key in keys):
        raise ValueError("资源数组长度必须与无人机数量对应")
    noise = config["channel"]["noise_psd_w_hz"]
    server = config["server"]
    records = []
    for i, uid in enumerate(problem.uav_ids):
        f, p, bu, bd = (float(resources[key][i]) for key in keys)
        if any(not math.isfinite(v) or v < 0 for v in (f, p, bu, bd)):
            raise ValueError("资源必须为有限非负数")
        c, d, up, down = (getattr(problem, key)[i] for key in
                          ("client_cycles", "server_cycles", "up_bits", "down_bits"))
        ru = _rate(bu, p, problem.channel_gains[i], noise) if up else 0.0
        rd = _rate(bd, server["downlink_power_w"], problem.channel_gains[i], noise) if down else 0.0
        tc = c / f if c and f > 0 else (math.inf if c else 0.0)
        tu = up / ru if up and ru > 0 else (math.inf if up else 0.0)
        td = down / rd if down and rd > 0 else (math.inf if down else 0.0)
        ts = d / server["frequency_hz"]
        ec = config["uav"]["kappa"] * c * f * f
        es = server["kappa"] * d * server["frequency_hz"] ** 2
        eu = p * tu if up else 0.0
        ed = server["downlink_power_w"] * td if down else 0.0
        records.append(dict(uav_id=uid, client_compute_s=tc, upload_s=tu,
                            server_compute_s=ts, download_s=td,
                            duration_s=tc + tu + ts + td,
                            client_compute_j=ec, upload_j=eu,
                            server_compute_j=es, download_j=ed,
                            uplink_rate_bps=ru, downlink_rate_bps=rd,
                            up_bits=up, down_bits=down))
    if problem.kind == "training":
        duration = max(row["duration_s"] for row in records)
    else:
        duration = (max(row["upload_s"] for row in records) + problem.aggregation_time_s
                    + max(row["download_s"] for row in records))
    totals = {key: sum(row[key] for row in records) for key in
              ("client_compute_j", "upload_j", "server_compute_j", "download_j")}
    totals["aggregation_j"] = (server["aggregation_power_w"] * problem.aggregation_time_s
                               if problem.kind == "aggregation" else 0.0)
    objective = totals["client_compute_j"] + totals["upload_j"] + totals["download_j"]
    return dict(duration_s=duration, energy_j=sum(totals.values()), objective_j=objective,
                per_uav=records, energy_components=totals,
                phase_max_s=dict(upload=max(r["upload_s"] for r in records),
                                 download=max(r["download_s"] for r in records)))


def compact_decision(result):
    """为长任务日志保留必要诊断，避免每块重复输出整套乘子与迭代向量。

    实际资源、原目标历史、四类KKT残差、方向残差、尺度和重试状态全部保留。
    失败块保留完整诊断；成功块省略可由API单独查看的逐次方向向量及乘子表。
    此函数只改变日志体积，不更改用于核验或执行的求解结果。
    """
    if result.get("status") != "success":
        return result
    output = dict(result)
    output["diagnostics"] = {key: value for key, value in result["diagnostics"].items()
                             if key not in ("iteration_records", "multiplier_certificate", "direction_components")}
    return output


class _ReducedProblem:
    """约简变量、固定尺度及原问题的解析函数；不依赖凸子问题的对偶结果。"""

    def __init__(self, problem, config):
        """删除零载荷对应变量，固定GHz、带宽份额、W以及块时间/能量归一化尺度。"""
        self.p, self.cfg = problem, config
        self.n = len(problem.uav_ids)
        self.names, self.indices = [], {}
        self.bounds = []
        u, radio = config["uav"], config["radio"]
        # 频率用 GHz，功率用 W，带宽用本方向总带宽的份额。后者避免
        # 20 MHz 数值直接进入近端范数而压制带宽方向的更新；速率仍用 Mbit/s。
        for group, loads, lower, upper in (
            ("f", problem.client_cycles, u["f_min_hz"] / 1e9, u["f_max_hz"] / 1e9),
            ("p", problem.up_bits, max(u["p_up_min_w"], 1e-10), u["p_up_max_w"]),
            ("bu", problem.up_bits, radio["min_up_hz"] / radio["total_up_hz"], 1.0),
            ("bd", problem.down_bits, radio["min_down_hz"] / radio["total_down_hz"], 1.0)):
            for i, load in enumerate(loads):
                if load > 0:
                    self.indices[group, i] = len(self.names)
                    self.names.append((group, i))
                    self.bounds.append((float(lower), float(upper)))
        self.resource_size = len(self.names)
        if problem.kind == "training":
            phases = ["train"] if any(problem.client_cycles + problem.server_cycles
                                       + problem.up_bits + problem.down_bits) else []
        else:
            phases = (["up"] if any(problem.up_bits) else []) + (["down"] if any(problem.down_bits) else [])
        self.phases = phases
        for phase in phases:
            self.indices[phase, -1] = len(self.names)
            self.names.append((phase, -1))
            self.bounds.append((0.0, None))
        self.size = len(self.names)
        self.tscale = max(float(problem.budget_s), 1e-6)
        reference = evaluate_block(problem, config, reference_resources(config, self.n))
        self.escale = max(reference["objective_j"], 1e-8)

    def pack(self, resources):
        """将SI资源转为归一化向量，辅助时间取原阶段最大值；无效热启动被拒绝。"""
        x = np.zeros(self.size)
        source = {"f": ("frequency_hz", 1e9), "p": ("upload_power_w", 1.0),
                  "bu": ("uplink_bandwidth_hz", self.cfg["radio"]["total_up_hz"]),
                  "bd": ("downlink_bandwidth_hz", self.cfg["radio"]["total_down_hz"])}
        for (group, i), j in self.indices.items():
            if group in source:
                key, scale = source[group]
                x[j] = float(resources[key][i]) / scale
                if not math.isfinite(x[j]) or x[j] <= 0:
                    raise ValueError("热启动的活动资源必须为有限正数")
        # 辅助时间取原阶段最大值，避免热启动中携带无意义的松弛。
        rows = evaluate_block(self.p, self.cfg, self.unpack(x))["per_uav"]
        for phase in self.phases:
            key = {"train": "duration_s", "up": "upload_s", "down": "download_s"}[phase]
            x[self.indices[phase, -1]] = max(row[key] for row in rows) / self.tscale
        return x

    def unpack(self, x):
        """将归一化向量恢复为每机Hz/W资源；已消除的非活动变量显式返回0。"""
        result = dict(frequency_hz=[0.0] * self.n, upload_power_w=[0.0] * self.n,
                      uplink_bandwidth_hz=[0.0] * self.n, downlink_bandwidth_hz=[0.0] * self.n)
        for group, key, scale in (("f", "frequency_hz", 1e9), ("p", "upload_power_w", 1),
                                  ("bu", "uplink_bandwidth_hz", self.cfg["radio"]["total_up_hz"]),
                                  ("bd", "downlink_bandwidth_hz", self.cfg["radio"]["total_down_hz"])):
            for i in range(self.n):
                if (group, i) in self.indices:
                    result[key][i] = float(x[self.indices[group, i]] * scale)
        return result

    def physical_terms(self, x):
        """返回原目标及梯度、每机阶段时间及梯度；全部对归一化 x 求导。"""
        grad = np.zeros(self.size)
        objective = 0.0
        times, time_grads = [], []
        noise = self.cfg["channel"]["noise_psd_w_hz"]
        pd = self.cfg["server"]["downlink_power_w"]
        for i in range(self.n):
            phase = dict(c=0.0, up=0.0, server=self.p.server_cycles[i] / self.cfg["server"]["frequency_hz"], down=0.0)
            jac = {key: np.zeros(self.size) for key in ("c", "up", "server", "down")}
            if ("f", i) in self.indices:
                j = self.indices["f", i]
                f, c = x[j], self.p.client_cycles[i]
                phase["c"] = c / (1e9 * f)
                jac["c"][j] = -c / (1e9 * f * f)
                coefficient = self.cfg["uav"]["kappa"] * c * 1e18
                objective += coefficient * f * f
                grad[j] += 2 * coefficient * f
            for direction, load in (("up", self.p.up_bits[i]), ("down", self.p.down_bits[i])):
                if not load:
                    continue
                jb = self.indices["bu" if direction == "up" else "bd", i]
                bandwidth_scale = self.cfg["radio"]["total_up_hz" if direction == "up" else "total_down_hz"] / 1e6
                b = x[jb] * bandwidth_scale
                jp = self.indices.get(("p", i)) if direction == "up" else None
                power = x[jp] if jp is not None else pd
                a = self.p.channel_gains[i] / (noise * 1e6)
                snr = a * power / b
                rate = b * math.log1p(snr) / math.log(2.0)
                if rate <= 0:
                    raise ValueError("活动链路速率为零")
                dr = np.zeros(self.size)
                dr[jb] = bandwidth_scale * (math.log1p(snr) - snr / (1 + snr)) / math.log(2.0)
                if jp is not None:
                    dr[jp] = a / ((1 + snr) * math.log(2.0))
                duration = (load / 1e6) / rate
                dt = -(load / 1e6) / rate ** 2 * dr
                phase[direction], jac[direction] = duration, dt
                objective += power * duration
                grad += power * dt
                if jp is not None:
                    grad[jp] += duration
            times.append(phase)
            time_grads.append(jac)
        return objective / self.escale, grad / self.escale, times, time_grads

    def original(self, x):
        """原约束统一写成 g(x)<=0，并返回解析 Jacobian。"""
        objective, grad, times, time_grads = self.physical_terms(x)
        values, jac, names = [], [], []

        def add(value, derivative, name):
            """追加一项原约束g<=0、解析Jacobian和约束名称，用于原KKT证书。"""
            values.append(float(value)); jac.append(derivative); names.append(name)

        for j, (lower, upper) in enumerate(self.bounds):
            e = np.zeros(self.size); e[j] = 1.0
            add(lower - x[j], -e, f"lower:{self.names[j]}")
            if upper is not None:
                add(x[j] - upper, e, f"upper:{self.names[j]}")
        for group, total in (("bu", 1.0), ("bd", 1.0)):
            ids = [j for (g, _), j in self.indices.items() if g == group]
            if ids:
                derivative = np.zeros(self.size); derivative[ids] = 1.0 / total
                add(sum(x[j] for j in ids) / total - 1, derivative, f"sum:{group}")
        for phase in self.phases:
            jt = self.indices[phase, -1]
            for i, row in enumerate(times):
                if phase == "train":
                    duration = sum(row.values())
                    derivative = sum(time_grads[i].values()) / self.tscale
                else:
                    if not (self.p.up_bits[i] if phase == "up" else self.p.down_bits[i]):
                        continue
                    duration, derivative = row[phase], time_grads[i][phase] / self.tscale
                derivative = derivative.copy(); derivative[jt] -= 1
                add(duration / self.tscale - x[jt], derivative, f"phase:{phase}:{i}")
        derivative = np.zeros(self.size)
        for phase in self.phases:
            derivative[self.indices[phase, -1]] = 1.0
        fixed = self.p.aggregation_time_s if self.p.kind == "aggregation" else 0.0
        add(sum(x[self.indices[phase, -1]] for phase in self.phases)
            + (fixed - self.p.budget_s) / self.tscale, derivative, "deadline")
        return objective, grad, np.array(values), np.array(jac), names

    def audit(self, x):
        """为原问题构造非负乘子证书，报告原始可行性、互补与驻点残差。

        对 g<=0，KKT 为 grad F+Jg^T lambda=0、lambda>=0、lambda*g=0。
        将驻点和互补方程一起做非负最小二乘；既不复用近端梯度，也不把
        Clarabel 的 gap 当作原问题 KKT。所有尺度在本块开始时固定。
        """
        from scipy.optimize import lsq_linear
        value, grad, constraints, jac, names = self.original(x)
        primal = max(0.0, float(np.max(constraints, initial=0.0)))
        if self.size:
            matrix = np.vstack((jac.T, np.diag(constraints)))
            target = np.concatenate((-grad, np.zeros(len(constraints))))
            fit = lsq_linear(matrix, target, bounds=(0.0, np.inf), tol=1e-12,
                             lsmr_tol=1e-12, max_iter=100)
            dual = fit.x
            stationarity = float(np.max(np.abs(grad + jac.T @ dual), initial=0.0))
            complementarity = float(np.max(np.abs(dual * constraints), initial=0.0))
        else:
            dual = np.zeros(len(constraints)); stationarity = complementarity = 0.0
        dual_violation = max(0.0, float(np.max(-dual, initial=0.0)))
        return dict(original_feasibility=primal, kkt_residual=max(primal, stationarity, complementarity, dual_violation),
                    stationarity_residual=stationarity, complementarity_residual=complementarity,
                    dual_feasibility=dual_violation, objective_scaled=value,
                    multiplier_certificate={name: float(v) for name, v in zip(names, dual) if v > 1e-8})


def solve_block(problem, config, warm_start=None):
    """联合优化频率、上传功率及上下行带宽；成功结果已通过原公式审核。

    热启动优先使用上一块资源。若它或参考资源不可行，先求带统一期限松弛
    的凸可行性恢复。上行能耗参数化后每次仍用原目标 Armijo 线搜索；超过
    迭代/回溯次数、原约束不满足或 KKT 未达标均返回 failed。
    """
    started = time.perf_counter()
    diagnostics = dict(iterations=0, objective_history=[], iteration_records=[],
                       original_feasibility=None, kkt_residual=None,
                       feasibility_recovery=False, cvxpy_solver="CLARABEL")
    latest_resources = None

    def result(status, reason, evaluation=None):
        """统一封装成功或失败状态、实际资源及求解墙钟；失败不生成参考资源回退。"""
        diagnostics["wall_s"] = time.perf_counter() - started
        output = dict(status=status, reason=reason, resources=latest_resources, diagnostics=diagnostics)
        if evaluation is not None:
            output.update(evaluation)
        return output

    try:
        import cvxpy as cp
        if "CLARABEL" not in cp.installed_solvers():
            return result("failed", "未安装 Clarabel 求解器")
        if any((u or d) and h <= 0 for u, d, h in zip(problem.up_bits, problem.down_bits, problem.channel_gains)):
            return result("failed", "活动链路信道增益为零")
        reduced = _ReducedProblem(problem, config)
        opt = config.get("optimization", {})
        mu = float(opt.get("proximal_coefficient", 0.01))
        shrink = float(opt.get("backtracking_ratio", opt.get("backtrack_ratio", 0.5)))
        armijo = float(opt.get("armijo_coefficient", 0.0001))
        max_iterations = int(opt.get("max_iterations", 100))
        max_backtracks = int(opt.get("max_backtracks", 30))
        direction_tol = float(opt.get("direction_tolerance", 1e-5))
        feasibility_tol = float(opt.get("feasibility_tolerance", 1e-6))
        kkt_tol = float(opt.get("kkt_tolerance", 1e-4))
        diagnostics.update(variable_count=reduced.size, active_resource_count=reduced.resource_size,
                           variable_names=[str(v) for v in reduced.names],
                           time_scale_s=reduced.tscale, energy_scale_j=reduced.escale)
        # 完全零负载时只保留聚合常数，不创建零维 CVXPY 变量。
        if reduced.size == 0:
            latest_resources = reduced.unpack(np.empty(0))
            evaluated = evaluate_block(problem, config, latest_resources)
            violation = max(0.0, (evaluated["duration_s"] - problem.budget_s) / reduced.tscale)
            diagnostics.update(original_feasibility=violation, kkt_residual=violation, direction_norm=0.0)
            return result("success" if violation <= feasibility_tol else "failed",
                          "zero_load" if violation <= feasibility_tol else "固定聚合时间超过块预算", evaluated)
        # 任何正通信载荷都需要正信道；零信道不能用人为速率下界伪装可执行。
        if any((u or d) and h <= 0 for u, d, h in zip(problem.up_bits, problem.down_bits, problem.channel_gains)):
            return result("failed", "活动链路信道增益为零")

        x = cp.Variable(reduced.size, name="normalized_resources_and_times")
        previous = cp.Parameter(reduced.size, name="previous_iterate")
        omega = {i: cp.Parameter(nonneg=True) for i, load in enumerate(problem.up_bits) if load}
        beta = {i: cp.Parameter(nonneg=True) for i, load in enumerate(problem.up_bits) if load}
        rate_up, rate_down = {}, {}
        hard, phase_constraints = [], []
        for j, (lower, upper) in enumerate(reduced.bounds):
            hard.append(x[j] >= lower)
            if upper is not None:
                hard.append(x[j] <= upper)
        radio = config["radio"]
        for group, key in (("bu", "total_up_hz"), ("bd", "total_down_hz")):
            ids = [j for (g, _), j in reduced.indices.items() if g == group]
            if ids:
                hard.append(cp.sum(x[ids]) <= 1.0)
        noise, pd = config["channel"]["noise_psd_w_hz"], config["server"]["downlink_power_w"]
        # B 以 MHz、R 以 Mbit/s 表示，D/1e6/R 恢复为秒。
        for i in range(reduced.n):
            a = problem.channel_gains[i] / (noise * 1e6)
            if problem.up_bits[i]:
                b, p = (x[reduced.indices["bu", i]] * radio["total_up_hz"] / 1e6,
                        x[reduced.indices["p", i]])
                capacity = _capacity_expression(cp, b, p, a)
                rate_up[i] = capacity
            if problem.down_bits[i]:
                b = x[reduced.indices["bd", i]] * radio["total_down_hz"] / 1e6
                capacity = _capacity_expression(cp, b, pd, a)
                rate_down[i] = capacity
        compute_energy, down_energy, surrogate_up = 0.0, 0.0, 0.0
        for i in range(reduced.n):
            tc = (problem.client_cycles[i] / 1e9 * cp.inv_pos(x[reduced.indices["f", i]])
                  if problem.client_cycles[i] else 0.0)
            tu = problem.up_bits[i] / 1e6 * cp.inv_pos(rate_up[i]) if problem.up_bits[i] else 0.0
            td = problem.down_bits[i] / 1e6 * cp.inv_pos(rate_down[i]) if problem.down_bits[i] else 0.0
            ts = problem.server_cycles[i] / config["server"]["frequency_hz"]
            if problem.kind == "training" and reduced.phases:
                phase_constraints.append((tc + tu + ts + td) / reduced.tscale <= x[reduced.indices["train", -1]])
            elif problem.kind == "aggregation":
                if problem.up_bits[i]:
                    phase_constraints.append(tu / reduced.tscale <= x[reduced.indices["up", -1]])
                if problem.down_bits[i]:
                    phase_constraints.append(td / reduced.tscale <= x[reduced.indices["down", -1]])
            if problem.client_cycles[i]:
                compute_energy += config["uav"]["kappa"] * problem.client_cycles[i] * 1e18 * cp.square(x[reduced.indices["f", i]])
            if problem.down_bits[i]:
                down_energy += pd * td
            if problem.up_bits[i]:
                p = x[reduced.indices["p", i]]
                # 使用 Mbit/s 时 omega=1/R_M，载荷 D_M；与 bit/s 公式等价。
                surrogate_up += beta[i] + omega[i] * (problem.up_bits[i] / 1e6 * p - beta[i] * rate_up[i])
        fixed = problem.aggregation_time_s if problem.kind == "aggregation" else 0.0
        deadline_expr = (cp.sum(cp.hstack([x[reduced.indices[phase, -1]] for phase in reduced.phases]))
                         + fixed / reduced.tscale)
        convex_constraints = hard + phase_constraints
        # 不收紧原始预算，避免数值误差使当前点被人为排除在后续子问题外。
        numerical_margin = 0.0
        full_constraints = convex_constraints + [deadline_expr <= problem.budget_s / reduced.tscale - numerical_margin]
        surrogate = ((compute_energy + down_energy + surrogate_up) / reduced.escale
                     + mu / 2 * cp.sum_squares(x - previous))
        subproblem = cp.Problem(cp.Minimize(surrogate), full_constraints)
        if not subproblem.is_dcp():
            return result("failed", "近端子问题未通过 CVXPY DCP 检查")
        options = dict(solver="CLARABEL", tol_gap_abs=1e-7, tol_gap_rel=1e-7,
                       tol_feas=1e-8, reduced_tol_gap_abs=1e-8,
                       reduced_tol_gap_rel=1e-8, reduced_tol_feas=1e-8,
                       static_regularization_constant=1e-6,
                       max_iter=300, verbose=False, warm_start=True)

        def run_cvx(model):
            # optimal_inaccurate 仍必须经过原问题残差审核，警告不当作成功。
            """求解同一CVXPY凸子问题；仅允许Clarabel数值正则化重试，原问题审计门限不变。"""
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate")
                warnings.filterwarnings("ignore", message="You are solving a parameterized problem that is not DPP")
                try:
                    model.solve(**options)
                except cp.error.SolverError as error:
                    # 相同凸子问题只调整 Clarabel 的数值正则化再次求解；
                    # 绝不替换求解器、资源政策或原问题的收敛与可行性阈值。
                    diagnostics["clarabel_retries"] = diagnostics.get("clarabel_retries", 0) + 1
                    diagnostics["last_solver_error"] = str(error)
                    retry = dict(options, static_regularization_enable=False,
                                 tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8,
                                 max_iter=500)
                    try:
                        model.solve(**retry)
                    except cp.error.SolverError:
                        # 强 LoS 链路可跨越多个增益数量级，最后一次只加强
                        # 数值正则化；输出仍受原模型的严格残差门限约束。
                        retry.update(static_regularization_enable=True, static_regularization_constant=1e-6,
                                     tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-8)
                        try:
                            model.solve(**retry)
                        except cp.error.SolverError:
                            return False
            return model.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and x.value is not None

        current = None
        for candidate in (warm_start, reference_resources(config, reduced.n)):
            if candidate is None:
                continue
            try:
                trial = reduced.pack(candidate)
                raw_constraints = reduced.original(trial)[2]
                if (np.all(np.isfinite(trial)) and max(raw_constraints, default=0) <= feasibility_tol
                        and raw_constraints[-1] <= -numerical_margin):
                    current = trial
                    break
            except (ValueError, KeyError, IndexError, ZeroDivisionError):
                continue
        if current is None:
            diagnostics["feasibility_recovery"] = True
            # 恢复问题只放松期限；硬件边界和各阶段上图始终保持。最小 slack
            # 为零后仍以原公式审核，不能把有正松弛的结果视为可行执行资源。
            slack = cp.Variable(nonneg=True, name="normalized_deadline_slack")
            reference = reduced.pack(reference_resources(config, reduced.n))
            previous.value = reference
            recovery = cp.Problem(cp.Minimize(slack + 1e-9 * cp.sum_squares(x - reference)),
                                  convex_constraints + [deadline_expr <= problem.budget_s / reduced.tscale - numerical_margin + slack])
            x.value = reference
            if not run_cvx(recovery):
                return result("failed", f"可行性恢复失败：{recovery.status}")
            current = np.asarray(x.value).ravel().copy()
            diagnostics["recovery_slack"] = float(slack.value)
            raw_violation = max(0.0, float(np.max(reduced.original(current)[2])))
            if raw_violation > feasibility_tol:
                diagnostics["original_feasibility"] = raw_violation
                return result("failed", "可行性恢复后原期限/硬件约束仍不满足")

        # 每次参数更新使用当前可行点的真实速率，避免未来信道和上次块参数混用。
        initial_value = reduced.original(current)[0] * reduced.escale
        diagnostics["objective_history"].append(float(initial_value))
        for iteration in range(1, max_iterations + 1):
            diagnostics["iterations"] = iteration
            previous.value = current
            x.value = current
            physical = evaluate_block(problem, config, reduced.unpack(current))
            for i in omega:
                r_mbps = physical["per_uav"][i]["uplink_rate_bps"] / 1e6
                omega[i].value = 1.0 / r_mbps
                beta[i].value = physical["per_uav"][i]["upload_j"]
            if not run_cvx(subproblem):
                latest_resources = reduced.unpack(current)
                return result("failed", f"近端凸子问题失败：{subproblem.status}", physical)
            proposed = np.asarray(x.value).ravel().copy()
            # 辅助时间无能耗目标，取原物理阶段最大值消除无意义的上图松弛。
            # 此操作不会更改任何执行资源，且仍须经过下方原约束审核。
            proposed = reduced.pack(reduced.unpack(proposed))
            direction = proposed - current
            # 各资源已归一化；与原约束/KKT 一样使用无穷范数定义方向残差。
            direction_norm = float(np.linalg.norm(direction, ord=np.inf))
            diagnostics["direction_components"] = {str(name): float(v) for name, v in zip(reduced.names, direction)}
            value, gradient, _, _, _ = reduced.original(current)
            directional_derivative = float(gradient @ direction)
            audit = reduced.audit(current)
            diagnostics.update(audit, direction_norm=direction_norm)
            # 指数锥在平坦目标附近可能给出微小的非下降 SCA 方向。此时额外
            # 核验原问题的投影方向：将 -grad F 投影到当前原约束线性化的可行
            # 多面体。该小型凸QP只生成停止证书，不替代资源优化、不提交新资源。
            # 因而不能用缩小线搜索步长来人为满足方向阈值，也没有固定资源回退。
            projected_norm = None
            if (direction_norm > direction_tol and audit["kkt_residual"] <= kkt_tol
                    and audit["original_feasibility"] <= feasibility_tol):
                _, grad0, g0, jac0, _ = reduced.original(current)
                tangent = cp.Variable(reduced.size)
                certificate = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(tangent) + grad0 @ tangent),
                                         [jac0 @ tangent + g0 <= 0])
                try:
                    certificate.solve(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                                      tol_feas=1e-12, max_iter=200, verbose=False)
                except cp.error.SolverError:
                    # 未取得方向证书时沿用原 SCA 方向判据，不能当作已收敛。
                    pass
                if certificate.status == cp.OPTIMAL and tangent.value is not None:
                    projected_norm = float(np.linalg.norm(tangent.value, ord=np.inf))
            diagnostics["projected_direction_norm"] = projected_norm
            diagnostics["sca_direction_norm"] = direction_norm
            direction_residual = min(direction_norm, projected_norm) if projected_norm is not None else direction_norm
            diagnostics["direction_residual"] = direction_residual
            if (direction_residual <= direction_tol and audit["kkt_residual"] <= kkt_tol
                    and audit["original_feasibility"] <= feasibility_tol):
                latest_resources = reduced.unpack(current)
                return result("success", "converged", physical)
            alpha, accepted = 1.0, False
            for backtrack in range(max_backtracks):
                trial = current + alpha * direction
                trial_value, _, constraints, _, _ = reduced.original(trial)
                violation = max(0.0, float(np.max(constraints, initial=0.0)))
                # tiny roundoff 允许量远低于用户指定的原问题可行性容差。
                if (violation <= feasibility_tol
                        and trial_value <= value + armijo * alpha * directional_derivative + 1e-12
                        and trial_value <= value + 1e-12):
                    accepted = True
                    break
                alpha *= shrink
            diagnostics["iteration_records"].append(dict(iteration=iteration, direction_norm=direction_norm,
                directional_derivative=directional_derivative, step_size=alpha,
                backtracks=backtrack, objective_j=float(value * reduced.escale),
                kkt_residual=audit["kkt_residual"], original_feasibility=audit["original_feasibility"]))
            if not accepted:
                latest_resources = reduced.unpack(current)
                return result("failed", "原目标 Armijo 回溯未找到可行下降步", physical)
            current = trial
            diagnostics["objective_history"].append(float(trial_value * reduced.escale))
        latest_resources = reduced.unpack(current)
        diagnostics.update(reduced.audit(current))
        diagnostics["direction_norm"] = direction_norm
        return result("failed", "达到最大迭代次数，方向范数或原问题 KKT 未达到要求",
                      evaluate_block(problem, config, latest_resources))
    except Exception as error:
        # 求解器异常是可报告的任务失败；资源绝不在此替换成均分/固定参考值。
        return result("failed", f"{type(error).__name__}: {error}")
