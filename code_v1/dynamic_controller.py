"""作用：周期切分选择、epoch 块预算分配与独立信道参考预演。

运行方式（在 code_v1 目录）：
    python -X utf8 -B train_sfl.py --prepare-only --epochs 100
    python -X utf8 -B train_sfl.py --help

本模块没有独立训练入口，由 train_sfl.py 调用。参考预演只按 FLOP
画像计算物理成本，不读取图像、不执行前后向传播、不更新神经网络。
它以独立校准信道生成各 epoch 的参考时长，再乘给定期限倍率，形成
正式训练不可互相借用的 epoch 预算。训练时按实际已知快照分配块预算。

切分对应正文式（26）—（28）：最近 W 次“块起始观测”的平均增益用于
周期能耗评分，当前已知增益用于参考可行性检查。整个周期固定切点，
按无人机逐一扫描候选，平局保留现有切点。这里既不读取未来 CSI，
也不把未来尚未采到的图片计入候选负载；实际块资源由 resource_solver
单独求解。全部载荷以 bit、时间以 s、能量以 J、频率以 Hz 计。
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Mapping, Sequence

from transmission import rate_bps


class SelectionInfeasible(RuntimeError):
    """参考切分没有可行初值；record 保存本次失败的完整决策依据。"""

    def __init__(self, record):
        """保存不可行选择的完整记录，并将明确的失败理由传给调用方。"""
        self.record = record
        super().__init__(record["failure_reason"])


def _vector(value, n, name):
    """将按 UAV 编号的字典、等长序列或共同标量转换为独立列表。"""
    if isinstance(value, Mapping):
        try:
            result = [value[i] if i in value else value[str(i)] for i in range(n)]
        except KeyError as exc:
            raise ValueError(f"{name} 缺少 UAV 编号") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
    else:
        result = [value] * n
    if len(result) != n:
        raise ValueError(f"{name} 必须恰好包含 {n} 个元素")
    return result


def _inputs(config, profiles, cuts, sample_counts, gains):
    """校验共同维度、实际样本量及线性信道增益，避免字典顺序改变机号。"""
    n = config["mission"]["num_uavs"]
    cuts = _vector(cuts, n, "cuts")
    counts = _vector(sample_counts, n, "sample_counts")
    channels = _vector(gains, n, "gains")
    if any(cut not in profiles for cut in cuts):
        raise ValueError("切点没有对应的计算与通信画像")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
           for count in counts):
        raise ValueError("每机样本数必须为非负整数")
    if any(not isinstance(gain, (float, int)) or isinstance(gain, bool)
           or not math.isfinite(gain) or gain <= 0 for gain in channels):
        raise ValueError("参考决策需要有限正线性信道增益")
    return cuts, counts, [float(gain) for gain in channels]


def batch_sample_counts(sample_counts, batch_size):
    """返回每个同步训练块的逐机样本数，保留不足 batch_size 的尾批。

    当前 IID 场景各机样本量相同；同时允许测试中的零负载分支，零样本
    分支不生成通信载荷。返回顺序就是 epoch 内的执行顺序。
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    if not sample_counts:
        return []
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
           for count in sample_counts):
        raise ValueError("样本量必须为非负整数")
    return [[min(batch_size, max(0, count - offset)) for count in sample_counts]
            for offset in range(0, max(sample_counts), batch_size)]


def samples_at(time_s, collection):
    """按原始固定采集速率取累计样本数，延长飞行时域不重新计算采集速率。

    D(t)=min(D_max,D_0+floor(lambda*t))。这里只使用参数和当前时间；
    不以原始 600 s 的场景终点限制校准，也不读取未来图片或样本索引。
    """
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("采集查询时间必须有限非负")
    maximum = collection["max_samples_per_uav"]
    initial = collection["initial_samples_per_uav"]
    rate = collection["rate_samples_per_s"]
    if rate == 0 or initial == maximum:
        return initial
    if time_s >= (maximum - initial) / rate:
        return maximum
    return min(maximum, initial + math.floor(rate * time_s + 1e-10))


def _reference_rates(config, gains):
    """相同参考频率/功率、上下行各均分带宽下的逐机 Shannon 速率。"""
    n = len(gains)
    up_hz = config["radio"]["total_up_hz"] / n
    down_hz = config["radio"]["total_down_hz"] / n
    p_up = config["uav"]["p_up_ref_w"]
    p_down = config["server"]["downlink_power_w"]
    n0 = config["channel"]["noise_psd_w_hz"]
    return ([rate_bps(up_hz, p_up, gain, n0) for gain in gains],
            [rate_bps(down_hz, p_down, gain, n0) for gain in gains])


def _sum_energy(parts):
    """给参考账本补充系统总能耗，固定飞行能耗不参与切分评分。"""
    parts["modeled_system_training_j"] = math.fsum(
        parts[key] for key in ("uav_compute_j", "server_compute_j", "uav_transmit_j",
                               "server_transmit_j", "aggregation_j"))
    return parts


def reference_training_cost(config, profiles, cuts, sample_counts, gains):
    """冻结当前 CSI，计算一个训练块的参考物理时长与系统能耗。

    各机顺序为前缀计算、激活与标签上行、服务器后缀计算、梯度下行、
    前缀反传；求共同完成时间时，前后缀同侧计算可合并。不同 UAV
    分支并行，块时长取各分支完整时长的最大值，能耗则逐机相加。
    """
    cuts, counts, channels = _inputs(config, profiles, cuts, sample_counts, gains)
    up_rates, down_rates = _reference_rates(config, channels)
    uav, server = config["uav"], config["server"]
    f, fe = uav["f_ref_hz"], server["frequency_hz"]
    client_s, server_s, up_s, down_s = [], [], [], []
    client_j, server_j, up_j, down_j = [], [], [], []
    for i, (cut, count) in enumerate(zip(cuts, counts)):
        profile = profiles[cut]
        c = count * profile["client_flops_per_sample"] * uav["cycles_per_flop"]
        d = count * profile["server_flops_per_sample"] * server["cycles_per_flop"]
        a = count * (profile["activation_bytes_per_sample"]
                     + profile["label_bytes_per_sample"]) * 8
        g = count * profile["gradient_bytes_per_sample"] * 8
        client_s.append(c / f)
        server_s.append(d / fe)
        up_s.append(a / up_rates[i])
        down_s.append(g / down_rates[i])
        client_j.append(uav["kappa"] * c * f ** 2)
        server_j.append(server["kappa"] * d * fe ** 2)
        up_j.append(uav["p_up_ref_w"] * up_s[-1])
        down_j.append(server["downlink_power_w"] * down_s[-1])
    branch_s = [math.fsum(parts) for parts in zip(client_s, up_s, server_s, down_s)]
    return _sum_energy(dict(
        duration_s=max(branch_s, default=0.0), branch_durations_s=branch_s,
        client_durations_s=client_s, server_durations_s=server_s,
        up_durations_s=up_s, down_durations_s=down_s,
        uav_compute_j=math.fsum(client_j), server_compute_j=math.fsum(server_j),
        uav_transmit_j=math.fsum(up_j), server_transmit_j=math.fsum(down_j),
        aggregation_j=0.0))


def reference_aggregation_cost(config, profiles, old_cuts, new_cuts, gains, *,
                               final=False, reserve_next_down=False):
    """参考聚合成本：旧前缀上传、聚合、新前缀下发分别同步。

    reserve_next_down=True 只用于 epoch 开始时预留预算：所有 UAV
    的下行载荷都取候选前缀状态的最大字节数，避免未知的下周期切分
    消耗未预留时间。末次聚合 final=True 时没有下发，也没有下行变量。
    """
    old_cuts, _, channels = _inputs(config, profiles, old_cuts, 0, gains)
    new_cuts = _vector(new_cuts, len(old_cuts), "new_cuts")
    if any(cut not in profiles for cut in new_cuts):
        raise ValueError("新切点缺少画像")
    up_rates, down_rates = _reference_rates(config, channels)
    candidates = config["selection"]["candidates"]
    maximum = max(profiles[cut]["prefix_state_bytes"] for cut in candidates)
    up_bits = [profiles[cut]["prefix_state_bytes"] * 8 for cut in old_cuts]
    down_bits = [0 if final else (maximum if reserve_next_down else
                 profiles[cut]["prefix_state_bytes"]) * 8 for cut in new_cuts]
    up_s = [bits / rate for bits, rate in zip(up_bits, up_rates)]
    down_s = [bits / rate for bits, rate in zip(down_bits, down_rates)]
    tau = config["server"]["aggregation_time_s"]
    return _sum_energy(dict(
        duration_s=max(up_s, default=0.0) + tau + max(down_s, default=0.0),
        up_durations_s=up_s, down_durations_s=down_s,
        uav_compute_j=0.0, server_compute_j=0.0,
        uav_transmit_j=config["uav"]["p_up_ref_w"] * math.fsum(up_s),
        server_transmit_j=config["server"]["downlink_power_w"] * math.fsum(down_s),
        aggregation_j=config["server"]["aggregation_power_w"] * tau))


def make_training_problem(config, profiles, cuts, sample_counts, gains, budget_s):
    """将实际切点、尾批样本量和块起始 CSI 转成求解器的 SI 单位负载。"""
    from resource_solver import BlockProblem

    cuts, counts, channels = _inputs(config, profiles, cuts, sample_counts, gains)
    return BlockProblem(
        kind="training", uav_ids=list(range(len(cuts))), channel_gains=channels,
        client_cycles=[m * profiles[cut]["client_flops_per_sample"]
                       * config["uav"]["cycles_per_flop"] for cut, m in zip(cuts, counts)],
        server_cycles=[m * profiles[cut]["server_flops_per_sample"]
                       * config["server"]["cycles_per_flop"] for cut, m in zip(cuts, counts)],
        up_bits=[m * (profiles[cut]["activation_bytes_per_sample"]
                      + profiles[cut]["label_bytes_per_sample"]) * 8
                 for cut, m in zip(cuts, counts)],
        down_bits=[m * profiles[cut]["gradient_bytes_per_sample"] * 8
                   for cut, m in zip(cuts, counts)], budget_s=float(budget_s),
        aggregation_time_s=0.0)


def make_aggregation_problem(config, profiles, old_cuts, new_cuts, gains,
                             budget_s, final=False):
    """旧切点决定上传载荷，新切点决定下发载荷，最后一次下发置零。"""
    from resource_solver import BlockProblem

    old_cuts, _, channels = _inputs(config, profiles, old_cuts, 0, gains)
    new_cuts = _vector(new_cuts, len(old_cuts), "new_cuts")
    if any(cut not in profiles for cut in new_cuts):
        raise ValueError("新切点缺少画像")
    n = len(old_cuts)
    return BlockProblem(
        kind="aggregation", uav_ids=list(range(n)), channel_gains=channels,
        client_cycles=[0.0] * n, server_cycles=[0.0] * n,
        up_bits=[profiles[cut]["prefix_state_bytes"] * 8 for cut in old_cuts],
        down_bits=[0 if final else profiles[cut]["prefix_state_bytes"] * 8
                   for cut in new_cuts], budget_s=float(budget_s),
        aggregation_time_s=config["server"]["aggregation_time_s"])


def allocate_epoch_budgets(config, profiles, cuts, sample_counts, gains,
                           epoch_budget_s, will_aggregate, final=False):
    """按轮初已知负载的参考时长比例分配固定 epoch 预算。

    一个 epoch 的预算完全在本 epoch 内分配，不借用后续预算；到达下一
    个 epoch 后重新冻结样本快照。聚合（若适用）为独立块，非末次时为
    最大候选前缀预留下行。后续块的实际 CSI 不参与轮初预算计算。
    """
    cuts, counts, channels = _inputs(config, profiles, cuts, sample_counts, gains)
    if not math.isfinite(epoch_budget_s) or epoch_budget_s <= 0:
        raise ValueError("epoch 时间预算必须有限且为正")
    batches = batch_sample_counts(counts, config["training"]["batch_size"])
    # 大多数块都是同样的满批，仅尾批不同；缓存参考成本避免候选扫描中
    # 为同一负载重复计算 Shannon 对数。缓存仅在本次已知 CSI 下有效。
    cache, weights = {}, []
    for batch in batches:
        key = tuple(batch)
        if key not in cache:
            cache[key] = reference_training_cost(config, profiles, cuts, batch, channels)["duration_s"]
        weights.append(cache[key])
    aggregation_weight = (reference_aggregation_cost(
        config, profiles, cuts, cuts, channels, final=final,
        reserve_next_down=True)["duration_s"] if will_aggregate else 0.0)
    total = math.fsum(weights) + aggregation_weight
    all_weights = weights + ([aggregation_weight] if will_aggregate else [])
    allocations = ([epoch_budget_s * weight / total for weight in all_weights]
                   if total else [0.0] * len(all_weights))
    if allocations and total:
        # 最后一个块吸收浮点舍入余量，各块预算和严格等于本轮固定预算。
        allocations[-1] = epoch_budget_s - math.fsum(allocations[:-1])
    return dict(
        epoch_budget_s=float(epoch_budget_s), batch_budgets_s=allocations[:len(weights)],
        aggregation_budget_s=(allocations[-1] if will_aggregate and allocations else 0.0),
        reference_batch_durations_s=weights,
        reference_aggregation_duration_s=aggregation_weight,
        reference_epoch_duration_s=total, batch_sample_counts=batches,
        unused_budget_s=(float(epoch_budget_s) if total == 0 else 0.0))


class DynamicController:
    """周期切分控制器；窗口只保存显式 observe 调用取得的块起始 CSI。"""

    def __init__(self, config, profiles, trace):
        """保存配置、画像和信道读取器；只建立有界历史窗口，不主动读取未来状态。"""
        self.config, self.profiles, self.trace = config, profiles, trace
        self.n = config["mission"]["num_uavs"]
        self.history = deque(maxlen=config["selection"]["channel_window_size"])
        self.observation_count = 0
        self.selection_count = 0

    def observe(self, time_s):
        """在块开始时观测一次 CSI；同一时刻的不同块也各算一次观测。"""
        gains = _vector(self.trace.channel_at(time_s), self.n, "gains")
        if any(not math.isfinite(gain) or gain <= 0 for gain in gains):
            raise ValueError("块起始 CSI 必须为有限正增益")
        self.history.append((float(time_s), list(gains)))
        self.observation_count += 1
        return dict(enumerate(gains))

    def _scores(self, channels, counts, first_period):
        """正文式（27）的逐机可加评分；总分另加一次服务器聚合常数。"""
        config = self.config
        uav, server = config["uav"], config["server"]
        f, fe = uav["f_ref_hz"], server["frequency_hz"]
        up, down = _reference_rates(config, channels)
        period = config["training"]["aggregation_period"]
        chi = 0 if first_period else 1
        scores = []
        for i, count in enumerate(counts):
            per_uav = {}
            for cut in config["selection"]["candidates"]:
                profile = self.profiles[cut]
                c = profile["client_flops_per_sample"] * uav["cycles_per_flop"]
                d = profile["server_flops_per_sample"] * server["cycles_per_flop"]
                a = (profile["activation_bytes_per_sample"] + profile["label_bytes_per_sample"]) * 8
                g = profile["gradient_bytes_per_sample"] * 8
                m = profile["prefix_state_bytes"] * 8
                per_uav[cut] = period * count * math.fsum((
                    uav["kappa"] * f ** 2 * c, server["kappa"] * fe ** 2 * d,
                    uav["p_up_ref_w"] * a / up[i], server["downlink_power_w"] * g / down[i]))
                per_uav[cut] += m * (uav["p_up_ref_w"] / up[i]
                                              + chi * server["downlink_power_w"] / down[i])
            scores.append(per_uav)
        return scores

    def _fastest(self, channels, counts, current_cuts, first_period):
        """按逐机参考链路时长构造最快初值；相同耗时优先原切点。"""
        config = self.config
        up, down = _reference_rates(config, channels)
        uav, server = config["uav"], config["server"]
        f, fe = uav["f_ref_hz"], server["frequency_hz"]
        period = config["training"]["aggregation_period"]
        chi = 0 if first_period else 1
        fastest = []
        for i, count in enumerate(counts):
            def latency(cut):
                """计算该机候选切分的参考周期时长，仅使用已知样本量和当前CSI。"""
                p = self.profiles[cut]
                work = (p["client_flops_per_sample"] * uav["cycles_per_flop"] / f
                        + p["server_flops_per_sample"] * server["cycles_per_flop"] / fe
                        + (p["activation_bytes_per_sample"] + p["label_bytes_per_sample"]) * 8 / up[i]
                        + p["gradient_bytes_per_sample"] * 8 / down[i])
                sync = p["prefix_state_bytes"] * 8 * (1 / up[i] + chi / down[i])
                return period * count * work + sync
            order = [current_cuts[i]] + [c for c in config["selection"]["candidates"]
                                         if c != current_cuts[i]]
            best = order[0]
            for cut in order[1:]:
                if latency(cut) < latency(best) - 1e-12:
                    best = cut
            fastest.append(best)
        return fastest

    def _feasible(self, cuts, counts, current_gains, budgets):
        """仅用已知 D、当前 CSI 检查未来周期固定 epoch 预算及当前下发。"""
        if budgets is None:
            return True, {"checked": False}
        if not isinstance(budgets, Mapping):
            budgets = {"epoch_budgets_s": list(budgets)}
        epoch_budgets = list(budgets.get("epoch_budgets_s", []))
        checks, feasible = [], True
        final_period = bool(budgets.get("final_period", False))
        for position, budget in enumerate(epoch_budgets):
            # 选择发生在周期边界；尚未到达的新数据不进入估计。只在本周期
            # 最后一个 epoch 预留聚合，若为最终周期则删除下发载荷。
            aggregate = position == len(epoch_budgets) - 1
            allocation = allocate_epoch_budgets(
                self.config, self.profiles, cuts, counts, current_gains, budget,
                will_aggregate=aggregate, final=aggregate and final_period)
            required = allocation["reference_epoch_duration_s"]
            okay = required <= budget + max(1e-9, abs(budget) * 1e-10)
            feasible = feasible and okay
            checks.append(dict(epoch_offset=position, required_s=required,
                               budget_s=budget, feasible=okay))
        incoming = budgets.get("incoming_aggregation")
        incoming_check = None
        if incoming is not None:
            value = reference_aggregation_cost(
                self.config, self.profiles, incoming["old_cuts"], cuts,
                incoming.get("gains", current_gains), final=False)["duration_s"]
            budget = incoming["budget_s"]
            okay = value <= budget + max(1e-9, abs(budget) * 1e-10)
            feasible = feasible and okay
            incoming_check = dict(required_s=value, budget_s=budget, feasible=okay)
        return feasible, dict(checked=True, epoch_checks=checks,
                              incoming_aggregation=incoming_check)

    def select(self, time_s, sample_counts, current_cuts, first_period=False, budgets=None):
        """逐机坐标扫描，返回逐机切点和可序列化记录，不增加窗口观测数。

        budgets 可含 epoch_budgets_s（下一周期各轮预算）、final_period
        及 incoming_aggregation={budget_s,old_cuts,gains}。初始参考预演
        尚无正式预算时传 None。没有参考可行初值时抛 SelectionInfeasible，
        调用方应记录并停止，不能把失败改写成固定资源执行。
        """
        current = _vector(current_cuts, self.n, "current_cuts")
        now = _vector(self.trace.channel_at(time_s), self.n, "gains")
        current, counts, now = _inputs(self.config, self.profiles, current, sample_counts, now)
        if first_period or not self.history:
            mean = now
        else:
            mean = [math.fsum(item[1][i] for item in self.history) / len(self.history)
                    for i in range(self.n)]
        scores = self._scores(mean, counts, first_period)
        constant = (self.config["server"]["aggregation_power_w"]
                    * self.config["server"]["aggregation_time_s"])
        score = lambda cuts: math.fsum(scores[i][cut] for i, cut in enumerate(cuts)) + constant
        record = dict(selection_index=self.selection_count, time_s=float(time_s),
                      first_period=bool(first_period), sample_counts=counts,
                      current_cuts=list(current), window_observations=(0 if first_period else len(self.history)),
                      total_block_observations=self.observation_count,
                      window_channel_gains=list(mean), current_channel_gains=list(now),
                      score_before_j=score(current), candidate_records=[])
        self.selection_count += 1
        feasible, details = self._feasible(current, counts, now, budgets)
        chosen = list(current)
        record["initial_reference_feasibility"] = details
        record["initialization"] = "current_cuts"
        if not feasible:
            chosen = self._fastest(now, counts, current, first_period)
            feasible, details = self._feasible(chosen, counts, now, budgets)
            record["initialization"] = "fastest_reference_cuts"
            record["fastest_reference_cuts"] = list(chosen)
            record["fastest_reference_feasibility"] = details
        if not feasible:
            record.update(status="infeasible", selected_cuts=list(chosen),
                          failure_reason="当前切分及逐机最快参考切分均不能满足已分配的时间预算")
            raise SelectionInfeasible(record)
        tolerance = self.config["selection"]["energy_tolerance_j"]
        for sweep in range(self.config["selection"]["max_sweeps"]):
            changed = False
            for i in range(self.n):
                best, best_score = chosen[i], score(chosen)
                # 只接受超过容差的严格下降；完全相同或数值平局保留当前点。
                for candidate in self.config["selection"]["candidates"]:
                    if candidate == chosen[i]:
                        continue
                    trial = list(chosen)
                    trial[i] = candidate
                    okay, check = self._feasible(trial, counts, now, budgets)
                    value = score(trial)
                    record["candidate_records"].append(dict(
                        sweep=sweep, uav_id=i, candidate=candidate, feasible=okay,
                        score_j=value, delta_j=value - score(chosen), reference_check=check))
                    if okay and value < best_score - tolerance:
                        best, best_score = candidate, value
                if best != chosen[i]:
                    chosen[i], changed = best, True
            if not changed:
                break
        record.update(status="selected", selected_cuts=list(chosen),
                      score_after_j=score(chosen),
                      changed_uav_ids=[i for i in range(self.n) if chosen[i] != current[i]],
                      reference_feasibility=self._feasible(chosen, counts, now, budgets)[1])
        return chosen, record


def calibrate(config, profiles, collection, trace, epochs):
    """独立信道上的参考资源预演，生成每 epoch 固定预算和总期限。

    trace 应为可连续查询的独立校准信道（默认正式种子加一），而不是正式
    CSV。查询严格按物理时间推进；每块冻结其起始 CSI。采集始终按原始
    collection.rate_samples_per_s，饱和后保持容量不再增加。预演没有
    神经网络计算或权重更新，墙钟耗时单独记录，不加入物理参考时长。
    """
    wall_start = time.perf_counter()
    period = config["training"]["aggregation_period"]
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1 or epochs % period:
        raise ValueError("参考预演 epoch 数必须为正且为聚合周期的整数倍")
    n = config["mission"]["num_uavs"]
    controller = DynamicController(config, profiles, trace)
    cuts = [config["selection"]["initial_split"]] * n
    cuts, first_record = controller.select(0.0, [samples_at(0.0, collection)] * n,
                                          cuts, first_period=True)
    selections, records, aggregations = [first_record], [], []
    clock, block_count = 0.0, 0
    energy_keys = ("uav_compute_j", "server_compute_j", "uav_transmit_j",
                   "server_transmit_j", "aggregation_j", "modeled_system_training_j")
    totals = dict.fromkeys(energy_keys, 0.0)
    for epoch in range(1, epochs + 1):
        start = clock
        counts = [samples_at(clock, collection)] * n
        epoch_cuts = list(cuts)
        epoch_energy = dict.fromkeys(energy_keys, 0.0)
        batches = batch_sample_counts(counts, config["training"]["batch_size"])
        for batch in batches:
            gains = controller.observe(clock)
            cost = reference_training_cost(config, profiles, cuts, batch, gains)
            clock += cost["duration_s"]
            block_count += 1
            for key in energy_keys:
                epoch_energy[key] += cost[key]
        training_end = clock
        if epoch % period == 0:
            final = epoch == epochs
            gains = controller.observe(clock)
            old = list(cuts)
            if not final:
                # 下一周期切分在当前聚合前决定：旧前缀上传，聚合完整模型，
                # 再按新切分下发。评分只用聚合开始时已经到达的数据量。
                known = [samples_at(clock, collection)] * n
                cuts, selection = controller.select(clock, known, cuts)
                selections.append(selection)
            cost = reference_aggregation_cost(config, profiles, old, cuts, gains, final=final)
            aggregations.append(dict(epoch=epoch, start_s=clock, old_cuts=old,
                                     new_cuts=list(cuts), final=final, **cost))
            clock += cost["duration_s"]
            block_count += 1
            for key in energy_keys:
                epoch_energy[key] += cost[key]
        for key in energy_keys:
            totals[key] += epoch_energy[key]
        records.append(dict(epoch=epoch, start_s=start, training_end_s=training_end,
                            end_s=clock, duration_s=clock - start,
                            sample_counts=counts, cuts=epoch_cuts,
                            batches=len(batches), tail_batch_samples=(batches[-1] if batches else [0] * n),
                            costs=epoch_energy))
    multiplier = config.get("optimization", {}).get("deadline_factor", 1.2)
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("参考时长的期限倍率必须为不小于 1 的有限数")
    durations = [record["duration_s"] for record in records]
    budgets = [multiplier * duration for duration in durations]
    return dict(schema_version=1, kind="independent_channel_reference_preflight",
                epochs=epochs, aggregation_period=period, aggregations=epochs // period,
                channel_seed=getattr(trace, "seed", None), total_time_s=clock,
                deadline_multiplier=multiplier, deadline_s=multiplier * clock,
                epoch_durations_s=durations, epoch_budgets_s=budgets,
                records=records, period_records=selections, aggregation_records=aggregations,
                block_count=block_count, block_start_observations=controller.observation_count,
                collection=dict(collection), costs=totals,
                wall_time_s=time.perf_counter() - wall_start,
                model_updates=0, csi_policy="freeze_at_block_start")
