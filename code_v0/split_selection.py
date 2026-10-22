"""用途：集中放置切分层选择算法，不改变训练数据、轮数和模型权重。
训练入口：python split_dynamic.py --data output/data_load_7500/data.json
独立能耗规划：python split_selection.py --data output/data_load_7500/data.json --rounds 100
完整参数：python split_selection.py --help

choose_latency_cuts 保留现有 GFLOPS/Mbps 时延选择，供真实 dynamic 训练调用。
select_tradeoff_cuts 根据系统总能耗比较候选切点，仅使用固定参考资源和当前信道。
独立规划不训练网络、不求解资源优化，也不把估算能耗当作真实设备测量值。
"""
from __future__ import annotations

import math


def choose_latency_cuts(profiles, sizes, compute, bandwidth, server_gflops, period):
    """旧时延算法原样保留：每架 UAV 独立选择估算一轮耗时最短的切点。

    此接口的 compute/server_gflops 单位是 GFLOPS，bandwidth 单位是 Mbps。
    它与下面使用 Hz/W/cycles 的物理模型是两种独立口径，不能混用单位。
    """
    cuts, costs = [], []
    for size, capacity, rate in zip(sizes, compute, bandwidth):
        by_cut = {}
        for cut, p in profiles.items():
            compute_s = size * (p['client_flops_per_sample'] / (capacity * 1e9)
                                + p['server_flops_per_sample'] / (server_gflops * 1e9))
            communication_s = size * 8 * (p['activation_bytes_per_sample'] + p['gradient_bytes_per_sample']
                                         + p['label_bytes_per_sample']) / (rate * 1e6)
            # 原算法将前缀上传与下发均摊到 K 轮；保持原有训练成本和打平时的选择顺序。
            communication_s += 2 * p['prefix_state_bytes'] * 8 / (rate * 1e6 * period)
            by_cut[cut] = compute_s + communication_s
        chosen = min(by_cut, key=by_cut.get)
        cuts.append(chosen)
        costs.append(by_cut)
    return cuts, costs


def _finite_or_none(value):
    """不可行候选可能没有有限目标值；日志用 null 表示，禁止输出 Infinity。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def select_tradeoff_cuts(config, profiles, sample_counts, channel_gain, reference_resources,
                         previous_cuts, period, *, preloaded=False, transition_cuts=None):
    """固定资源的坐标扫描：按完整系统能耗差依次调整各 UAV 的切点。

    一个 period 包含 K 个完整本地 epoch。每次只换一架 UAV 的切点，其他
    UAV 保持当前值，重新计算三个同步阶段的最大时延与全体悬停能耗。
    这使得比较具有式（29）—（30）的计算/传输/同步/悬停 trade-off，
    而不是分别最小化各架 UAV 的能耗。默认只扫描一次，不声称全局最优。

    previous_cuts 仅作为拟执行周期的比较起点，不代表本轮要上传的切点。
    本轮当前前缀上传对选择下一切点是固定成本，不进入候选评分；候选
    上传属于拟执行周期的末尾。preloaded=True 时只免除首周期的初始下发。
    整个候选周期用决策时已观察到的信道评估，不读取未来信道。
    transition_cuts 非空时还检查“当前上传、候选下发”的实际过渡轮预算，
    但不把该轮成本重复加进候选能耗差。
    """
    from resource_allocation import build_round_problem, evaluate_cycle_reference, evaluate_resources

    options = config['selection']
    candidates = list(options['candidates'])
    cuts = list(previous_cuts)
    count = len(sample_counts)
    if not count or len(cuts) != count or len(channel_gain) != count:
        raise ValueError('切点、信道和非空 UAV 样本数列表必须等长')
    if transition_cuts is not None:
        transition_cuts = list(transition_cuts)
        if len(transition_cuts) != count or any(cut not in profiles for cut in transition_cuts):
            raise ValueError('实际过渡轮切点必须与 UAV 数量匹配，并包含在画像中')
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('候选切点必须非空且不能重复')
    if any(cut not in profiles for cut in candidates + cuts):
        raise ValueError('所有候选与当前切点必须有模型画像')
    if any(cut not in candidates for cut in cuts):
        raise ValueError('当前切点必须包含在候选集合中')
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError('拟执行周期 period 必须是正整数')
    sweeps = options['max_sweeps']
    tolerance = options['energy_tolerance_j']
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps <= 0:
        raise ValueError('max_sweeps 必须是正整数')
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('energy_tolerance_j 必须为非负有限数')

    def evaluate(proposal):
        # 不调用 allocate_resources：各候选共享完全相同的参考资源，比较才有效。
        result = evaluate_cycle_reference(config, profiles, sample_counts, channel_gain,
                                          proposal, reference_resources, period,
                                          preloaded=preloaded)
        # 下一个周期的起始下发实际发生在当前聚合轮末：本轮预算必须单独检查。
        # 候选周期评分已计入该次下发；这里只判断可行性，避免能耗重复计数。
        result = dict(result)
        result['transition_violations'] = []
        if transition_cuts is not None:
            transition = build_round_problem(config, profiles, sample_counts, channel_gain,
                                             transition_cuts, next_cuts=proposal,
                                             aggregation_due=True, final_round=False)
            checked = evaluate_resources(transition, reference_resources)
            result['transition_violations'] = checked['constraints']['violations']
            result['feasible'] = bool(result['feasible'] and checked['feasible'])
        if result['feasible'] and (_finite_or_none(result['cycle_energy_j']) is None
                                   or _finite_or_none(result['cycle_time_s']) is None):
            raise ValueError('可行资源评价必须返回有限的能耗和时延')
        return result

    current = evaluate(cuts)
    audit = []
    completed = 0
    for sweep in range(sweeps):
        changed = False
        for uav in range(count):
            before_cut = cuts[uav]
            before = current
            best_cut, best = before_cut, before
            candidate_audit = []
            for candidate in candidates:
                proposal = cuts[:]
                proposal[uav] = candidate
                scored = before if candidate == before_cut else evaluate(proposal)
                feasible = bool(scored['feasible'])
                delta = (scored['cycle_energy_j'] - before['cycle_energy_j']
                         if feasible and before['feasible'] else None)
                candidate_audit.append(dict(
                    cut=candidate, feasible=feasible,
                    cycle_energy_j=_finite_or_none(scored.get('cycle_energy_j')),
                    cycle_time_s=_finite_or_none(scored.get('cycle_time_s')),
                    delta_energy_j=_finite_or_none(delta),
                    violations=(scored['constraints'].get('violations', [])
                                if isinstance(scored.get('constraints'), dict) else []),
                    transition_violations=scored['transition_violations']))
                # 当前不可行时允许换到可行点；当前可行时只有严格降低能耗才换点。
                # 能耗打平保留原切点，不用切点编号或更短时延额外破坏平局。
                if feasible and (not best['feasible'] or
                                 scored['cycle_energy_j'] < best['cycle_energy_j'] - tolerance):
                    best_cut, best = candidate, scored
            if not best['feasible']:
                raise ValueError(f'UAV {uav + 1} 在固定参考资源下没有可行候选切点；'
                                 '请检查预算、参考资源与候选集合')
            cuts[uav], current = best_cut, best
            if best_cut != before_cut:
                changed = True
            audit.append(dict(sweep=sweep + 1, uav=uav + 1, previous_cut=before_cut,
                              selected_cut=best_cut,
                              delta_energy_j=_finite_or_none(
                                  best['cycle_energy_j'] - before['cycle_energy_j']
                                  if before['feasible'] else None),
                              candidates=candidate_audit))
        completed = sweep + 1
        if not changed:
            break
    return dict(cuts=cuts, audit=audit, cycle_energy_j=current['cycle_energy_j'],
                cycle_time_s=current['cycle_time_s'], feasible=bool(current['feasible']),
                objective='system total energy at fixed reference resources',
                algorithm='coordinate trade-off; no claim of global optimality',
                sweeps_completed=completed, period=period, initial_prefix_preloaded=preloaded,
                optimized=False, energy_measured=False)


def run_reference_plan(config, workload, profiles):
    """根据当前信道逐周期规划，再按实际上下行顺序评价每一个 SL epoch。

    注意：计划完成后才汇总输出，但选择第 r 轮切点时不会读取 r+1 的信道。
    这里不加载样本张量或执行梯度更新；样本数量只用于工作量与估算成本。
    """
    from resource_allocation import (
        build_round_problem, evaluate_resources, observe_channel, reference_resources,
    )

    sizes = workload['sizes']
    rounds = config['training']['rounds']
    period = config['training']['aggregation_period']
    resources = reference_resources(config)
    cuts = [config['selection']['initial_split']] * len(sizes)
    history, schedule = [], []
    for epoch in range(rounds):
        channel = observe_channel(config, epoch)
        gains = channel['gain']
        if epoch == 0:
            # 首周期尚未训练，选择后假设其前缀已预加载，不计初始模型分发。
            selected = select_tradeoff_cuts(config, profiles, sizes, gains, resources, cuts,
                                            min(period, rounds), preloaded=True)
            cuts = selected['cuts'][:]
            schedule.append(dict(decision_after_round=0, effective_round=1, **selected))
        final = epoch + 1 == rounds
        aggregation_due = (epoch + 1) % period == 0 or final
        next_cuts = None
        if aggregation_due and not final:
            # 在本轮上传之前选下周期切点。本轮上传的模型仍是 cuts，之后下发 next_cuts。
            # 当前上传成本不随 next_cuts 改变，因此不加入下一周期候选差值。
            selected = select_tradeoff_cuts(config, profiles, sizes, gains, resources, cuts,
                                            min(period, rounds - epoch - 1), preloaded=False,
                                            transition_cuts=cuts)
            next_cuts = selected['cuts'][:]
            schedule.append(dict(decision_after_round=epoch + 1,
                                 effective_round=epoch + 2, **selected))
        problem = build_round_problem(config, profiles, sizes, gains, cuts,
                                      next_cuts=next_cuts, aggregation_due=aggregation_due,
                                      final_round=final)
        evaluated = evaluate_resources(problem, resources)
        history.append(dict(round=epoch + 1, cuts=cuts[:], next_cuts=next_cuts,
                            channel=channel,
                            aggregation_due=aggregation_due, final_round=final, **evaluated))
        if next_cuts is not None:
            cuts = next_cuts
    return history, schedule


def main(argv=None):
    from resource_allocation import planning_parser, prepare_planning, save_reference_plan

    parser = planning_parser(__doc__)
    args = parser.parse_args(argv)
    try:
        config, workload, profiles, output = prepare_planning(args)
        history, schedule = run_reference_plan(config, workload, profiles)
        summary = save_reference_plan(output, config, workload, profiles, history, schedule,
                                      label='tradeoff_reference_plan')
        print(f'参考资源 trade-off 规划已保存：{output}\n'
              f'估算总时间 {summary["system_time_s"]:.6g} s；'
              f'估算系统能耗 {summary["system_energy_j"]:.6g} J；'
              f'可行={summary["feasible"]}；未优化资源、未测量真实能耗')
    except (ValueError, OSError, KeyError, NotImplementedError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
