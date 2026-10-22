"""用途：按动态 UAV-SFL 物理模型建立资源问题、评价参考资源与检查约束。
运行：python resource_allocation.py --data output/data_load_7500/data.json --rounds 2 --split P4
运行：python resource_allocation.py --mode optimize  # 求解器尚未接入，明确报错

资源单位统一为 Hz、W、bits、cycles、s、J。一个 SL 轮遍历各 UAV 的全部样本。
本文件只计算仿真估计；均分带宽是参考资源，并不是优化结果或真实能耗测量。
频率 f 决定计算功率 kappa*f**3，因此不再把计算功率作为独立优化变量。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from experiment_config import validate_config

ROOT = Path(__file__).resolve().parent


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} 必须为有限数')
    if (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} 必须为{"非负" if zero else "正"}数')
    return float(value)


def _vector(values, count, name, *, zero=False):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f'{name} 需要恰好 {count} 个数')
    return [_positive(value, f'{name}[{i}]', zero=zero) for i, value in enumerate(values)]


def _caps(value, count, name):
    if value is None:
        return None
    return _vector([value] * count if isinstance(value, (int, float)) else value, count, name)


def _bound(rows, name, value, bound):
    # 所有不等式都写成 value <= bound；residual > 0 表示违反约束。
    residual = value - bound
    rows.append(dict(name=name, value=value, bound=bound, residual=residual,
                     satisfied=residual <= 1e-9 * max(1.0, abs(bound))))


def _constraint_result(rows):
    return dict(feasible=all(row['satisfied'] for row in rows), checks=rows,
                violations=[row for row in rows if not row['satisfied']])


def reference_resources(config):
    """固定参考频率和功率、均分带宽；不做任何优化或隐式归一化。"""
    validate_config(config)
    count = len(config['uavs'])
    return dict(f_hz=[u['f_ref_hz'] for u in config['uavs']],
                p_up_w=[u['p_up_ref_w'] for u in config['uavs']],
                bandwidth_up_hz=[config['radio']['total_up_hz'] / count] * count,
                bandwidth_down_hz=[config['radio']['total_down_hz'] / count] * count)


def observe_channel(config, round_index):
    """固定巡逻点的 LoS 状态只抽一次；每个 epoch 的小尺度衰落独立更新。

    同一 seed 和 round_index 可直接复现当前观测，不需要生成或查看未来信道。
    使用三维距离的自由空间损耗、LoS/NLoS 附加损耗和莱斯/瑞利功率增益。
    上下行共用当前信道增益（互易信道假设），噪声谱密度的单位是 W/Hz。
    """
    validate_config(config)
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError('round_index 必须是从 0 开始的非负整数')
    env, server = config['environment'], config['server']
    los_rng = random.Random(env['seed'] + 710003)
    fading_rng = random.Random(env['seed'] + 910009 + round_index * 1000003)
    gains, states, distances, probabilities, losses = [], [], [], [], []
    for uav in config['uavs']:
        displacement = [a - b for a, b in zip(uav['position_m'], server['position_m'])]
        distance = math.sqrt(sum(x * x for x in displacement))
        _positive(distance, 'UAV 与服务器距离')
        horizontal = math.hypot(displacement[0], displacement[1])
        elevation = math.degrees(math.atan2(abs(displacement[2]), horizontal))
        probability = 1 / (1 + env['los_a'] * math.exp(-env['los_b'] * (elevation - env['los_a'])))
        los = los_rng.random() < probability
        loss_db = 20 * math.log10(4 * math.pi * distance * env['carrier_hz'] / 299792458.0)
        loss_db += env['los_loss_db'] if los else env['nlos_loss_db']
        k = env['rician_k'] if los else 0.0  # NLoS 使用瑞利衰落，即莱斯因子为 0。
        sigma = math.sqrt(1 / (2 * (k + 1)))
        real = math.sqrt(k / (k + 1)) + fading_rng.gauss(0, sigma)
        imaginary = fading_rng.gauss(0, sigma)
        gain = 10 ** (-loss_db / 10) * (real * real + imaginary * imaginary)
        gains.append(_positive(gain, '信道增益'))
        states.append(los)
        distances.append(distance)
        probabilities.append(probability)
        losses.append(loss_db)
    return dict(round_index=round_index, gain=gains, los=states, distance_m=distances,
                los_probability=probabilities, large_scale_loss_db=losses,
                protocol='fixed point LoS; epoch fading; reciprocal uplink/downlink; current observation only')


def read_manifest_workload(path, config):
    """校验原生 CBCL 清单，只读样本和客户端映射，不加载裁剪图片。"""
    from data import validate_manifest
    from experiment_config import validate_manifest_uavs
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= 'data.json'
    raw = path.read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    validate_manifest(manifest)
    validate_manifest_uavs(config, manifest)
    rows, clients = manifest['samples'], manifest['clients']
    counts = {name: sum(row['split'] == name for row in rows) for name in ('train', 'val', 'test')}
    # 客户端索引针对筛选后的 train 子集；资源评价只读取其数量。
    if manifest.get('subset_counts') != counts:
        raise ValueError('清单 subset_counts 与实际样本数不一致')
    sizes = [len(client['indices']) for client in clients]
    if any(client.get('samples') != size for client, size in zip(clients, sizes)):
        raise ValueError('清单客户端 samples 与 indices 数量不一致')
    return dict(data=str(path), sizes=sizes, classes=manifest['classes'], subset_counts=counts,
                manifest_sha256=hashlib.sha256(raw).hexdigest(), fingerprint=manifest['fingerprint'],
                round_definition='1 complete local epoch; every assigned sample once; no padding')


def build_round_problem(config, profiles, sample_counts, channel_gain, cuts, *, next_cuts=None,
                        aggregation_due=False, final_round=False):
    """将一轮完整 epoch 工作量转换成资源分配问题，不依赖 torch。

    首周期模型已预加载，因此此接口没有初始下发。聚合轮上传当前切点前缀；
    下发下一周期切点前缀，最后一轮不下发。BN 等 state_dict 状态已计入画像。
    """
    validate_config(config)
    count = len(config['uavs'])
    if count > config['server']['parallel_branches']:
        raise ValueError('每架 UAV 需要一个独立服务器分支，不能重复使用共享算力容量')
    if not isinstance(sample_counts, (list, tuple)) or len(sample_counts) != count:
        raise ValueError('样本数必须与 UAV 数量相同')
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in sample_counts):
        raise ValueError('每架 UAV 的完整 epoch 样本数必须为正整数')
    gain = _vector(channel_gain, count, 'channel_gain')
    if len(cuts) != count or any(cut not in profiles for cut in cuts):
        raise ValueError('每架 UAV 的切点必须有对应画像')
    next_cuts = list(cuts) if next_cuts is None else list(next_cuts)
    if len(next_cuts) != count or any(cut not in profiles for cut in next_cuts):
        raise ValueError('下一周期切点必须有对应画像')
    client_cycles, server_cycles = [], []
    bits = {name: [] for name in ('activation_up', 'gradient_down', 'model_up', 'model_down')}
    for n, size in enumerate(sample_counts):
        profile = profiles[cuts[n]]
        for key in ('client_flops_per_sample', 'server_flops_per_sample', 'activation_bytes_per_sample',
                    'gradient_bytes_per_sample', 'label_bytes_per_sample', 'prefix_state_bytes'):
            _positive(profile[key], f'profiles[{cuts[n]}].{key}', zero=True)
        _positive(profiles[next_cuts[n]]['prefix_state_bytes'], '下一前缀字节数', zero=True)
        client_cycles.append(size * profile['client_flops_per_sample'] * config['uavs'][n]['cycles_per_flop'])
        server_cycles.append(size * profile['server_flops_per_sample'] * config['server']['cycles_per_flop'])
        bits['activation_up'].append(size * 8 * (profile['activation_bytes_per_sample'] + profile['label_bytes_per_sample']))
        bits['gradient_down'].append(size * 8 * profile['gradient_bytes_per_sample'])
        bits['model_up'].append(8 * profile['prefix_state_bytes'] if aggregation_due else 0.0)
        bits['model_down'].append(8 * profiles[next_cuts[n]]['prefix_state_bytes']
                                  if aggregation_due and not final_round else 0.0)
    for name, values in dict(client_cycles=client_cycles, server_cycles=server_cycles, **bits).items():
        _vector(values, count, name, zero=True)  # 有限输入的乘法也可能溢出，必须再次检查。
    return dict(config=copy.deepcopy(config), cuts=list(cuts), next_cuts=next_cuts,
                sample_counts=list(sample_counts), channel_gain=gain, client_cycles=client_cycles,
                server_cycles=server_cycles, payload_bits=bits, aggregation_due=bool(aggregation_due),
                final_round=bool(final_round), optimized=False,
                auxiliary_variables=['sl_stage_time_s', 'model_up_stage_time_s', 'model_down_stage_time_s'],
                objective='system total energy J: UAV compute/tx/hover + server compute/tx/aggregation',
                initial_distribution='preloaded, excluded',
                profile_limitations='covered forward/backward FLOPs only; excludes optimizer, uncounted BN/operators, migration and queueing')


def _hardware_constraints(problem, resources):
    """物理计算前先检查变量域；硬件越界返回 infeasible，不自动修正。"""
    config, rows = problem['config'], []
    count = len(config['uavs'])
    vectors = {name: _vector(resources.get(name), count, name) for name in
               ('f_hz', 'p_up_w', 'bandwidth_up_hz', 'bandwidth_down_hz')}
    for n, uav in enumerate(config['uavs']):
        _bound(rows, f'f_min[{n}]', uav['f_min_hz'], vectors['f_hz'][n])
        _bound(rows, f'f_max[{n}]', vectors['f_hz'][n], uav['f_max_hz'])
        _bound(rows, f'p_up_min[{n}]', uav['p_up_min_w'], vectors['p_up_w'][n])
        _bound(rows, f'p_up_max[{n}]', vectors['p_up_w'][n], uav['p_up_max_w'])
        _bound(rows, f'bandwidth_up_min[{n}]', config['radio']['min_up_hz'], vectors['bandwidth_up_hz'][n])
        _bound(rows, f'bandwidth_down_min[{n}]', config['radio']['min_down_hz'], vectors['bandwidth_down_hz'][n])
    _bound(rows, 'bandwidth_up_total', sum(vectors['bandwidth_up_hz']), config['radio']['total_up_hz'])
    _bound(rows, 'bandwidth_down_total', sum(vectors['bandwidth_down_hz']), config['radio']['total_down_hz'])
    return _constraint_result(rows)


def validate_resources(problem, resources):
    """返回硬件和单轮预算的全部约束；与 evaluate_resources 的可行性完全一致。"""
    return evaluate_resources(problem, resources)['constraints']


def _rate(bandwidth, power, gain, noise):
    # log1p 避免极低 SNR 时 1+snr 被舍入为 1，产生虚假的零速率。
    received_power = _positive(power * gain, '接收信号功率')
    noise_power = _positive(noise * bandwidth, '带宽内噪声功率')
    snr = _positive(received_power / noise_power, 'SNR')
    rate = bandwidth * math.log1p(snr) / math.log(2)
    return _positive(rate, 'Shannon rate')


def _budget_checks(config, time_s, energy_j, *, mission=False, prefix=''):
    rows, budgets = [], config['budgets']
    time_key = 'mission_time_s' if mission else 'round_time_s'
    energy_key = 'mission_uav_energy_j' if mission else 'round_uav_energy_j'
    if budgets[time_key] is not None:
        _bound(rows, prefix + time_key, time_s, budgets[time_key])
    caps = _caps(budgets[energy_key], len(energy_j), energy_key)
    if caps is not None:
        for n, (energy, cap) in enumerate(zip(energy_j, caps)):
            _bound(rows, prefix + f'{energy_key}[{n}]', energy, cap)
    return rows


def evaluate_resources(problem, resources):
    """评价一个完整 SL epoch；每个同步阶段分别取最大分支时间。

    不取 max(各分支全部阶段之和)：SL、上传、下发有独立同步屏障。
    所有 UAV 在系统执行的全部时间悬停，包括等待其他 UAV 的时间。
    """
    config, count = problem['config'], len(problem['sample_counts'])
    constraints = _hardware_constraints(problem, resources)['checks']
    server, noise, bits = config['server'], config['environment']['noise_psd_w_hz'], problem['payload_bits']
    times = {name: [] for name in ('client_compute', 'activation_up', 'server_compute',
                                   'gradient_down', 'sl_branch', 'model_up', 'model_down')}
    energies = {name: [] for name in ('uav_compute', 'uav_tx', 'uav_hover', 'server_compute', 'server_tx')}
    rates = {'up': [], 'down': []}
    for n, uav in enumerate(config['uavs']):
        frequency, power = resources['f_hz'][n], resources['p_up_w'][n]
        gain = problem['channel_gain'][n]
        up = _rate(resources['bandwidth_up_hz'][n], power, gain, noise)
        down = _rate(resources['bandwidth_down_hz'][n], server['downlink_power_w'], gain, noise)
        rates['up'].append(up)
        rates['down'].append(down)
        times['client_compute'].append(problem['client_cycles'][n] / frequency)
        times['server_compute'].append(problem['server_cycles'][n] / server['frequency_hz'])
        for name, rate in (('activation_up', up), ('gradient_down', down), ('model_up', up), ('model_down', down)):
            times[name].append(bits[name][n] / rate)
        times['sl_branch'].append(sum(times[name][n] for name in
                                      ('client_compute', 'activation_up', 'server_compute', 'gradient_down')))
        energies['uav_compute'].append(uav['kappa'] * frequency * frequency * problem['client_cycles'][n])
        energies['uav_tx'].append(power * (times['activation_up'][n] + times['model_up'][n]))
        energies['server_compute'].append(server['kappa'] * server['frequency_hz'] * server['frequency_hz'] * problem['server_cycles'][n])
        energies['server_tx'].append(server['downlink_power_w'] * (times['gradient_down'][n] + times['model_down'][n]))
    aggregation = server['aggregation_time_s'] if problem['aggregation_due'] else 0.0
    stages = dict(sl=max(times['sl_branch']), model_up=max(times['model_up']),
                  model_down=max(times['model_down']), aggregation=aggregation)
    system_time = sum(stages.values())
    energies['uav_hover'] = [uav['hover_power_w'] * system_time for uav in config['uavs']]
    uav_energy = [sum(energies[name][n] for name in ('uav_compute', 'uav_tx', 'uav_hover')) for n in range(count)]
    server_aggregation = server['aggregation_power_w'] * aggregation
    server_energy = sum(energies['server_compute']) + sum(energies['server_tx']) + server_aggregation
    system_energy = sum(uav_energy) + server_energy
    _positive(system_time, '系统时延', zero=True)
    _positive(system_energy, '系统能耗', zero=True)
    _vector(uav_energy, count, 'UAV 能耗', zero=True)
    constraints += _budget_checks(config, system_time, uav_energy)
    result = _constraint_result(constraints)
    return dict(system_time_s=system_time, system_energy_j=system_energy, uav_energy_j=uav_energy,
                feasible=result['feasible'], optimized=False, energy_measured=False,
                resources=copy.deepcopy(resources), times_s=dict(stages=stages, branches=times),
                rates_bits_s=rates, energies_j=dict(components=energies, server_aggregation=server_aggregation,
                                                  server_total=server_energy, uav_total=uav_energy, system_total=system_energy),
                payload_bits=copy.deepcopy(bits), constraints=result,
                uav_compute_power_w=[u['kappa'] * f * f * f for u, f in zip(config['uavs'], resources['f_hz'])])


def evaluate_cycle_reference(config, profiles, sizes, gains, cuts, resources, period, preloaded=False):
    """对拟执行周期评分：候选前缀开头下发、周期末上传，中间执行 period 个 epoch。

    固定当前参考资源及信道，仅用于比较切点；不是未来随机信道的预测。
    上一周期末当前前缀的上传是固定历史成本，本函数不重复计入。
    """
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError('period 必须为正整数')
    base_problem = build_round_problem(config, profiles, sizes, gains, cuts)
    base = evaluate_resources(base_problem, resources)
    last_problem = build_round_problem(config, profiles, sizes, gains, cuts, aggregation_due=True, final_round=True)
    last = evaluate_resources(last_problem, resources)
    down_times = [0.0 if preloaded else profiles[cut]['prefix_state_bytes'] * 8 / rate
                  for cut, rate in zip(cuts, base['rates_bits_s']['down'])]
    start_down = max(down_times)
    hover = [u['hover_power_w'] for u in config['uavs']]
    start_uav = [power * start_down for power in hover]
    start_server = config['server']['downlink_power_w'] * sum(down_times)
    cycle_time = period * base['system_time_s'] + last['system_time_s'] - base['system_time_s'] + start_down
    uav_energy = [period * b + final - b + initial
                  for b, final, initial in zip(base['uav_energy_j'], last['uav_energy_j'], start_uav)]
    cycle_energy = period * base['system_energy_j'] + last['system_energy_j'] - base['system_energy_j']
    cycle_energy += sum(start_uav) + start_server
    _positive(cycle_time, '周期时延', zero=True)
    _positive(cycle_energy, '周期能耗', zero=True)
    _vector(uav_energy, len(sizes), '周期 UAV 能耗', zero=True)
    # 候选下发用于周期目标，但实际发生在上一聚合轮末，由切点模块的
    # transition_cuts 检查该轮预算；不能再占用下一周期首轮的时限/电量。
    # 未来周期末的下一次下发尚未决定，留到下一次选点时再检查。
    checks = _hardware_constraints(base_problem, resources)['checks']
    checks += _budget_checks(config, last['system_time_s'], last['uav_energy_j'], prefix='cycle_last.')
    if period > 1:
        checks += _budget_checks(config, base['system_time_s'], base['uav_energy_j'], prefix='cycle_sl.')
    checks += _budget_checks(config, cycle_time, uav_energy, mission=True, prefix='cycle_only.')
    constraints = _constraint_result(checks)
    return dict(cycle_time_s=cycle_time, cycle_energy_j=cycle_energy, system_time_s=cycle_time,
                system_energy_j=cycle_energy, uav_energy_j=uav_energy, feasible=constraints['feasible'],
                constraints=constraints, optimized=False, energy_measured=False, period=period,
                preloaded=bool(preloaded), start_down_time_s=start_down,
                start_down_payload_bits=[0.0 if preloaded else profiles[c]['prefix_state_bytes'] * 8 for c in cuts],
                cycle_end_upload_payload_bits=last['payload_bits']['model_up'],
                scoring_protocol='current channel held for hypothetical cycle; prior upload excluded; start download charged to preceding transition round; mission history checked separately')


def allocate_resources(problem, warm_start=None):
    """论文求解算法预留接口：未来返回满足约束的资源，不使用均分作为优化回退。"""
    raise NotImplementedError('资源优化求解器尚未实现；请选择 reference 模式评价固定参考资源，不能将均分带宽称为优化结果')


def planning_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data', default=None, help='已划分的 CBCL data.json')
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--aggregation-period', type=int, default=None)
    parser.add_argument('--split', default=None, help='参考固定切点，例如 P4；默认读取配置')
    parser.add_argument('--output', default=None, help='新的或空的输出目录')
    parser.add_argument('--mode', choices=('reference', 'optimize'), default='reference')
    return parser


def prepare_planning(args):
    """独立入口才加载 torch 和现有画像函数，避免配置/物理评价产生重依赖。"""
    if args.mode == 'optimize':
        allocate_resources(None)  # 必须在创建输出或画像前报错。
    from experiment_config import load_config, validate_config
    config = load_config()
    for name in ('data', 'rounds', 'aggregation_period'):
        value = getattr(args, name, None)
        if value is not None:
            config['training'][name] = value
    if args.split is not None:
        config['selection']['initial_split'] = args.split.upper()
    validate_config(config)
    workload = read_manifest_workload(config['training']['data'], config)
    from models import SplitResNet18
    from training_common import profile_splits, seed_all
    import torch
    seed_all(config['training']['seed'])
    torch.set_num_threads(config['training']['cpu_threads'])
    profile_args = SimpleNamespace(**config['training'])
    base = SplitResNet18(num_classes=len(workload['classes']), bn_mode=profile_args.bn)
    profiles = profile_splits(base, profile_args, config['selection']['candidates'])
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (Path(args.output) if args.output else ROOT / 'output' / 'resource_reference' / stamp).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'输出目录必须不存在或为空：{output}')
    output.mkdir(parents=True, exist_ok=True)
    return config, workload, profiles, output


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_reference_plan(output, config, workload, profiles, history, schedule, *, label):
    """保存独立物理评价记录；累计任务预算失败也保留完整结果，不能静默截断。"""
    time_s = sum(row['system_time_s'] for row in history)
    energy_j = sum(row['system_energy_j'] for row in history)
    uav_energy = [sum(row['uav_energy_j'][n] for row in history) for n in range(len(config['uavs']))]
    _positive(time_s, '累计时延', zero=True)
    _positive(energy_j, '累计能耗', zero=True)
    _vector(uav_energy, len(config['uavs']), '累计 UAV 能耗', zero=True)
    mission = _constraint_result(_budget_checks(config, time_s, uav_energy, mission=True))
    summary = dict(label=label, rounds=len(history), client_sizes=workload['sizes'],
                   manifest_sha256=workload['manifest_sha256'], system_time_s=time_s, system_energy_j=energy_j,
                   uav_energy_j=uav_energy, feasible=all(row['feasible'] for row in history) and mission['feasible'],
                   mission_constraints=mission, optimized=False, energy_measured=False, actual_training=False,
                   resource_policy='fixed reference frequency/power and equal bandwidth; not optimized',
                   initial_distribution='first prefix preloaded; final round has no next download',
                   physics='parallel server branch per UAV; independent stage barriers; all UAV hover during system time',
                   profile_limitations='covered forward/backward FLOPs only; excludes optimizer, uncounted operators, migration and queueing')
    values = {'config.json': config, 'data_manifest.json': workload, 'split_profile.json': profiles,
              'resource_history.json': history, 'split_schedule.json': schedule, 'summary.json': summary}
    for filename, value in values.items():
        _save_json(Path(output) / filename, value)
    return summary


def main():
    parser = planning_parser('固定参考资源下的完整 epoch 物理评价（不执行训练或资源优化）')
    args = parser.parse_args()
    try:
        config, workload, profiles, output = prepare_planning(args)
        resources = reference_resources(config)
        cuts = [config['selection']['initial_split']] * len(workload['sizes'])
        history, schedule = [], []
        rounds, period = config['training']['rounds'], config['training']['aggregation_period']
        for index in range(rounds):
            channel = observe_channel(config, index)
            aggregation = (index + 1) % period == 0 or index == rounds - 1
            problem = build_round_problem(config, profiles, workload['sizes'], channel['gain'], cuts,
                                          next_cuts=cuts, aggregation_due=aggregation, final_round=index == rounds - 1)
            row = dict(round=index + 1, cuts=cuts[:], next_cuts=cuts[:] if aggregation and index != rounds - 1 else None,
                       channel=channel, **evaluate_resources(problem, resources))
            history.append(row)
            schedule.append(dict(round=index + 1, cuts=cuts[:], algorithm='fixed_reference', aggregation_due=aggregation))
        summary = save_reference_plan(output, config, workload, profiles, history, schedule, label='fixed_reference_evaluation')
        print(f'参考资源评价：{output}\n估算总时间 {summary["system_time_s"]:.6g} s；'
              f'估算系统能耗 {summary["system_energy_j"]:.6g} J；可行={summary["feasible"]}；未优化、未测量真实能耗')
    except (ValueError, OSError, KeyError, NotImplementedError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
