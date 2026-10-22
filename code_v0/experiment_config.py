# 用途：集中配置训练、UAV、服务器和无线环境；修改下面 CONFIG 即可改变默认参数。
# 检查/打印指令：python experiment_config.py
# 带数据清单检查：python experiment_config.py --data output/data_load_7500/data.json
# 训练示例：python centralized.py --rounds 100；显式命令行参数覆盖这里的训练默认值。
# 注意：物理参数是可修改的仿真示例，不是实测设备数据，也不会对真实 CPU/GPU 限速。
"""统一参数入口，仅使用 Python 标准库，因此检查配置不需要加载 PyTorch。

训练使用 training/latency 的参数。频率、功率、信道等物理参数供独立资源模型使用；
当前训练不会调用资源优化器，也不会因这些物理参数生成真实测量能耗。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve()
ROOT = CONFIG_PATH.parent

# ---------------------- 初学者主要修改这个字典 ----------------------
# 单位直接写进键名：Hz 是每秒周期数或带宽，W 是功率，J 是能量，m 是米。
# 每架 UAV 的样本数不在此处重填，实际负载由 data.json 的 clients 决定。
CONFIG = {
    'training': {
        'data': str(ROOT / 'output/data_load_7500/data.json'),
        'rounds': 100, 'batch_size': 32, 'eval_batch_size': 64, 'image_size': 64,
        'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005,
        'scheduler': 'cosine', 'bn': 'batch', 'seed': 42, 'aggregation_period': 1,
        'device': 'auto', 'client_device': None, 'server_device': None,
        'cpu_threads': 1, 'workers': 0,
    },
    # full 基线保留完整数据；其他训练统一使用上面的 7500 样本清单。
    'full': {'data': str(ROOT / 'output/data_full/data.json'), 'mode': 'holdout'},
    # 保留旧动态时延选择的 GFLOPS/Mbps 接口，不能把它们直接当作频率/带宽 Hz。
    'latency': {'client_gflops': [2.0], 'server_gflops': 100.0, 'bandwidth_mbps': [20.0]},
    'uavs': [
        {
            'position_m': list(position),
            'f_min_hz': 2e8, 'f_max_hz': 1.5e9, 'f_ref_hz': 1e9,
            'kappa': 1e-28, 'cycles_per_flop': 0.5,
            'p_up_min_w': 0.01, 'p_up_max_w': 1.0, 'p_up_ref_w': 0.5,
            'hover_power_w': 100.0,
        }
        for position in [(100, 0, 100), (0, 120, 100), (-140, 0, 100),
                         (0, -160, 100), (180, 0, 100)]
    ],
    # 每个 UAV 有一个独立并行服务器分支；frequency_hz 是每分支能力，不是共享总能力。
    'server': {
        'position_m': [0, 0, 0], 'parallel_branches': 5, 'frequency_hz': 3e9,
        'cycles_per_flop': 0.03, 'kappa': 1e-28, 'downlink_power_w': 2.0,
        'aggregation_time_s': 0.01, 'aggregation_power_w': 10.0,
    },
    'radio': {'total_up_hz': 2e7, 'total_down_hz': 2e7, 'min_up_hz': 1e5, 'min_down_hz': 1e5},
    'environment': {
        'seed': 42, 'carrier_hz': 2.4e9, 'noise_psd_w_hz': 3.981e-21,
        'los_a': 9.61, 'los_b': 0.16, 'los_loss_db': 1.0, 'nlos_loss_db': 20.0,
        'rician_k': 10.0,  # 线性值，不是 dB；0 表示无直达分量。
    },
    'selection': {'candidates': [f'P{i}' for i in range(9)], 'initial_split': 'P4',
                  'max_sweeps': 1, 'energy_tolerance_j': 1e-9},
    # None 表示暂不限制。UAV 能量预算可以是一个共同值或按 UAV 顺序给出列表。
    'budgets': {'round_time_s': None, 'round_uav_energy_j': None,
                'mission_time_s': None, 'mission_uav_energy_j': None},
}


def load_config():
    """每次返回独立副本；某次实验改参数不会修改模块内的默认 CONFIG。"""
    return copy.deepcopy(CONFIG)


def _number(value, name, minimum=0, allow_equal=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < minimum if allow_equal else value <= minimum)):
        relation = '>=' if allow_equal else '>'
        raise ValueError(f'{name} 必须是有限数，且 {relation} {minimum}')


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} 必须是 >= {minimum} 的整数')


def _position(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} 必须是 [x,y,z] 三个以米为单位的坐标')
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'{name} 的坐标必须是有限数')


def _validate(config):
    # 边界检查不读取图片或清单；普通训练可用 1/3/5 等任意 UAV 数量的数据。
    required = ('training', 'full', 'latency', 'uavs', 'server', 'radio',
                'environment', 'selection', 'budgets')
    if not isinstance(config, dict) or any(key not in config for key in required):
        raise ValueError(f'配置需要包含 {required}')
    for section in required:
        if section != 'uavs' and not isinstance(config[section], dict):
            raise ValueError(f'{section} 必须是字典')
    training = config['training']
    for key in ('rounds', 'batch_size', 'eval_batch_size', 'cpu_threads', 'aggregation_period'):
        _integer(training[key], f'training.{key}')
    _integer(training['image_size'], 'training.image_size', 32)
    _integer(training['workers'], 'training.workers', 0)
    _integer(training['seed'], 'training.seed', 0)
    if training['seed'] >= 2**32:
        raise ValueError('training.seed 必须小于 2**32，以兼容 NumPy 的随机种子')
    _number(training['lr'], 'training.lr')
    _number(training['momentum'], 'training.momentum', allow_equal=True)
    if training['momentum'] >= 1:
        raise ValueError('training.momentum 必须小于 1')
    _number(training['weight_decay'], 'training.weight_decay', allow_equal=True)
    if training['scheduler'] not in ('cosine', 'constant') or training['bn'] not in ('batch', 'frozen'):
        raise ValueError('scheduler 只能是 cosine/constant，bn 只能是 batch/frozen')
    for key in ('device', 'client_device', 'server_device'):
        value = training[key]
        if value is None and key != 'device':
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f'training.{key} 需要设备字符串，例如 auto/cpu/cuda:0')
    if any(not isinstance(section['data'], str) or not section['data']
           for section in (training, config['full'])):
        raise ValueError('training.data 和 full.data 需要非空文件路径')
    if config['full']['mode'] not in ('holdout', 'all'):
        raise ValueError('full.mode 只能是 holdout/all')
    uavs = config['uavs']
    if not isinstance(uavs, list) or not uavs or any(not isinstance(uav, dict) for uav in uavs):
        raise ValueError('uavs 必须是非空的 UAV 参数字典列表')
    for index, uav in enumerate(uavs):
        _position(uav['position_m'], f'uavs[{index}].position_m')
        for key in ('f_min_hz', 'f_max_hz', 'f_ref_hz', 'kappa', 'cycles_per_flop',
                    'p_up_min_w', 'p_up_max_w', 'p_up_ref_w', 'hover_power_w'):
            _number(uav[key], f'uavs[{index}].{key}')
        if not uav['f_min_hz'] <= uav['f_ref_hz'] <= uav['f_max_hz']:
            raise ValueError(f'UAV {index} 的参考频率必须位于最小/最大频率之间')
        if not uav['p_up_min_w'] <= uav['p_up_ref_w'] <= uav['p_up_max_w']:
            raise ValueError(f'UAV {index} 的参考发射功率必须位于最小/最大功率之间')
    server = config['server']
    _position(server['position_m'], 'server.position_m')
    _integer(server['parallel_branches'], 'server.parallel_branches')
    if server['parallel_branches'] < len(uavs):
        raise ValueError('当前独立并行分支模型要求服务器分支数 >= UAV 数；不能重复使用共享总算力')
    for key in ('frequency_hz', 'cycles_per_flop', 'kappa', 'downlink_power_w', 'aggregation_power_w'):
        _number(server[key], f'server.{key}')
    _number(server['aggregation_time_s'], 'server.aggregation_time_s', allow_equal=True)
    if any(math.dist(uav['position_m'], server['position_m']) == 0 for uav in uavs):
        raise ValueError('UAV 与服务器位置不能完全重合，否则路径损耗距离为零')
    radio = config['radio']
    for key in ('total_up_hz', 'total_down_hz', 'min_up_hz', 'min_down_hz'):
        _number(radio[key], f'radio.{key}')
    for direction in ('up', 'down'):
        if len(uavs)*radio[f'min_{direction}_hz'] > radio[f'total_{direction}_hz']:
            raise ValueError(f'{direction} 方向总带宽不能满足所有 UAV 的最小链路带宽')
    env = config['environment']
    _integer(env['seed'], 'environment.seed', 0)
    if env['seed'] >= 2**32:
        raise ValueError('environment.seed 必须小于 2**32')
    for key in ('carrier_hz', 'noise_psd_w_hz', 'los_a', 'los_b'):
        _number(env[key], f'environment.{key}')
    for key in ('los_loss_db', 'nlos_loss_db', 'rician_k'):
        _number(env[key], f'environment.{key}', allow_equal=True)
    latency = config['latency']
    _number(latency['server_gflops'], 'latency.server_gflops')
    for key in ('client_gflops', 'bandwidth_mbps'):
        values = latency[key]
        if not isinstance(values, list) or len(values) not in (1, len(uavs)):
            raise ValueError(f'latency.{key} 需要一个共同值或每架 UAV 一个值')
        for value in values:
            _number(value, f'latency.{key}')
    selection = config['selection']
    candidates = selection['candidates']
    if (not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates)
            or any(cut not in [f'P{i}' for i in range(9)] for cut in candidates)):
        raise ValueError('selection.candidates 需要互不重复的 P0..P8 列表')
    if selection['initial_split'] not in candidates:
        raise ValueError('selection.initial_split 必须属于候选切点')
    _integer(selection['max_sweeps'], 'selection.max_sweeps')
    _number(selection['energy_tolerance_j'], 'selection.energy_tolerance_j', allow_equal=True)
    for key in ('round_time_s', 'round_uav_energy_j', 'mission_time_s', 'mission_uav_energy_j'):
        value = config['budgets'][key]
        if value is None:
            continue
        if key.endswith('uav_energy_j') and isinstance(value, list):
            if len(value) != len(uavs):
                raise ValueError(f'budgets.{key} 列表长度必须等于 UAV 数')
            for item in value:
                _number(item, f'budgets.{key}')
        else:
            _number(value, f'budgets.{key}')


def validate_config(config):
    """无返回值；不合法配置立即报错，避免把单位错误或不可行上限带入物理模型。"""
    try:
        _validate(config)
    except (KeyError, TypeError) as error:
        raise ValueError(f'配置字段缺失或类型错误：{error}') from error


def validate_manifest_uavs(config, manifest):
    """独立物理实验检查 UAV 数量和完整负载；普通训练不受物理配置中 UAV 数限制。"""
    validate_config(config)
    if not isinstance(manifest, dict):
        raise ValueError('数据清单必须是 JSON 对象')
    clients = manifest.get('clients')
    if not isinstance(clients, list) or len(clients) != len(config['uavs']):
        raise ValueError(f'物理配置含 {len(config["uavs"])} 架 UAV，需使用相同 UAV 数的数据清单')
    settings = manifest.get('settings', {})
    if not isinstance(settings, dict) or settings.get('uavs', len(clients)) != len(clients):
        raise ValueError('数据清单 settings.uavs 与 clients 数量不一致')
    indices = []
    for index, client in enumerate(clients):
        if not isinstance(client, dict):
            raise ValueError(f'数据清单 UAV {index} 需要参数字典')
        values = client.get('indices')
        if not isinstance(values, list) or not values:
            raise ValueError(f'数据清单 UAV {index} 的训练索引不能为空')
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError('UAV 训练索引必须是非负整数')
        if 'samples' in client and client['samples'] != len(values):
            raise ValueError(f'UAV {index} 的 samples 与 indices 不一致')
        indices.extend(values)
    if len(indices) != len(set(indices)):
        raise ValueError('不同 UAV 不能持有重复的训练索引')
    if 'samples' in manifest:
        if (not isinstance(manifest['samples'], list)
                or any(not isinstance(row, dict) or 'split' not in row for row in manifest['samples'])):
            raise ValueError('数据清单 samples 必须是含 split 字段的样本字典列表')
        train_count = sum(row['split'] == 'train' for row in manifest['samples'])
        if sorted(indices) != list(range(train_count)):
            raise ValueError('UAV 索引必须恰好覆盖清单中全部训练样本')


def training_defaults(method='centralized'):
    """提供 argparse 默认值；argparse 自动让显式 CLI 值优先，无需重写训练循环。"""
    config = load_config()
    validate_config(config)
    defaults = copy.deepcopy(config['training'])
    defaults.update(config['latency'])
    defaults['candidates'] = config['selection']['candidates'][:]
    defaults['split'] = [config['selection']['initial_split']]
    if method == 'resnet18_full':
        defaults.update(config['full'])
        defaults['epochs'] = defaults['rounds']
    return defaults


def configuration_snapshot(args, method):
    """同时保存可复查的配置源代码和 CLI 覆盖后的值；物理参数未接入真实训练。"""
    config = load_config()
    for key in config['training']:
        if hasattr(args, key):
            config['training'][key] = getattr(args, key)
    if method == 'resnet18_full':
        config['training']['rounds'] = args.epochs
        config['full'].update(data=args.data, mode=args.mode)
    for key in config['latency']:
        if hasattr(args, key):
            config['latency'][key] = copy.deepcopy(getattr(args, key))
    if hasattr(args, 'candidates'):
        config['selection']['candidates'] = args.candidates[:]
    return dict(source_path=str(CONFIG_PATH), source_code=CONFIG_PATH.read_text(encoding='utf-8'),
                resolved=config, physical_parameters_applied_to_training=False,
                resource_optimization_applied=False, energy_measured=False,
                scope='training/latency defaults applied; physical resource model is evaluated separately')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='可选：检查现有 data.json 与物理配置的 UAV 数和负载；默认只检查配置')
    args = parser.parse_args(argv)
    try:
        config = load_config()
        validate_config(config)
        if args.data:
            path = Path(args.data).expanduser().resolve()
            if path.is_dir():
                path /= 'data.json'
            manifest = json.loads(path.read_text(encoding='utf-8-sig'))
            validate_manifest_uavs(config, manifest)
            print(f'清单检查通过：{path}，UAV 样本数={[len(c["indices"]) for c in manifest["clients"]]}')
        print('配置检查通过。以下物理数值为仿真示例，尚未用于真实训练资源分配或实测能耗。')
        print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))
        return config
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
