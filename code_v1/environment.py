"""用途：生成并读取多无人机共用的物理时间航迹与信道环境，仅依赖标准库。

运行与使用方式（在 code_v1 目录执行）：
    python -X utf8 -B prepare_scenario.py --stage environment
    python -X utf8 -B example_read.py

也可在 Python 中读取已生成的场景：
    from environment import EnvironmentTrace
    trace = EnvironmentTrace('output/scenario/environment.csv')
    state = trace.state_at(12.5, 0)
    gains = trace.window_mean(12.5, window_size=20)

本模块没有命令行主入口；生成工作由 prepare_scenario.py 调用
generate_environment(config, manifest_path, output_dir) 完成。输入为统一配置
config 和已准备的数据清单 data.json；输出为 environment.csv、scenario.json
及 patrol_map.svg，分别保存环境时序、参数与数据来源、航迹示意图。

无人机沿预定闭合折线恒速飞行，不设置悬停。环境生成一次后，各比较方案
按自己的实际执行时刻读取同一份 CSV。位置按相邻坐标线性插值；信道只取
不晚于查询时刻的最近观测，不利用未来 CSI。读取接口不会推进内部时钟、
重新采样信道或修改样本到达顺序，因此重复或乱序查询得到相同结果。
"""
from __future__ import annotations

import bisect
import copy
import csv
import hashlib
import html
import io
import json
import math
import random
from pathlib import Path

from config import available_samples_at, resolve_collection, validate_config


# 长表中一行对应一个物理时刻的一架 UAV，按 (time_s, uav_id) 排列。
# 时间单位为 s，坐标与距离为 m；path_loss_db 为不含小尺度衰落的 dB 损耗，
# channel_gain 为包含衰落的线性功率增益，los 用 0/1 表示传播状态。
CSV_FIELDS = (
    'time_s', 'uav_id', 'x_m', 'y_m', 'z_m', 'distance_m',
    'los_probability', 'los', 'path_loss_db', 'channel_gain',
)


def _route(mission, uav_id):
    """为编号从 0 开始的一架 UAV 构造分区内蛇形闭合航迹。

    将矩形区域沿 x 方向等宽分为 N 个条带，在本条带内设置沿 y 方向的
    平行扫线。扫线位于各小条带中心，实际间距不超过 lane_spacing_m；
    相邻扫线交替向上、向下飞行，并在端点以水平线段连接。

    返回字典中的坐标、累计路程、单圈长度单位为 m；航点到达时间、单圈
    时间单位为 s，速度单位为 m/s。最后一个航点与起点重合，便于循环。
    完成圈数仅统计整圈，mission_distance_m 则包含任务末尾的未完成部分。
    此处采用折线运动学模型，不单独模拟转弯加速度或转弯驻留时间。
    """
    count = mission['num_uavs']
    width = (mission['x_max_m'] - mission['x_min_m']) / count
    left = mission['x_min_m'] + uav_id * width
    right = left + width
    bottom, top = mission['y_min_m'], mission['y_max_m']
    altitude = mission['altitude_m']
    lanes = max(1, math.ceil(width / mission['lane_spacing_m']))
    points = []
    for lane in range(lanes):
        x = left + (lane + 0.5) * width / lanes
        ys = (bottom, top) if lane % 2 == 0 else (top, bottom)
        for y in ys:
            point = [x, y, altitude]
            if not points or point != points[-1]:
                points.append(point)
    # 偶数条扫线结束于起点所在的下边界，可水平返回；奇数条扫线先沿上边界
    # 返回第一条中心线，再回到起点。只有一条扫线时采用连续往返路径。
    if points[-1][1] != points[0][1] and points[-1][0] != points[0][0]:
        points.append([points[0][0], points[-1][1], altitude])
    if points[-1] != points[0]:
        points.append(points[0][:])
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    loop_time = cumulative[-1] / mission['speed_m_s']
    return dict(uav_id=uav_id, x_min_m=left, x_max_m=right,
                y_min_m=bottom, y_max_m=top, lane_count=lanes,
                waypoints_m=points, cumulative_distance_m=cumulative,
                waypoint_times_s=[distance / mission['speed_m_s'] for distance in cumulative],
                loop_length_m=cumulative[-1], loop_time_s=loop_time,
                completed_loops=math.floor(mission['duration_s'] / loop_time),
                mission_distance_m=mission['speed_m_s'] * mission['duration_s'],
                speed_m_s=mission['speed_m_s'])


def _position(route, time_s):
    """按恒速路程参数化计算任意任务时刻的三维位置，返回单位为 m 的列表。

    先计算单圈内路程 s=(v*t) mod L，再用累计路程定位所在直线段，按段内
    距离比例插值。闭合后从起点继续飞行；到达航点时直接进入下一段，不
    插入等待。time_s 单位为 s，本内部函数由已验证的生成时间网格调用。
    """
    distance = (time_s * route['speed_m_s']) % route['loop_length_m']
    cumulative, points = route['cumulative_distance_m'], route['waypoints_m']
    index = min(bisect.bisect_right(cumulative, distance) - 1, len(points) - 2)
    fraction = (distance - cumulative[index]) / (cumulative[index + 1] - cumulative[index])
    return [a + fraction * (b - a) for a, b in zip(points[index], points[index + 1])]


def _time_grid(mission, routes):
    """合并规则采样时刻、各 UAV 每圈转角时刻及任务起止时刻，单位为 s。

    规则采样间隔为 trace_interval_s。额外保留全部转角可避免读取时的
    线性插值跨越折线拐点、将航迹错误连成斜线。各 UAV 使用合并后的同一
    时间网格，故相邻观测间隔不一定完全相等，观测数也不等于训练轮数。
    """
    duration, interval = mission['duration_s'], mission['trace_interval_s']
    times = [i * interval for i in range(math.floor(duration / interval) + 1)]
    times.extend([0.0, duration])
    for route in routes:
        period = route['loop_length_m'] / route['speed_m_s']
        for cycle in range(math.floor(duration / period) + 1):
            for distance in route['cumulative_distance_m']:
                value = cycle * period + distance / route['speed_m_s']
                if 0 <= value <= duration:
                    times.append(value)
    times.sort()
    # 仅合并浮点计算产生的近重合时刻，容差远小于常规采样间隔；保留转角。
    tolerance = min(interval * 1e-7, 1e-9)
    merged = []
    for value in times:
        if not merged or value - merged[-1] > tolerance:
            merged.append(value)
    if duration - merged[-1] <= tolerance:
        merged[-1] = duration
    else:
        merged.append(duration)
    return merged


def _channel(position, server_position, parameters, rng):
    """按当前位置生成一次空地链路功率增益及其诊断量。

    两个位置均为单位 m 的三维坐标。由三维距离 d 和以度计的仰角 theta
    得到 LoS 概率 1/[1+a*exp(-b*(theta-a))]，再抽取本次 LoS/NLoS 状态。
    大尺度损耗为 20*log10(4*pi*d*fc/c0) 加相应状态的额外损耗，单位 dB；
    fc 单位 Hz，c0 为真空光速。配置中的噪声谱密度留给后续速率计算使用。

    小尺度功率因子为 |sqrt(K)+z|^2/(1+K)，z 为单位方差复高斯变量。
    LoS 使用配置中的线性莱斯因子 K；NLoS 令 K=0，得到瑞利衰落。
    最终 channel_gain=10^(-loss/10)*衰落功率因子，是无量纲线性功率增益。

    rng 由生成器用固定 seed 初始化，并按物理时间及 UAV 编号依次使用；
    当前简化模型在各观测时刻独立抽取状态与衰落，不额外模拟时间相关性。
    本函数只在生成阶段调用，查询已有 CSV 时不会调用。
    """
    difference = [a - b for a, b in zip(position, server_position)]
    distance = math.sqrt(sum(value * value for value in difference))
    if distance <= 0:
        raise ValueError('A UAV trajectory must not intersect the server position')
    elevation = math.degrees(math.atan2(abs(difference[2]), math.hypot(*difference[:2])))
    exponent = -parameters['los_b'] * (elevation - parameters['los_a'])
    # 对极端指数直接采用趋近于零的概率，避免 exp 溢出。
    if exponent > 700:
        probability = 0.0
    else:
        probability = 1 / (1 + parameters['los_a'] * math.exp(exponent))
    los = rng.random() < probability
    loss = 20 * math.log10(4 * math.pi * distance * parameters['carrier_hz'] / 299792458.0)
    loss += parameters['los_loss_db'] if los else parameters['nlos_loss_db']
    k = parameters['rician_k'] if los else 0.0
    sigma = math.sqrt(1 / (2 * (k + 1)))
    real = math.sqrt(k / (k + 1)) + rng.gauss(0, sigma)
    imaginary = rng.gauss(0, sigma)
    gain = 10 ** (-loss / 10) * (real * real + imaginary * imaginary)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('Channel parameters produced a nonpositive or nonfinite gain')
    return dict(distance_m=distance, los_probability=probability, los=int(los),
                path_loss_db=loss, channel_gain=gain)


def _patrol_svg(config, routes):
    """生成俯视航迹图的 SVG 文本，不写文件，也不参与物理量计算。

    按相同比例将以 m 为单位的世界坐标映射到画布像素；区域边界、UAV
    分区、完整闭合路径、起点、首段方向箭头和服务器位置同时显示。
    高度、速度、任务时长写入图注，图中的折线表示一圈几何路径。
    """
    mission, server = config['mission'], config['server']['position_m']
    width, height, padding = 960, 760, 85
    xmin, xmax = min(mission['x_min_m'], server[0]), max(mission['x_max_m'], server[0])
    ymin, ymax = min(mission['y_min_m'], server[1]), max(mission['y_max_m'], server[1])
    scale = min((width - 2 * padding) / (xmax - xmin), (height - 2 * padding) / (ymax - ymin))
    xoffset = (width - (xmax - xmin) * scale) / 2
    yoffset = (height - (ymax - ymin) * scale) / 2

    def project(x, y):
        """将以 m 为单位的平面坐标映射为 SVG 像素坐标，并翻转屏幕 y 轴。"""
        return xoffset + (x - xmin) * scale, height - yoffset - (y - ymin) * scale

    colours = ('#2166ac', '#d95f02', '#1b9e77', '#984ea3', '#b2182b', '#636363')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="14" fill="#222">',
             '<text x="32" y="30" font-size="20">Fixed continuous UAV patrol routes</text>',
             f'<text x="32" y="53">Altitude {mission["altitude_m"]:g} m; speed {mission["speed_m_s"]:g} m/s; duration {mission["duration_s"]:g} s</text>']
    for route in routes:
        colour = colours[route['uav_id'] % len(colours)]
        x, y = project(route['x_min_m'], route['y_max_m'])
        zone_width = (route['x_max_m'] - route['x_min_m']) * scale
        zone_height = (route['y_max_m'] - route['y_min_m']) * scale
        parts.append(f'<rect x="{x:.4f}" y="{y:.4f}" width="{zone_width:.4f}" height="{zone_height:.4f}" fill="{colour}" fill-opacity="0.04" stroke="#aaa" stroke-dasharray="5 4"/>')
        polyline = ' '.join(f'{px:.4f},{py:.4f}' for px, py in (project(p[0], p[1]) for p in route['waypoints_m']))
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>')
        sx, sy = project(*route['waypoints_m'][0][:2])
        parts.append(f'<circle cx="{sx:.4f}" cy="{sy:.4f}" r="5" fill="{colour}"/>')
        ex, ey = project(*route['waypoints_m'][1][:2])
        norm = math.hypot(ex - sx, ey - sy)
        ux, uy = (ex - sx) / norm, (ey - sy) / norm
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        arrow = ((mx + 7 * ux, my + 7 * uy),
                 (mx - 6 * ux - 5 * uy, my - 6 * uy + 5 * ux),
                 (mx - 6 * ux + 5 * uy, my - 6 * uy - 5 * ux))
        arrow_points = ' '.join(f'{px:.4f},{py:.4f}' for px, py in arrow)
        parts.append(f'<polygon points="{arrow_points}" fill="{colour}"/>')
        parts.append(f'<text x="{x + 6:.4f}" y="{y + 20:.4f}" fill="{colour}">UAV {route["uav_id"]}</text>')
    sx, sy = project(*server[:2])
    parts.extend([f'<path d="M {sx - 6:.4f} {sy:.4f} H {sx + 6:.4f} M {sx:.4f} {sy - 6:.4f} V {sy + 6:.4f}" stroke="#111" stroke-width="3"/>',
                  f'<text x="{sx + 9:.4f}" y="{sy - 9:.4f}">Server</text>',
                  f'<text x="32" y="{height - 32}">{html.escape("x/y coordinates in metres; dots mark initial positions; routes repeat without hovering.")}</text>',
                  f'<text x="32" y="{height - 12}">Area: x [{mission["x_min_m"]:g}, {mission["x_max_m"]:g}], y [{mission["y_min_m"]:g}, {mission["y_max_m"]:g}]</text>',
                  '</g></svg>'])
    return '\n'.join(parts)


def generate_environment(config, manifest_path, output_dir=None):
    """根据配置和数据清单生成可复用场景，返回三个输出文件的绝对路径。

    config 遵循 config.py 的统一结构；manifest_path 指向已准备的 data.json，
    本函数只读取清单中的分配信息，不打开图片或重新划分数据。output_dir
    未指定时使用 config['output']['scenario_dir']。

    environment.csv 保存所有 UAV 共用的物理时间观测；scenario.json 保存
    配置副本、解析后的服务器能力、采集参数、轨迹及数据来源；patrol_map.svg
    展示固定飞行路径。返回键为 environment_csv、scenario_json、patrol_map_svg。

    manifest_sha256 校验清单原始字节，data_fingerprint 沿用数据准备阶段的
    来源标识，csv_sha256 关联本次环境时序。哈希用于识别内容变化，不替代
    数据或模型的统计有效性检查。先在内存中生成内容，再处理输出目录：
    三个已有文件均完全相同时直接复用，否则拒绝覆盖，要求选择新目录。
    """
    validate_config(config)
    manifest_path = Path(manifest_path).expanduser().resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest.decode('utf-8-sig'))
    collection = resolve_collection(config, manifest)
    # 采集容量来自已准备清单，避免环境配置与真正的每机样本上限脱节。
    mission = config['mission']
    routes = [_route(mission, uav) for uav in range(mission['num_uavs'])]
    times = _time_grid(mission, routes)
    output = Path(output_dir if output_dir is not None else config['output']['scenario_dir']).expanduser().resolve()
    paths = dict(environment_csv=str(output / 'environment.csv'),
                 scenario_json=str(output / 'scenario.json'),
                 patrol_map_svg=str(output / 'patrol_map.svg'))
    if manifest_path in [Path(value) for value in paths.values()]:
        raise ValueError('Environment output must not overwrite the data manifest')
    scenario = dict(
        schema_version=1, config=copy.deepcopy(config), collection=collection,
        resolved_hardware=dict(server=dict(
            parallel_branches=(config['server']['parallel_branches']
                               if config['server']['parallel_branches'] is not None else mission['num_uavs']),
            max_downlink_power_w=(config['server']['max_downlink_power_w']
                                 if config['server']['max_downlink_power_w'] is not None
                                 else mission['num_uavs'] * config['server']['downlink_power_w']))),
        uav_ids=list(range(mission['num_uavs'])), trajectories=routes,
        manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        data_fingerprint=manifest.get('fingerprint'),
        trace=dict(csv='environment.csv', start_time_s=0.0, end_time_s=mission['duration_s'],
                   observation_count=len(times), row_count=len(times) * mission['num_uavs'],
                   interval_s=mission['trace_interval_s'], includes_turning_times=True,
                   position_interpolation='linear', channel_interpolation='previous',
                   channel_process='independent LoS and fading draws on shared physical-time observations',
                   window_mean='last W CSV observations at or before query time'))
    # 使用局部随机数发生器，不影响训练代码的全局随机状态。遍历顺序固定为
    # “物理时间优先、UAV 编号次之”；不同算法读取同一 CSV，而非按轮数重抽。
    rng = random.Random(config['channel']['seed'])
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for time_s in times:
        for route in routes:
            position = _position(route, time_s)
            channel = _channel(position, config['server']['position_m'], config['channel'], rng)
            writer.writerow(dict(time_s=time_s, uav_id=route['uav_id'],
                                 x_m=position[0], y_m=position[1], z_m=position[2], **channel))
    # 对最终实际写入的 UTF-8 字节计算哈希，读取时可发现 CSV 被单独改动。
    csv_bytes = buffer.getvalue().encode('utf-8')
    scenario['trace']['csv_sha256'] = hashlib.sha256(csv_bytes).hexdigest()
    expected = {
        'environment_csv': csv_bytes,
        'scenario_json': (json.dumps(scenario, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'),
        'patrol_map_svg': (_patrol_svg(config, routes) + '\n').encode('utf-8'),
    }
    # 完全相同的重复运行可复用；部分残留或参数不同的输出不能静默覆盖。
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if output.is_dir() and all(Path(paths[key]).is_file() and Path(paths[key]).read_bytes() == value
                                   for key, value in expected.items()):
            return paths
        raise ValueError('Output contains a different or incomplete scenario; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for key, value in expected.items():
        Path(paths[key]).write_bytes(value)
    return paths


class EnvironmentTrace:
    """对已生成环境进行只读查询，接口以物理时间 s 和从 0 开始的 UAV 编号定位。

    position_m 及 x/y/z 对应查询时刻；channel_gain 与传播状态、路径损耗、
    distance_m 等诊断量对应最近过去的 channel_time_s。若场景含服务器
    坐标，current_distance_m 另给出查询时刻的实际距离，避免混淆两种时间。

    对所有 UAV 使用同一时间网格，支持重复及乱序查询。读取未来航点用于
    已知固定轨迹的插值，不等于读取未来信道；CSI 始终采用最近过去值。
    不进行训练、资源优化或随机采样，也不把当前观测外推到任务范围之外。
    """

    def __init__(self, csv_path, scenario_path=None):
        """加载 CSV，并校验字段、数值、时间顺序及各 UAV 时间网格的一致性。

        scenario_path 默认寻找 CSV 同目录下的 scenario.json。自动寻找时
        允许该文件不存在，此时仍可查询位置和信道，但无法查询样本计数；
        显式指定的场景文件不存在则报错。若元数据提供 CSV 哈希，则先核对
        字节内容，再验证 UAV 编号和任务时长是否匹配，避免混用两个场景。
        """
        self.csv_path = Path(csv_path).expanduser().resolve()
        candidate = self.csv_path.with_name('scenario.json') if scenario_path is None else Path(scenario_path).expanduser().resolve()
        self.scenario = None
        if candidate.exists():
            self.scenario = json.loads(candidate.read_text(encoding='utf-8-sig'))
            if not isinstance(self.scenario, dict):
                raise ValueError('scenario.json must contain an object')
            checksum = self.scenario.get('trace', {}).get('csv_sha256')
            if checksum is not None and hashlib.sha256(self.csv_path.read_bytes()).hexdigest() != checksum:
                raise ValueError('Environment CSV does not match the scenario checksum')
        elif scenario_path is not None:
            raise FileNotFoundError(candidate)
        # 以 UAV 为键保存行，保留 CSV 原有时间顺序；不靠排序修复错误输入。
        grouped = {}
        previous_key = None
        with self.csv_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not set(CSV_FIELDS).issubset(reader.fieldnames):
                raise ValueError('Environment CSV is missing required columns')
            for line, source in enumerate(reader, 2):
                try:
                    row = {key: float(source[key]) for key in CSV_FIELDS if key not in ('uav_id', 'los')}
                    row['uav_id'] = int(source['uav_id'])
                    if source['los'] not in ('0', '1'):
                        raise ValueError('los must be 0 or 1')
                    row['los'] = bool(int(source['los']))
                except (TypeError, ValueError) as error:
                    raise ValueError(f'Invalid environment CSV row {line}: {error}') from error
                if any(not math.isfinite(value) for value in row.values()):
                    raise ValueError(f'Nonfinite value at CSV row {line}')
                if (row['time_s'] < 0 or row['uav_id'] < 0 or row['distance_m'] <= 0
                        or row['channel_gain'] <= 0 or not 0 <= row['los_probability'] <= 1):
                    raise ValueError(f'Invalid physical value at CSV row {line}')
                key = (row['time_s'], row['uav_id'])
                if previous_key is not None and key <= previous_key:
                    raise ValueError('CSV rows must be unique and ordered by (time_s, uav_id)')
                previous_key = key
                grouped.setdefault(row['uav_id'], []).append(row)
        if not grouped:
            raise ValueError('Environment CSV is empty')
        self.uav_ids = tuple(sorted(grouped))
        if self.uav_ids != tuple(range(len(self.uav_ids))):
            raise ValueError('UAV IDs must be consecutive integers starting at zero')
        # 全体 UAV 必须覆盖完全相同的时间点，保证同一物理时刻可公平回放。
        self.times = tuple(row['time_s'] for row in grouped[self.uav_ids[0]])
        if self.times[0] != 0 or len(self.times) < 2:
            raise ValueError('Environment trace must start at zero and contain an end time')
        for uav, rows in grouped.items():
            if tuple(row['time_s'] for row in rows) != self.times:
                raise ValueError(f'UAV {uav} does not have the complete shared time grid')
        self._rows = grouped
        self.duration_s = self.times[-1]
        self.collection = self.scenario.get('collection') if self.scenario else None
        self._server_position = None
        if self.scenario is not None:
            if self.scenario.get('uav_ids') != list(self.uav_ids):
                raise ValueError('Scenario and CSV UAV IDs do not match')
            configured_duration = self.scenario.get('config', {}).get('mission', {}).get('duration_s')
            if configured_duration != self.duration_s:
                raise ValueError('Scenario and CSV mission durations do not match')
            self._server_position = self.scenario.get('config', {}).get('server', {}).get('position_m')

    def _index(self, time_s):
        """校验查询时间并返回不晚于该时刻的最后一个观测下标。

        time_s 必须是任务区间 [0, duration_s] 内的有限秒数。恰好命中观测
        时刻时使用该行；不接受负时间、超出任务终点、NaN 或无穷大，亦不
        静默截断或外推。二分查找仅定位已存储数据，不改变读取状态。
        """
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            raise ValueError('Query time must be a finite number of seconds')
        if not 0 <= time_s <= self.duration_s:
            raise ValueError(f'Query time must lie in [0, {self.duration_s}] seconds')
        return bisect.bisect_right(self.times, time_s) - 1

    def state_at(self, t, uav_id):
        """返回单架 UAV 在 t 秒的坐标，以及最近过去的 CSI 和观测诊断量。

        相邻观测之间按时间比例插值位置；由于生成网格保留全部转角，这一
        插值遵循原定恒速折线路径。信道增益、LoS 状态及损耗不插值，始终
        保留左侧观测值。next_change_time_s 表示下一次保存的 CSI 观测时刻，
        任务终点为 None，调用方可据此识别当前信道保持区间。

        返回新字典，不修改内部记录。position_m、x_m/y_m/z_m 单位为 m；
        distance_m 与 channel_distance_m 属于 channel_time_s 的链路距离，
        current_distance_m 则在已知服务器坐标时按当前插值位置重新计算。
        """
        index = self._index(t)
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or uav_id not in self._rows:
            raise ValueError(f'Unknown UAV ID: {uav_id!r}')
        observed = self._rows[uav_id][index]
        position = [observed[key] for key in ('x_m', 'y_m', 'z_m')]
        next_time = self.times[index + 1] if index + 1 < len(self.times) else None
        # 只使用下一行的已知轨迹坐标；不读取该行的 channel_gain 来插值。
        if next_time is not None:
            fraction = (t - self.times[index]) / (next_time - self.times[index])
            following = self._rows[uav_id][index + 1]
            position = [a + fraction * (following[key] - a)
                        for key, a in zip(('x_m', 'y_m', 'z_m'), position)]
        result = dict(observed, time_s=float(t), channel_time_s=self.times[index],
                      channel_distance_m=observed['distance_m'], next_change_time_s=next_time,
                      position_m=position, x_m=position[0], y_m=position[1], z_m=position[2])
        if self._server_position is not None:
            result['current_distance_m'] = math.dist(position, self._server_position)
        return result

    def positions_at(self, t):
        """返回 t 秒全部 UAV 的 {uav_id: [x_m, y_m, z_m]}，不推进物理时间。"""
        return {uav: self.state_at(t, uav)['position_m'] for uav in self.uav_ids}

    def channel_at(self, t, uav_id):
        """返回指定 UAV 在 t 秒可用的最近过去线性功率增益，不含资源分配。"""
        return self.state_at(t, uav_id)['channel_gain']

    def window_mean(self, t, window_size=20):
        """返回各 UAV 截至 t 秒最近 W 个 CSV 信道观测的算术平均。

        window_size 即 W，是观测个数，不是秒数、训练 epoch 数或接口调用
        次数；转角补充的时间点也算一次观测。初始阶段不足 W 个时使用已有
        全部观测，任务起点只使用首行。返回 {uav_id: 平均线性功率增益}，
        不包含未来观测，也不对 dB 值、传输速率或时间长度进行加权平均。
        """
        index = self._index(t)
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError('window_size must be a positive integer')
        start = max(0, index + 1 - window_size)
        return {uav: math.fsum(row['channel_gain'] for row in rows[start:index + 1]) / (index + 1 - start)
                for uav, rows in self._rows.items()}

    def sample_count_at(self, t):
        """返回 t 秒每架 UAV 已可用的累计样本数，为所有 UAV 共用的整数。

        使用 scenario.json 中由实际清单解析出的初始缓存、共同采集速率
        （样本/s）和单机样本上限，具体计数由 available_samples_at 完成。
        此接口只计算到达数量，不读图、不改变顺序，也不决定当前 epoch 的
        数据快照；训练方应在 epoch 开始时调用并冻结相应样本集合。
        """
        self._index(t)
        if self.collection is None:
            raise ValueError('A scenario with collection settings is required for sample counts')
        return available_samples_at(t, self.collection)
