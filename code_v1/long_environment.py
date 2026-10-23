"""用途：流式准备完整长环境、有限缓存读取分片，并提供独立校准信道。

运行方式（在 code_v1 目录执行）：
    python -X utf8 -B train_sfl.py --prepare-only
    python -X utf8 -B train_sfl.py --prepared-run output/output_具体时间戳

本模块由训练准备入口调用，没有独立命令行入口。正式环境在训练前一次
生成完毕：每个 CSV 分片不超过指定字节数，完整的同一时刻 UAV 记录组
始终位于一个分片中。scenario.json 保存文件索引、时域、哈希与采集参数。
读取时只保留至多两个分片，随机查询不重新抽样，也不续写环境。

原始 600 s 采集设定先解析为固定速率，再延长查询时域；延长任务不会把
采集速率重新除以总时长。ReferenceChannel 使用独立种子流式预演，无需
保存校准 CSV，随机抽样顺序与正式环境生成器一致，不访问正式环境未来。
"""
from __future__ import annotations

import bisect
import copy
import csv
import hashlib
import heapq
import io
import itertools
import json
import math
import random
from array import array
from collections import OrderedDict
from pathlib import Path

from config import available_samples_at, resolve_collection, validate_config
from environment import CSV_FIELDS, EnvironmentTrace, _channel, _patrol_svg, _position, _route


def _sha256(path):
    """按小块计算文件 SHA-256，不把完整长环境读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _time_grid_iter(mission, routes, duration_s=None):
    """逐个产生共同物理时间点，合并规则网格、航迹转角与有限终点。

    duration_s=None 表示校准用的无限时间流。堆合并只保留每条输入流的
    下一个时间点；内存与任务长度无关。浮点去重容差与旧环境一致，故
    相同种子、时域与参数在单文件和分片生成方式中得到相同 CSI 序列。
    """
    interval = mission['trace_interval_s']
    tolerance = min(interval * 1e-7, 1e-9)

    def regular():
        """流式产生规则采样网格，内存与任务总时长无关。"""
        for index in itertools.count():
            value = index * interval
            if duration_s is not None and value > duration_s:
                break
            yield value

    def turns(route):
        """逐圈产生该航线的转角时刻，使坐标插值保持恒速折线路径。"""
        period = route['loop_time_s']
        for cycle in itertools.count():
            start = cycle * period
            if duration_s is not None and start > duration_s:
                break
            for distance in route['cumulative_distance_m']:
                value = start + distance / route['speed_m_s']
                if duration_s is None or value <= duration_s:
                    yield value

    streams = [regular(), *(turns(route) for route in routes)]
    if duration_s is not None:
        streams.append(iter((float(duration_s),)))
    # 暂存一个点，便于把与终点近重合的浮点数替换为精确的任务终点。
    previous = None
    for value in heapq.merge(*streams):
        if previous is not None and value - previous <= tolerance:
            if duration_s is not None and value == duration_s:
                previous = float(duration_s)
            continue
        if previous is not None:
            yield previous
        previous = value
    if previous is not None:
        yield previous


def _header_bytes():
    """生成与旧 CSV 完全相同的 UTF-8 表头及换行约定。"""
    buffer = io.StringIO(newline='')
    csv.DictWriter(buffer, fieldnames=CSV_FIELDS).writeheader()
    return buffer.getvalue().encode('utf-8')


def _group_bytes(time_s, routes, config, rng):
    """序列化一个共同时间点的全部 UAV；该记录组不可跨分片切开。"""
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    for route in routes:
        position = _position(route, time_s)
        channel = _channel(position, config['server']['position_m'], config['channel'], rng)
        writer.writerow(dict(time_s=time_s, uav_id=route['uav_id'],
                             x_m=position[0], y_m=position[1], z_m=position[2], **channel))
    return buffer.getvalue().encode('utf-8')


def generate_long_environment(config, manifest_path, output_dir, duration_s,
                              collection=None, max_part_bytes=20_000_000):
    """训练前生成覆盖给定总期限的完整分片环境，返回 scenario.json 的 Path。

    duration_s 单位 s，向上对齐 trace_interval_s 网格。collection 省略时
    先用原始任务时长解析固定采集速率；显式提供时仍核对真实清单容量。
    仅改变 collection.duration_s，原始初始缓存、速率和样本封顶保持不变。
    新目录可为空；完整相同场景可校验后复用，不覆盖其他场景或残留文件。
    """
    validate_config(config)
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError('Long-environment duration_s must be positive and finite')
    if (isinstance(max_part_bytes, bool) or not isinstance(max_part_bytes, int)
            or not 1 <= max_part_bytes <= 20_000_000):
        raise ValueError('max_part_bytes must be between 1 and 20,000,000 bytes')
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    source_collection = resolve_collection(config, manifest)
    chosen_collection = copy.deepcopy(source_collection if collection is None else collection)
    for key in ('num_uavs', 'max_samples_per_uav'):
        if chosen_collection.get(key) != source_collection[key]:
            raise ValueError(f'Long environment collection.{key} does not match the data manifest')
    # 对提供的采集计划先做通常的数值检查；之后只延长允许查询的时间。
    available_samples_at(0, chosen_collection)
    interval = config['mission']['trace_interval_s']
    aligned_duration = math.ceil(float(duration_s) / interval) * interval
    chosen_collection['duration_s'] = aligned_duration
    prepared_config = copy.deepcopy(config)
    prepared_config['mission']['duration_s'] = aligned_duration
    # 保存在场景中的 rate 显式化，防止读取方对长任务重新推算较低采集率。
    prepared_config['data']['sample_rate_per_s'] = chosen_collection['rate_samples_per_s']
    output = Path(output_dir).expanduser().resolve()
    scenario_path = output / 'scenario.json'
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if scenario_path.is_file():
            existing = json.loads(scenario_path.read_text(encoding='utf-8-sig'))
            if (existing.get('schema_version') == 2 and existing.get('config') == prepared_config
                    and existing.get('manifest_sha256') == manifest_sha
                    and existing.get('collection') == chosen_collection
                    and existing.get('trace', {}).get('max_part_bytes') == max_part_bytes):
                parts = existing.get('trace', {}).get('parts', [])
                if (parts and all((output / part['file']).is_file()
                                  and _sha256(output / part['file']) == part['sha256']
                                  for part in parts)
                        and (output / 'patrol_map.svg').is_file()):
                    return scenario_path
        raise ValueError('Long-environment output contains a different or incomplete scenario')
    if output == manifest_path or manifest_path.parent == output:
        raise ValueError('Long environment must use its own directory, separate from data.json')
    output.mkdir(parents=True, exist_ok=True)
    mission = prepared_config['mission']
    routes = [_route(mission, uav) for uav in range(mission['num_uavs'])]
    header = _header_bytes()
    rng = random.Random(prepared_config['channel']['seed'])
    parts, created, handle = [], [], None
    total_hash = hashlib.sha256()
    observation_count = 0
    part_info, part_hash = None, None

    def close_part():
        """关闭当前CSV分片，固化记录数、首末时刻和SHA256，再加入索引。"""
        nonlocal handle
        if handle is not None:
            handle.close()
            handle = None
            part_info['sha256'] = part_hash.hexdigest()
            parts.append(part_info)

    try:
        for time_s in _time_grid_iter(mission, routes, aligned_duration):
            group = _group_bytes(time_s, routes, prepared_config, rng)
            if len(header) + len(group) > max_part_bytes:
                raise ValueError('max_part_bytes cannot hold a header and one complete UAV time group')
            if handle is None or part_info['bytes'] + len(group) > max_part_bytes:
                close_part()
                name = f'environment_{len(parts):05d}.csv'
                path = output / name
                handle = path.open('xb')
                created.append(path)
                part_hash = hashlib.sha256()
                handle.write(header)
                part_hash.update(header)
                total_hash.update(header)
                part_info = dict(file=name, start_time_s=time_s, end_time_s=time_s,
                                 start_observation=observation_count, observation_count=0,
                                 row_count=0, bytes=len(header))
            handle.write(group)
            part_hash.update(group)
            total_hash.update(group)
            part_info['end_time_s'] = time_s
            part_info['observation_count'] += 1
            part_info['row_count'] += mission['num_uavs']
            part_info['bytes'] += len(group)
            observation_count += 1
        close_part()
        trace_hash = total_hash.hexdigest()
        scenario = dict(
            schema_version=2, config=prepared_config, collection=chosen_collection,
            source_mission_duration_s=float(config['mission']['duration_s']),
            resolved_hardware=dict(server=dict(
                parallel_branches=(config['server']['parallel_branches']
                                   if config['server']['parallel_branches'] is not None else mission['num_uavs']),
                max_downlink_power_w=(config['server']['max_downlink_power_w']
                                     if config['server']['max_downlink_power_w'] is not None
                                     else mission['num_uavs'] * config['server']['downlink_power_w']))),
            uav_ids=list(range(mission['num_uavs'])), trajectories=routes,
            manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
            data_fingerprint=manifest.get('fingerprint'),
            trace=dict(format='chunked_csv', csv=parts[0]['file'], parts=parts,
                       max_part_bytes=max_part_bytes, start_time_s=0.0,
                       end_time_s=aligned_duration, requested_end_time_s=float(duration_s),
                       observation_count=observation_count,
                       row_count=observation_count * mission['num_uavs'],
                       interval_s=interval, includes_turning_times=True,
                       position_interpolation='linear', channel_interpolation='previous',
                       channel_process='independent LoS and fading draws on shared physical-time observations',
                       window_mean='last W CSV observations at or before query time',
                       csv_sha256=trace_hash,
                       checksum_definition='SHA-256 of ordered complete CSV part bytes, including each header'))
        svg_path = output / 'patrol_map.svg'
        svg_path.write_text(_patrol_svg(prepared_config, routes) + '\n', encoding='utf-8')
        created.append(svg_path)
        scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2,
                                             allow_nan=False) + '\n', encoding='utf-8')
        created.append(scenario_path)
        return scenario_path
    except BaseException:
        # 只清理本次已创建的文件；不遍历或删除用户的其他目录内容。
        if handle is not None:
            handle.close()
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


class ChunkedEnvironmentTrace:
    """分片环境的只读查询器；仅保存索引和至多 cache_parts 个已解码分片。

    每个分片按列存为标准库 array，避免逐行大字典造成大量内存开销。
    分片首次加载时检查文件长度、哈希、数值、顺序和共同时间网格；跨
    分片坐标插值只读取下一分片的已知航迹，信道仍取最近历史记录。
    """

    def __init__(self, scenario_path, cache_parts=2):
        """读取参数与索引，建立有限缓存或独立校准随机流；不加载完整长时域。"""
        if isinstance(cache_parts, bool) or not isinstance(cache_parts, int) or cache_parts < 2:
            raise ValueError('cache_parts must be an integer of at least two')
        self.scenario_path = Path(scenario_path).expanduser().resolve()
        self.scenario = json.loads(self.scenario_path.read_text(encoding='utf-8-sig'))
        trace = self.scenario.get('trace', {})
        self.parts = trace.get('parts', [])
        if not self.parts or trace.get('format') != 'chunked_csv':
            raise ValueError('Scenario does not contain a chunked CSV index')
        self._directory = self.scenario_path.parent
        self._cache = OrderedDict()
        self._cache_limit = cache_parts
        self._checked = set()
        self.uav_ids = tuple(self.scenario['uav_ids'])
        if not self.uav_ids or self.uav_ids != tuple(range(len(self.uav_ids))):
            raise ValueError('UAV IDs must be consecutive integers starting at zero')
        previous_end, total_observations = -math.inf, 0
        for part in self.parts:
            path = (self._directory / part['file']).resolve()
            if path.parent != self._directory or not path.is_file():
                raise ValueError('CSV part must be a file within its scenario directory')
            if (part['start_time_s'] <= previous_end or part['end_time_s'] < part['start_time_s']
                    or part['observation_count'] < 1
                    or part['row_count'] != part['observation_count'] * len(self.uav_ids)
                    or part['start_observation'] != total_observations
                    or part['bytes'] > trace['max_part_bytes']):
                raise ValueError('Invalid or inconsistent CSV part index')
            previous_end = part['end_time_s']
            total_observations += part['observation_count']
        self.duration_s = float(trace['end_time_s'])
        if (self.parts[0]['start_time_s'] != 0 or previous_end != self.duration_s
                or total_observations != trace['observation_count']
                or trace['row_count'] != total_observations * len(self.uav_ids)
                or self.scenario['config']['mission']['duration_s'] != self.duration_s):
            raise ValueError('Scenario duration or record count does not match the part index')
        self._starts = tuple(part['start_time_s'] for part in self.parts)
        self.collection = self.scenario.get('collection')
        self._server_position = self.scenario['config']['server']['position_m']
        # csv_path 供旧的文件位置展示使用；完整文件集必须以 parts 为准。
        self.csv_path = self._directory / self.parts[0]['file']

    def _load_part(self, part_index):
        """加载一个分片并更新 LRU；不会预加载全部长时域记录。"""
        if part_index in self._cache:
            self._cache.move_to_end(part_index)
            return self._cache[part_index]
        metadata = self.parts[part_index]
        path = self._directory / metadata['file']
        if path.stat().st_size != metadata['bytes'] or _sha256(path) != metadata['sha256']:
            raise ValueError(f'CSV part does not match its checksum: {path.name}')
        fields = [key for key in CSV_FIELDS if key not in ('time_s', 'uav_id')]
        grouped = {uav: {key: array('b' if key == 'los' else 'd') for key in fields}
                   for uav in self.uav_ids}
        times = array('d')
        rows = 0
        previous_time = -math.inf
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(CSV_FIELDS):
                raise ValueError(f'CSV part has unexpected columns: {path.name}')
            for source in reader:
                try:
                    uav = int(source['uav_id'])
                    time_s = float(source['time_s'])
                    values = {key: float(source[key]) for key in fields}
                except (TypeError, ValueError) as error:
                    raise ValueError(f'Invalid CSV values in {path.name}') from error
                if (uav != rows % len(self.uav_ids) or not math.isfinite(time_s)
                        or any(not math.isfinite(value) for value in values.values())
                        or values['channel_gain'] <= 0 or values['distance_m'] <= 0
                        or not 0 <= values['los_probability'] <= 1 or values['los'] not in (0, 1)):
                    raise ValueError(f'Invalid physical record or UAV order in {path.name}')
                if uav == 0:
                    if time_s <= previous_time:
                        raise ValueError(f'CSV observation times must strictly increase: {path.name}')
                    times.append(time_s)
                    previous_time = time_s
                elif time_s != previous_time:
                    raise ValueError(f'UAVs must share one time grid: {path.name}')
                for key, value in values.items():
                    grouped[uav][key].append(int(value) if key == 'los' else value)
                rows += 1
        if (rows != metadata['row_count'] or len(times) != metadata['observation_count']
                or times[0] != metadata['start_time_s'] or times[-1] != metadata['end_time_s']):
            raise ValueError(f'CSV records do not match part index: {path.name}')
        self._cache[part_index] = (times, grouped)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return times, grouped

    def _index(self, time_s):
        """从分片起点索引定位最近历史观测，返回分片号和分片内下标。"""
        if (isinstance(time_s, bool) or not isinstance(time_s, (int, float))
                or not math.isfinite(time_s) or not 0 <= time_s <= self.duration_s):
            raise ValueError(f'Query time must lie in [0, {self.duration_s}] seconds')
        part_index = bisect.bisect_right(self._starts, time_s) - 1
        times, _ = self._load_part(part_index)
        return part_index, bisect.bisect_right(times, time_s) - 1

    def state_at(self, t, uav_id):
        """返回任意 t 秒位置及最近历史 CSI；跨分片保持相同查询语义。"""
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or uav_id not in self.uav_ids:
            raise ValueError(f'Unknown UAV ID: {uav_id!r}')
        part_index, index = self._index(t)
        times, grouped = self._load_part(part_index)
        columns = grouped[uav_id]
        observed = {key: values[index] for key, values in columns.items()}
        observed.update(time_s=times[index], uav_id=uav_id, los=bool(observed['los']))
        position = [observed[key] for key in ('x_m', 'y_m', 'z_m')]
        next_time, next_columns, next_index = None, None, None
        if index + 1 < len(times):
            next_time, next_columns, next_index = times[index + 1], columns, index + 1
        elif part_index + 1 < len(self.parts):
            following_times, following_grouped = self._load_part(part_index + 1)
            next_time, next_columns, next_index = following_times[0], following_grouped[uav_id], 0
        if next_time is not None:
            fraction = (t - times[index]) / (next_time - times[index])
            position = [value + fraction * (next_columns[key][next_index] - value)
                        for key, value in zip(('x_m', 'y_m', 'z_m'), position)]
        return dict(observed, time_s=float(t), channel_time_s=times[index],
                    channel_distance_m=observed['distance_m'], next_change_time_s=next_time,
                    position_m=position, x_m=position[0], y_m=position[1], z_m=position[2],
                    current_distance_m=math.dist(position, self._server_position))

    def positions_at(self, t):
        """返回全部 UAV 在 t 秒的三维位置，单位 m。"""
        return {uav: self.state_at(t, uav)['position_m'] for uav in self.uav_ids}

    def channel_at(self, t, uav_id=None):
        """返回最近历史线性功率增益；省略 UAV 编号时返回完整字典。"""
        if uav_id is None:
            return {uav: self.state_at(t, uav)['channel_gain'] for uav in self.uav_ids}
        return self.state_at(t, uav_id)['channel_gain']

    def window_mean(self, t, window_size=20):
        """计算最近 W 条 CSV 观测均值，跨分片查询仍只使用有限缓存。

        此接口保留旧读取语义。动态切分控制器的“20 次块起始观测”由
        控制器自己保存，不能用这里的 CSV 网格窗口替代。
        """
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError('window_size must be a positive integer')
        part_index, index = self._index(t)
        values = {uav: [] for uav in self.uav_ids}
        remaining = window_size
        while part_index >= 0 and remaining:
            times, grouped = self._load_part(part_index)
            end = min(index + 1, len(times))
            start = max(0, end - remaining)
            for uav in self.uav_ids:
                values[uav].extend(grouped[uav]['channel_gain'][start:end])
            remaining -= end - start
            part_index -= 1
            if part_index >= 0:
                index = self.parts[part_index]['observation_count'] - 1
        return {uav: math.fsum(gains) / len(gains) for uav, gains in values.items()}

    def sample_count_at(self, t):
        """返回固定速率采集至 t 秒的每机累计样本数，达到真实容量后封顶。"""
        self._index(t)
        if self.collection is None:
            raise ValueError('Scenario collection settings are required')
        return available_samples_at(t, self.collection)


def open_environment(path):
    """统一打开旧单 CSV 与新分片场景，path 可为目录、JSON 或 CSV。"""
    path = Path(path).expanduser().resolve()
    scenario_path = path / 'scenario.json' if path.is_dir() else (
        path if path.suffix.lower() == '.json' else path.with_name('scenario.json'))
    if scenario_path.exists():
        scenario = json.loads(scenario_path.read_text(encoding='utf-8-sig'))
        if scenario.get('trace', {}).get('format') == 'chunked_csv':
            return ChunkedEnvironmentTrace(scenario_path)
        csv_path = scenario_path.parent / scenario.get('trace', {}).get('csv', 'environment.csv')
        return EnvironmentTrace(csv_path, scenario_path)
    if path.is_dir() or path.suffix.lower() == '.json':
        raise FileNotFoundError(scenario_path)
    return EnvironmentTrace(path)


class ReferenceChannel:
    """独立校准信道，按非递减物理时间流式推进，不保存长 CSV。

    seed 省略时采用正式信道种子加 1；传入时用于复现实验。channel_at(t)
    返回全部 UAV 的增益字典，channel_at(t, id) 返回单个增益。为了维持
    与已生成环境相同的随机抽样顺序，跳过的网格观测也会按顺序生成，
    但只保留当前一条；不支持向过去跳转，可在需要时新建同种子实例。
    """

    def __init__(self, config, seed=None):
        """读取参数与索引，建立有限缓存或独立校准随机流；不加载完整长时域。"""
        validate_config(config)
        self.config = copy.deepcopy(config)
        self.config['channel']['seed'] = int(config['channel']['seed'] + 1 if seed is None else seed)
        self.seed = self.config['channel']['seed']
        self.routes = [_route(config['mission'], uav) for uav in range(config['mission']['num_uavs'])]
        self.uav_ids = tuple(range(len(self.routes)))
        self._rng = random.Random(self.seed)
        self._grid = iter(_time_grid_iter(config['mission'], self.routes))
        self._next_time = next(self._grid)
        self._last_query = -math.inf
        self._observed_time, self._observed = None, None
        self.channel_at(0.0)

    def channel_at(self, time_s, uav_id=None):
        """返回不晚于 time_s 的校准 CSI；时间单位 s，不读取正式信道。"""
        if (isinstance(time_s, bool) or not isinstance(time_s, (int, float))
                or not math.isfinite(time_s) or time_s < self._last_query):
            raise ValueError('Calibration channel queries must use finite nondecreasing times')
        while self._next_time <= time_s:
            self._observed = {
                route['uav_id']: _channel(_position(route, self._next_time),
                                          self.config['server']['position_m'],
                                          self.config['channel'], self._rng)['channel_gain']
                for route in self.routes}
            self._observed_time = self._next_time
            self._next_time = next(self._grid)
        self._last_query = time_s
        if uav_id is None:
            return dict(self._observed)
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or uav_id not in self.uav_ids:
            raise ValueError(f'Unknown UAV ID: {uav_id!r}')
        return self._observed[uav_id]
