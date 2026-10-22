"""Shared physical-time patrol and channel trace; uses only the standard library.

The generator runs once for a scenario. Readers never draw random numbers: position
is linearly interpolated and CSI is held from the latest past observation.
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


CSV_FIELDS = (
    'time_s', 'uav_id', 'x_m', 'y_m', 'z_m', 'distance_m',
    'los_probability', 'los', 'path_loss_db', 'channel_gain',
)


def _route(mission, uav_id):
    """Parallel centre lines inside one x strip, joined into a rectilinear loop."""
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
    # An odd lane count closes along the top, then the first centre line.
    # A single centre line is a continuous out-and-back loop.
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
    distance = (time_s * route['speed_m_s']) % route['loop_length_m']
    cumulative, points = route['cumulative_distance_m'], route['waypoints_m']
    index = min(bisect.bisect_right(cumulative, distance) - 1, len(points) - 2)
    fraction = (distance - cumulative[index]) / (cumulative[index + 1] - cumulative[index])
    return [a + fraction * (b - a) for a, b in zip(points[index], points[index + 1])]


def _time_grid(mission, routes):
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
    # Merge only floating-point duplicates; keep all genuine turning times.
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
    difference = [a - b for a, b in zip(position, server_position)]
    distance = math.sqrt(sum(value * value for value in difference))
    if distance <= 0:
        raise ValueError('A UAV trajectory must not intersect the server position')
    elevation = math.degrees(math.atan2(abs(difference[2]), math.hypot(*difference[:2])))
    exponent = -parameters['los_b'] * (elevation - parameters['los_a'])
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
    mission, server = config['mission'], config['server']['position_m']
    width, height, padding = 960, 760, 85
    xmin, xmax = min(mission['x_min_m'], server[0]), max(mission['x_max_m'], server[0])
    ymin, ymax = min(mission['y_min_m'], server[1]), max(mission['y_max_m'], server[1])
    scale = min((width - 2 * padding) / (xmax - xmin), (height - 2 * padding) / (ymax - ymin))
    xoffset = (width - (xmax - xmin) * scale) / 2
    yoffset = (height - (ymax - ymin) * scale) / 2

    def project(x, y):
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
    """Write one reproducible shared environment and return its three file paths."""
    validate_config(config)
    manifest_path = Path(manifest_path).expanduser().resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest.decode('utf-8-sig'))
    collection = resolve_collection(config, manifest)
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
    csv_bytes = buffer.getvalue().encode('utf-8')
    scenario['trace']['csv_sha256'] = hashlib.sha256(csv_bytes).hexdigest()
    expected = {
        'environment_csv': csv_bytes,
        'scenario_json': (json.dumps(scenario, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'),
        'patrol_map_svg': (_patrol_svg(config, routes) + '\n').encode('utf-8'),
    }
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
    """Read-only environment lookup; query order never changes the stored process.

    ``state_at`` interpolates x/y/z and ``position_m`` at the query time. Channel
    diagnostics (including ``distance_m``) belong to ``channel_time_s``. When a
    scenario is available, ``current_distance_m`` also gives the query-time range.
    """

    def __init__(self, csv_path, scenario_path=None):
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
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            raise ValueError('Query time must be a finite number of seconds')
        if not 0 <= time_s <= self.duration_s:
            raise ValueError(f'Query time must lie in [0, {self.duration_s}] seconds')
        return bisect.bisect_right(self.times, time_s) - 1

    def state_at(self, t, uav_id):
        index = self._index(t)
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or uav_id not in self._rows:
            raise ValueError(f'Unknown UAV ID: {uav_id!r}')
        observed = self._rows[uav_id][index]
        position = [observed[key] for key in ('x_m', 'y_m', 'z_m')]
        next_time = self.times[index + 1] if index + 1 < len(self.times) else None
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
        return {uav: self.state_at(t, uav)['position_m'] for uav in self.uav_ids}

    def channel_at(self, t, uav_id):
        return self.state_at(t, uav_id)['channel_gain']

    def window_mean(self, t, window_size=20):
        """Arithmetic mean of up to W most recent stored observations, causally."""
        index = self._index(t)
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError('window_size must be a positive integer')
        start = max(0, index + 1 - window_size)
        return {uav: math.fsum(row['channel_gain'] for row in rows[start:index + 1]) / (index + 1 - start)
                for uav, rows in self._rows.items()}

    def sample_count_at(self, t):
        self._index(t)
        if self.collection is None:
            raise ValueError('A scenario with collection settings is required for sample counts')
        return available_samples_at(t, self.collection)
