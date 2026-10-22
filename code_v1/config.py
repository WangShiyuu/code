"""统一配置入口。单位写在键名中；环境回放只依赖 Python 标准库。

主要修改 CONFIG。也可通过 prepare_scenario.py --config 配置.json 局部覆盖。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def default_data_root():
    for parent in ROOT.parents:
        candidate = parent / "DataSet" / "CBCL StreetScenes"
        if all((candidate / name).is_file() for name in ("Images.zip", "Annotations.zip")):
            return str(candidate)
    return str(ROOT / "datasets" / "CBCL StreetScenes")


CONFIG = {
    "mission": {
        "num_uavs": 5, "duration_s": 600.0,
        "x_min_m": -500.0, "x_max_m": 500.0,
        "y_min_m": -500.0, "y_max_m": 500.0,
        "altitude_m": 100.0, "speed_m_s": 10.0,
        "lane_spacing_m": 100.0, "trace_interval_s": 0.1,
    },
    "data": {
        "root": default_data_root(), "output": str(ROOT / "output" / "data"),
        "image_size": 64, "val_fraction": 0.15, "test_fraction": 0.15,
        "seed": 42,
        # None: 用实际训练池等分；整数: 每架无人机统一的样本上限。
        "samples_per_uav": None,
        # 初始缓存属于样本上限的一部分；None 时按 initial_fraction 取整。
        "initial_samples_per_uav": None, "initial_fraction": 0.1,
        # None: 剩余样本均匀释放到任务结束；显式值单位为样本/秒。
        "sample_rate_per_s": None,
    },
    "training": {
        "total_epochs": 100, "aggregation_period": 2, "epoch_step_size": 0.18,
        "batch_size": 32, "eval_batch_size": 64,
        "optimizer": "sgd", "momentum": 0.0, "weight_decay": 0.0,
        "scheduler": "constant", "bn": "frozen", "initialization_seed": 42,
        "initialization_checkpoint": None, "device": "auto", "workers": 0,
        "split_profile_path": None,
    },
    "uav": {
        "f_min_hz": 2e8, "f_max_hz": 1.5e9, "f_ref_hz": 1e9,
        "kappa": 1e-28, "cycles_per_flop": 0.5,
        "p_up_min_w": 0.01, "p_up_max_w": 1.0, "p_up_ref_w": 0.5,
    },
    "server": {
        "position_m": [0.0, 0.0, 0.0],
        # None: 每架无人机一个独立后缀分支。
        "parallel_branches": None, "frequency_hz": 3e9,
        "cycles_per_flop": 0.03, "kappa": 1e-28,
        "downlink_power_w": 2.0, "max_downlink_power_w": None,
        "aggregation_time_s": 0.01, "aggregation_power_w": 10.0,
    },
    "radio": {
        "total_up_hz": 2e7, "total_down_hz": 2e7,
        "min_up_hz": 1e5, "min_down_hz": 1e5,
    },
    "channel": {
        "seed": 42, "carrier_hz": 2.4e9, "noise_psd_w_hz": 3.981e-21,
        "los_a": 9.61, "los_b": 0.16, "los_loss_db": 1.0,
        "nlos_loss_db": 20.0, "rician_k": 10.0,
    },
    "selection": {
        "candidates": [f"P{i}" for i in range(9)], "initial_split": "P4",
        "channel_window_size": 20, "max_sweeps": 1, "energy_tolerance_j": 1e-9,
    },
    "energy": {
        # 每架无人机的共同参数，未校准时不填虚构数值。核算电量前必须填全。
        "initial_energy_j": None, "flight_energy_j": None,
        "fixed_energy_j": None, "reserve_energy_j": None,
    },
    "theory": {
        # 条件性归一化配置，不代表已验证具体网络的参数界。
        "L": 1.0, "G": 1.0, "V": 0.02, "rho": 0.002,
        "delta": 0.10, "epsilon": 0.05,
    },
    "output": {"scenario_dir": str(ROOT / "output" / "scenario")},
}


def _merge(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError("配置覆盖必须为 JSON 对象")
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"未知配置项：{prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def load_config(path=None):
    config = copy.deepcopy(CONFIG)
    if path is not None:
        source = Path(path).expanduser().resolve()
        _merge(config, json.loads(source.read_text(encoding="utf-8-sig")))
        path_fields = [("data", "root"), ("data", "output"), ("output", "scenario_dir"),
                       ("training", "initialization_checkpoint"), ("training", "split_profile_path")]
        for section, key in path_fields:
            value = config[section][key]
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{section}.{key} 需要非空路径字符串")
                resolved = Path(value).expanduser()
                config[section][key] = str((source.parent / resolved).resolve())
    validate_config(config)
    return config


def _number(value, name, *, minimum=0.0, inclusive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} 必须为有限数且 {relation} {minimum}")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数")


def validate_config(config):
    required = set(CONFIG)
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"配置顶层字段必须为 {sorted(required)}")
    if any(not isinstance(config[key], dict) for key in required):
        raise ValueError("各配置分组必须为字典")
    for section in required:
        if set(config[section]) != set(CONFIG[section]):
            raise ValueError(f"{section} 配置字段不完整或含未知字段")
    mission, data = config["mission"], config["data"]
    _integer(mission["num_uavs"], "mission.num_uavs")
    for key in ("duration_s", "altitude_m", "speed_m_s", "lane_spacing_m", "trace_interval_s"):
        _number(mission[key], f"mission.{key}")
    for key in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"):
        value = mission[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"mission.{key} 必须为有限坐标")
    if mission["x_min_m"] >= mission["x_max_m"] or mission["y_min_m"] >= mission["y_max_m"]:
        raise ValueError("工作区域边界必须满足 min < max")
    if mission["trace_interval_s"] > mission["duration_s"]:
        raise ValueError("环境采样间隔不得大于任务时长")
    _integer(data["image_size"], "data.image_size", 32)
    _integer(data["seed"], "data.seed", 0)
    for key in ("root", "output"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"data.{key} 必须为非空路径")
    for key in ("val_fraction", "test_fraction", "initial_fraction"):
        _number(data[key], f"data.{key}")
    if data["val_fraction"] + data["test_fraction"] >= 1 or data["initial_fraction"] > 1:
        raise ValueError("验证测试比例之和须小于1，初始缓存比例不得大于1")
    for key in ("samples_per_uav", "initial_samples_per_uav"):
        if data[key] is not None:
            _integer(data[key], f"data.{key}")
    if data["sample_rate_per_s"] is not None:
        _number(data["sample_rate_per_s"], "data.sample_rate_per_s", inclusive=True)
    training = config["training"]
    for key in ("total_epochs", "aggregation_period", "batch_size", "eval_batch_size"):
        _integer(training[key], f"training.{key}")
    for key in ("workers", "initialization_seed"):
        _integer(training[key], f"training.{key}", 0)
    _number(training["epoch_step_size"], "training.epoch_step_size")
    if training["total_epochs"] % training["aggregation_period"]:
        raise ValueError("total_epochs 必须是 aggregation_period 的整数倍")
    if (training["optimizer"] != "sgd" or training["scheduler"] != "constant"
            or training["bn"] != "frozen" or training["momentum"] != 0
            or training["weight_decay"] != 0):
        raise ValueError("当前推导对应普通SGD、固定BN、固定epoch累计步长，不支持动量/权重衰减")
    if not isinstance(training["device"], str) or not training["device"]:
        raise ValueError("training.device 必须为非空设备字符串")
    for key in ("initialization_checkpoint", "split_profile_path"):
        if training[key] is not None and (not isinstance(training[key], str) or not training[key]):
            raise ValueError(f"training.{key} 需要非空路径或 None")
    uav = config["uav"]
    for key, value in uav.items():
        _number(value, f"uav.{key}", inclusive=key == "p_up_min_w")
    if not uav["f_min_hz"] <= uav["f_ref_hz"] <= uav["f_max_hz"]:
        raise ValueError("无人机参考频率越界")
    if not uav["p_up_min_w"] <= uav["p_up_ref_w"] <= uav["p_up_max_w"]:
        raise ValueError("无人机参考上行功率越界")
    server = config["server"]
    position = server["position_m"]
    if (not isinstance(position, (list, tuple)) or len(position) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in position)):
        raise ValueError("server.position_m 需要三个有限坐标")
    if mission["altitude_m"] <= position[2]:
        raise ValueError("无人机高度必须高于地面服务器")
    if server["parallel_branches"] is not None:
        _integer(server["parallel_branches"], "server.parallel_branches")
        if server["parallel_branches"] < mission["num_uavs"]:
            raise ValueError("服务器独立分支数量不能小于无人机数量")
    for key in ("frequency_hz", "cycles_per_flop", "kappa", "downlink_power_w", "aggregation_power_w"):
        _number(server[key], f"server.{key}")
    _number(server["aggregation_time_s"], "server.aggregation_time_s", inclusive=True)
    if server["max_downlink_power_w"] is not None:
        _number(server["max_downlink_power_w"], "server.max_downlink_power_w")
        if mission["num_uavs"] * server["downlink_power_w"] > server["max_downlink_power_w"]:
            raise ValueError("服务器下行总功率上限不足")
    radio = config["radio"]
    for key, value in radio.items():
        _number(value, f"radio.{key}")
    for direction in ("up", "down"):
        if mission["num_uavs"] * radio[f"min_{direction}_hz"] > radio[f"total_{direction}_hz"]:
            raise ValueError(f"{direction}方向总带宽不足以分配最低链路带宽")
    channel = config["channel"]
    _integer(channel["seed"], "channel.seed", 0)
    for key in ("carrier_hz", "noise_psd_w_hz", "los_a", "los_b"):
        _number(channel[key], f"channel.{key}")
    for key in ("los_loss_db", "nlos_loss_db", "rician_k"):
        _number(channel[key], f"channel.{key}", inclusive=True)
    selection = config["selection"]
    candidates = selection["candidates"]
    if (not isinstance(candidates, list) or not candidates
            or any(c not in tuple(f"P{i}" for i in range(9)) for c in candidates)
            or len(set(candidates)) != len(candidates)
            or selection["initial_split"] not in candidates):
        raise ValueError("切分候选应为互不重复的P0..P8，且包含初始切分")
    for key in ("channel_window_size", "max_sweeps"):
        _integer(selection[key], f"selection.{key}")
    _number(selection["energy_tolerance_j"], "selection.energy_tolerance_j", inclusive=True)
    for key, value in config["energy"].items():
        if value is not None:
            _number(value, f"energy.{key}", inclusive=True)
    if all(v is not None for v in config["energy"].values()):
        training_energy_capacity(config)
    for key, value in config["theory"].items():
        _number(value, f"theory.{key}", inclusive=key in ("V", "rho", "delta"))
    if not isinstance(config["output"]["scenario_dir"], str) or not config["output"]["scenario_dir"]:
        raise ValueError("output.scenario_dir 必须为非空路径")
    # 物理原始数据和输出目录不可重合，防止把预处理缓存写入原始包目录。
    data_root = Path(data["root"]).expanduser().resolve()
    data_output = Path(data["output"]).expanduser().resolve()
    if data_output == data_root:
        raise ValueError("数据输出目录不得与原始数据目录相同")


def resolve_collection(config, manifest):
    """实际容量只取自已准备的清单；不在环境模块中再次分配样本。"""
    validate_config(config)
    count = config["mission"]["num_uavs"]
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("需要schema=1的data.json")
    clients, settings = manifest.get("clients"), manifest.get("settings", {})
    if (not isinstance(clients, list) or len(clients) != count
            or settings.get("uavs") != count or settings.get("partition") != "iid"):
        raise ValueError("数据清单必须与无人机数量一致且采用IID划分")
    sizes, flat = [], []
    for i, client in enumerate(clients):
        if not isinstance(client, dict):
            raise ValueError("每个UAV清单条目必须为字典")
        indices = client.get("indices")
        if (client.get("uav_id") != i or not isinstance(indices, list) or not indices
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in indices)):
            raise ValueError("清单UAV编号或训练子集局部索引非法")
        if client.get("samples") != len(indices):
            raise ValueError("清单samples与indices长度不一致")
        sizes.append(len(indices))
        flat.extend(indices)
    if len(set(sizes)) != 1 or sorted(flat) != list(range(len(flat))):
        raise ValueError("需要互不重复、恰好覆盖选中训练集的等量IID清单")
    maximum = sizes[0]
    explicit = config["data"]["samples_per_uav"]
    if explicit is not None and explicit != maximum:
        raise ValueError("配置的每机样本数与已准备清单不一致")
    if "samples" in manifest:
        rows = manifest["samples"]
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or sum(row.get("split") == "train" for row in rows) != len(flat)):
            raise ValueError("清单训练样本数与UAV索引不一致")
    data = config["data"]
    initial = data["initial_samples_per_uav"]
    if initial is None:
        initial = max(1, math.floor(maximum * data["initial_fraction"]))
    if initial > maximum:
        raise ValueError("初始缓存不能超过每机样本上限")
    duration = float(config["mission"]["duration_s"])
    rate = data["sample_rate_per_s"]
    if rate is None:
        rate = (maximum - initial) / duration
    return dict(num_uavs=count, max_samples_per_uav=maximum,
                initial_samples_per_uav=initial, rate_samples_per_s=float(rate), duration_s=duration)


def available_samples_at(time_s, collection):
    """无状态采集数量查询；单位秒，不推进时钟或修改数据排列。"""
    required = {"duration_s", "max_samples_per_uav", "initial_samples_per_uav", "rate_samples_per_s"}
    if not isinstance(collection, dict) or not required.issubset(collection):
        raise ValueError("采集配置缺少必要字段")
    _number(time_s, "time_s", inclusive=True)
    duration = collection["duration_s"]
    _number(duration, "collection.duration_s")
    if time_s > duration:
        raise ValueError("查询时间超出任务范围")
    maximum = collection["max_samples_per_uav"]
    initial = collection["initial_samples_per_uav"]
    rate = collection["rate_samples_per_s"]
    _integer(maximum, "collection.max_samples_per_uav")
    _integer(initial, "collection.initial_samples_per_uav")
    _number(rate, "collection.rate_samples_per_s", inclusive=True)
    if initial > maximum:
        raise ValueError("初始缓存大于样本上限")
    if rate > 0 and time_s >= (maximum - initial) / rate:
        return maximum
    acquired = math.floor(math.nextafter(float(rate * time_s), math.inf))
    return min(maximum, initial + acquired)


def training_energy_capacity(config):
    values = config["energy"]
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError("电量核算前需填写：" + ", ".join(missing))
    capacity = (values["initial_energy_j"] - values["flight_energy_j"]
                - values["fixed_energy_j"] - values["reserve_energy_j"])
    if capacity < 0:
        raise ValueError("扣除固定飞行、基础运行及储备后训练电量为负")
    return capacity


def theory_budget(config):
    """计算正文中的条件性充分预算；不验证这些界对实际网络成立。"""
    p, t = config["theory"], config["training"]
    K, eta = t["aggregation_period"], t["epoch_step_size"]
    a = K * p["L"] * eta
    if not 0 < a < 0.5:
        raise ValueError("充分预算需要0 < K*L*eta < 1/2")
    kappa = (a / (1 - a)) ** 2
    A = 2 * p["delta"] / (eta * (1 - kappa)) + K * p["G"] ** 2
    C = (kappa * p["V"] + 2 * p["G"] * p["rho"] * (kappa + a) + p["rho"] ** 2) / (1 - kappa)
    if C >= p["epsilon"]:
        raise ValueError("误差底项不小于目标精度，不能推出有限充分预算")
    epochs = K * max(1, math.ceil(A / (K * (p["epsilon"] - C))))
    return dict(a=a, kappa=kappa, A=A, C_infinity=C, sufficient_epochs=epochs,
                configured_epochs=t["total_epochs"], configured_bound=A / t["total_epochs"] + C,
                conditional=True)


if __name__ == "__main__":
    current = load_config()
    print(json.dumps(current, ensure_ascii=False, indent=2, allow_nan=False))
    print("条件性训练预算：", theory_budget(current))
