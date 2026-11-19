"""UAV-SFL 默认参数、参数校验、配置加载与共享数据格式。"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
CLASSES = ["bicycle", "building", "car", "pedestrian", "road", "sidewalk", "sky", "store", "tree"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BLOCKS = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
          (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]

BLOCKS_PER_GROUP = 2
SPLIT_POINTS = tuple(range(0, len(BLOCKS) + 1, BLOCKS_PER_GROUP))


def validate_split_point(split: int) -> None:
    """切点沿用客户端残差块数编号，只允许整组边界。"""
    if isinstance(split, bool) or not isinstance(split, (int, np.integer)) or split not in SPLIT_POINTS:
        raise ValueError(f"split 必须是组边界 {SPLIT_POINTS} 之一，收到 {split}")


def uav_parameter(config: dict, key: str) -> np.ndarray:
    """将共享标量或逐机列表统一为长度 N 的只读数值视图。"""
    return np.broadcast_to(np.asarray(config["uav"][key], dtype=float), (config["num_uavs"],))


def default_config(num_uavs: int = 4) -> dict:
    """返回独立 Config；每次调用重新创建全部嵌套字典。

    编辑顺序建议：先设 data 的数据路径/尺寸，再设 training 的轮数/步长，
    然后设硬件、时限、电池及 optimization 策略。修改后执行 config 子命令，
    由 validate_config 检查约束；这些默认数字是场景输入，需按实验标定。
    """
    config = {
        # seed 同时确定数据划分与模型初始化；批次顺序再派生 epoch/UAV 子种子。
        # 相同种子保证输入可复现；不同资源策略仍可能因时间不同使用不同 D_k。
        "seed": 42,
        "num_uavs": 4,
        # root 放原始 Images.zip/Annotations.zip；区域图像在读取 batch 时裁剪。
        # val/test_fraction 按原始街景划分，避免同场景区域跨集合泄漏。
        # samples_per_uav=None 自动按训练池等量分配；整数则指定每机容量。
        # 首轮默认每机已有 500 个巡逻样本；initial_samples_per_uav=None 才回退
        # 到 initial_fraction。固定初始数超过真实容量时明确报错，不静默截断。
        # 此后每经过一个采样间隔就入库一批新区域。
        # sample_interval_s=None 时按规划时段和剩余批数标定固定间隔。
        # patrol_epoch_s 是任务规划用的名义轮长，独立于资源求解的硬时限。
        # 显式设置 rate_per_s 可复现旧的恒速释放及启动等待。
        # partition='dirichlet' 时 alpha 越小，通常类别偏斜越强，实际计数另存。
        # fixed_epoch_counts=None 按物理时间释放；给列表则改为共同快照消融。
        "data": {"kind": "cbcl", "root": str(ROOT.parents[1] / "DataSet" / "CBCL StreetScenes"),
                 "image_size": 64, "val_fraction": 0.1, "test_fraction": 0.1,
                 "samples_per_uav": None, "initial_samples_per_uav": 500,
                 "initial_fraction": 0.1, "rate_per_s": None,
                 "patrol_epoch_s": 600.0, "patrol_collection_fraction": 0.95,
                 "sample_interval_s": None, "samples_per_capture": 1,
                 "synthetic_per_uav": 20, "synthetic_eval_size": 12,
                 "partition": "iid", "dirichlet_alpha": 0.5,
                 "fixed_epoch_counts": None},
        # epochs 是本地 epoch 数，aggregation_period=2 表示每两轮完整聚合一次。
        # epoch_step_size 是一整轮的累计步长，batch 实际步长还要乘 |B|/D_k。
        # checkpoint=None 时正式 train 要求 --checkpoint；smoke 显式允许随机初始化。
        # device 选择本机执行设备；torch_threads 控制 CPU 并行度，不代表 UAV 算力。
        "training": {"epochs": 120, "aggregation_period": 2, "epoch_step_size": 0.01,
                     "batch_size": 32, "device": "cpu", "torch_threads": 4,
                     "checkpoint": None, "allow_random_init": False, "save_checkpoint": True},
        # empirical 使用上面的 epochs；theoretical 使用式(23)并要求已验证有效界。
        # epsilon 的含义是梯度范数平方目标上界，不能解释为分类错误率。
        "budget": {"mode": "empirical", "A_bar": 3.85, "C_bar": 0.0115,
                   "epsilon": 0.05, "effective_bound_verified": False},
        # 默认四机在同一个正方形区域的四个旋转子区执行等长闭环蛇形扫描；
        # 每个直角以固定半径圆角连接，转弯速度较低，直段按切向加速度升降速。
        # 相位按周期时间比例指定；坐标/半径单位 m，速度 m/s，加速度 m/s²。
        # circle 保留原偏心圆，旧 JSON 未指定 kind 时按旧圆字段迁移。
        # 轨迹是外部给定条件，连续资源优化不会改变飞行路线或速度。
        "trajectory": {"kind": "serpentine_equal",
                       "region_size_m": 1000.0, "center_xy_m": [0.0, 0.0],
                       "scan_rows": 6, "edge_margin_m": 10.0, "cross_margin_m": 40.0,
                       "turn_radius_m": 10.0, "turn_speed_m_s": 5.0,
                       "tangential_acceleration_m_s2": 2.0,
                       "phase_offsets_fraction": [0.0, 0.25, 0.5, 0.75],
                       "server_xyz_m": [0.0, 0.0, 0.0], "altitude_m": 100.0,
                       "radius_min_m": 250.0, "radius_max_m": 450.0,
                       "center_offset_m": 150.0, "speed_m_s": 10.0},
        # beta_l/beta_n 是线性参考功率增益，alpha_l/alpha_n 是路径损耗指数。
        # los_a/los_b 的仰角单位为度；noise_psd_w_hz 是 W/Hz，需乘分配带宽。
        "channel": {"los_a": 9.61, "los_b": 0.16, "beta_l": 1e-4, "beta_n": 1e-6,
                    "alpha_l": 2.0, "alpha_n": 2.5, "noise_psd_w_hz": 3.981e-21},
        # 计算功率 Pc=kappa*f^3；compute_power_max_w 据此决定频率上限。
        # power_max_w 是上行发射功率 W，cycles_per_flop 将 FLOPs 换成等效周期。
        # 所有 UAV 参数均可为公共标量或按 UAV 1..N 排列的列表。
        # energy_cap_j 是扣除固定消耗后的训练可用电量 J。
        "uav": {"kappa": 1e-28, "compute_power_max_w": [0.1, 0.1728, 0.3375, 0.5832],
                "power_max_w": [0.3, 0.5, 0.8, 1.0],
                "cycles_per_flop": [1.0, 0.75, 0.5, 0.375],
                "energy_cap_j": [12000.0, 16000.0, 20000.0, 24000.0]},
        # frequency_hz 为每条并行后缀分支的频率，downlink_power_w 为每条链路功率。
        # aggregation_time_s/energy_j 每次完整聚合只计一次，不乘 UAV 数量。
        "server": {"frequency_hz": 3e9, "kappa": 1e-28, "cycles_per_flop": 0.03,
                   "downlink_power_w": 2.0, "aggregation_time_s": 0.01,
                   "aggregation_energy_j": 0.1},
        # 上下行各自共享总带宽，单位 Hz；两方向预算不能互相挪用。
        "radio": {"total_up_hz": 2e7, "total_down_hz": 2e7},
        # 初始参考频率/功率相对于对应上限的比例，初始带宽按 N 等分。
        # 参考配置只提供候选评分/初值，执行前仍需校验时限和电池。
        "reference": {"frequency_fraction": 0.8, "power_fraction": 0.5},
        # strategy: fixed_fixed / fixed_opt / dynamic_opt / no_window，详见 README。
        # fixed_split 为客户端残差块数 0/2/4/6/8；动态策略从它开始寻找首周期切分，之后可改变。
        # window_epochs 只统计已完成轮次；max_iterations 为每次 P2 的迭代上限。
        # relative_tolerance 控制真实能耗收敛；feasibility_tolerance 用于原约束核验。
        # positive_floor 是比例变量的数值正下界，限制开区间边缘的病态求解。
        # 动态决策穷举所有无人机的组边界组合，按可行组合的参考总能耗排序。
        "optimization": {"strategy": "dynamic_opt", "fixed_split": 4,
                         "split_objective": "optimized_window", "split_workers": 6,
                         "reference_recovery": "rebuild",
                         "window_epochs": 5, "max_iterations": 50,
                         "relative_tolerance": 1e-5, "feasibility_tolerance": 2e-6,
                         "positive_floor": 1e-5, "solver": "CLARABEL"},
        # deadline 是单 epoch 时限；fixed_energy_j 是系统固定消耗，仅汇总时加一次。
        "task": {"epoch_deadline_s": 600.0, "fixed_energy_j": 0.0},
        "output_root": str(ROOT.parents[1] / "outputs" / "code_v3"),
    }
    if num_uavs != 4:
        # 兼容只覆盖机器数量的旧配置，其未显式设置的设备参数沿用同质默认值。
        config["num_uavs"] = num_uavs
        # 等分蛇形模板定义为四机；两机实现自检及旧任意机数场景继续使用圆轨迹。
        config["trajectory"]["kind"] = "circle"
        config["uav"] = {"kappa": 1e-28, "compute_power_max_w": 0.3375, "power_max_w": 1.0,
                         "cycles_per_flop": 0.5, "energy_cap_j": 200000.0}
    return config


def merge_config(base: dict, updates: dict) -> dict:
    """严格递归覆盖已知字段，返回新 dict；拼错字段立即报错。"""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"未知配置字段: {key}")
        if key == "optimization":
            # 允许加载旧实验配置。穷举没有初始化/坐标扫描上限或降分阈值。
            obsolete = {"max_initial_candidates", "max_coordinate_sweeps", "score_tolerance_j"}
            value = {k: v for k, v in value.items() if k not in obsolete}
        result[key] = merge_config(base[key], value) if isinstance(base[key], dict) else copy.deepcopy(value)
    return result


def training_budget(config: dict) -> dict:
    """按式(23)返回条件性轮数与实际轮数；经验预算不宣称理论保证。"""
    b, k = config["budget"], config["training"]["aggregation_period"]
    gap = b["epsilon"] - b["C_bar"]
    if gap <= 0 or b["A_bar"] < 0 or b["C_bar"] < 0:
        raise ValueError("预算要求 A_bar>=0、0<=C_bar<epsilon")
    ratio = b["A_bar"] / (k * gap)
    conditional = k * max(1, math.ceil(ratio - 1e-12 * max(1, abs(ratio))))
    if b["mode"] == "theoretical" and not b["effective_bound_verified"]:
        raise ValueError("尚未验证 epoch 级有效界，只能使用 empirical 预算")
    actual = conditional if b["mode"] == "theoretical" else config["training"]["epochs"]
    return {"epochs": actual, "aggregations": actual // k, "conditional_epochs": conditional,
            "mode": b["mode"], "effective_bound_verified": b["effective_bound_verified"]}


def require_positive(x, name, integer=False):
    """校验单个有限正数；integer=True 时同时要求整数。"""
    try:
        finite = math.isfinite(x)
    except (TypeError, OverflowError):
        finite = False
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not finite or x <= 0:
        raise ValueError(f"{name} 必须为有限正数")
    if integer and not isinstance(x, int):
        raise ValueError(f"{name} 必须为整数")


def validate_training_device(device: str) -> None:
    """校验设备字符串格式，具体设备由运行环境检查。"""
    if (not isinstance(device, str) or re.fullmatch(r"[a-z][a-z0-9_]*(?::(?:0|[1-9][0-9]*))?", device) is None
            or device.startswith("auto:")):
        raise ValueError(f"无效的 training.device {device!r}；请使用 auto、cpu、cuda 或 cuda:N")


def validate_training_config(c: dict):
    """校验机器数量、种子和训练预算，并更新副本中的实际轮数。"""
    for key in ("num_uavs",):
        require_positive(c[key], key, True)
    if isinstance(c["seed"], bool) or not isinstance(c["seed"], int) or c["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    for key in ("epochs", "batch_size", "torch_threads"):
        require_positive(c["training"][key], key, True)
    validate_training_device(c["training"].setdefault("device", "cpu"))
    if c["training"]["aggregation_period"] != 2 or c["training"]["epochs"] % 2:
        raise ValueError("本研究 K=2，epochs 必须是正偶数")
    require_positive(c["training"]["epoch_step_size"], "epoch_step_size")
    if c["budget"]["mode"] not in ("empirical", "theoretical"):
        raise ValueError("budget.mode 必须是 empirical/theoretical")
    c["training"]["epochs"] = training_budget(c)["epochs"]


def resolve_training_device(device: str) -> str:
    """在执行计算前解析本机设备，固定 CUDA 设备编号。"""
    validate_training_device(device)
    if device == "cpu":
        return device

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError("当前 Python 环境未安装 PyTorch，请使用安装了 PyTorch 的环境运行。") from exc

    if device not in ("auto", "cuda") and not device.startswith("cuda:"):
        try:
            requested = torch.device(device)
        except RuntimeError as exc:
            raise ValueError(f"当前 PyTorch 无法识别 training.device {device!r}") from exc
        return "cpu" if requested.type == "cpu" else str(requested)
    if not torch.cuda.is_available():
        if device == "auto":
            return "cpu"
        raise RuntimeError(f"请求了 {device}，但当前 PyTorch ({torch.__version__}) 无法使用 CUDA。"
                           "请检查 GPU 驱动和 PyTorch 构建，或使用 --device cpu / --device auto。")
    count = torch.cuda.device_count()
    index = int(device.removeprefix("cuda:")) if device.startswith("cuda:") else torch.cuda.current_device()
    if not 0 <= index < count:
        raise ValueError(f"请求了 {device}，但当前 PyTorch 只检测到 {count} 个 CUDA 设备。"
                         "请使用有效编号，或使用 --device auto。")
    return f"cuda:{index}"


def validate_data_config(c: dict):
    """校验数据与采样参数，并补齐兼容字段。"""
    d = c["data"]
    if d["kind"] not in ("cbcl", "synthetic") or d["partition"] not in ("iid", "dirichlet"):
        raise ValueError("data.kind=cbcl/synthetic，partition=iid/dirichlet")
    require_positive(d["image_size"], "image_size", True)
    if d["image_size"] < 32:
        raise ValueError("ResNet18 image_size 至少 32")
    for key in ("patrol_epoch_s", "patrol_collection_fraction", "dirichlet_alpha", "synthetic_per_uav", "synthetic_eval_size"):
        require_positive(d[key], key, key.startswith("synthetic"))
    if d["patrol_collection_fraction"] > 1:
        raise ValueError("patrol_collection_fraction 必须属于 (0,1]")
    if d["rate_per_s"] is not None:
        require_positive(d["rate_per_s"], "rate_per_s")
    d.setdefault("samples_per_capture", 1)
    d.setdefault("sample_interval_s", None)
    # 直接验证缺少新字段的旧完整 Config 时保留原比例语义；新默认已给 500。
    d.setdefault("initial_samples_per_uav", None)
    if d["initial_samples_per_uav"] is not None:
        require_positive(d["initial_samples_per_uav"], "initial_samples_per_uav", True)
    require_positive(d["samples_per_capture"], "samples_per_capture", True)
    if d["sample_interval_s"] is not None:
        require_positive(d["sample_interval_s"], "sample_interval_s")
    if d["rate_per_s"] is not None and (d["sample_interval_s"] is not None or d["samples_per_capture"] != 1):
        raise ValueError("旧版 rate_per_s 与巡逻采样间隔/批量不能同时指定")
    if d["samples_per_uav"] is not None:
        require_positive(d["samples_per_uav"], "samples_per_uav", True)
    require_positive(d["initial_fraction"], "initial_fraction")
    if d["initial_fraction"] > 1:
        raise ValueError("initial_fraction 必须属于 (0,1]")
    if min(d["val_fraction"], d["test_fraction"]) <= 0 or d["val_fraction"] + d["test_fraction"] >= 1:
        raise ValueError("验证/测试比例必须为正且和小于 1")


def validate_hardware_config(c: dict):
    """校验硬件、通信和任务的物理边界。"""
    for group in ("uav", "radio", "channel"):
        for key, val in c[group].items():
            if group == "uav" and isinstance(val, list):
                if len(val) != c["num_uavs"]:
                    raise ValueError(f"uav.{key} 数组长度必须为 N")
                for v in val:
                    require_positive(v, f"uav.{key}")
            else:
                require_positive(val, key)
    for key, val in c["server"].items():
        if key in ("aggregation_time_s", "aggregation_energy_j"):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{key} 必须非负")
        else:
            require_positive(val, key)
    require_positive(c["task"]["epoch_deadline_s"], "epoch_deadline_s")
    if not math.isfinite(c["task"]["fixed_energy_j"]) or c["task"]["fixed_energy_j"] < 0:
        raise ValueError("fixed_energy_j 必须非负")


def require_finite_vector(values, length: int, name: str):
    """校验有限坐标/相位向量；拒绝 bool、字符串及嵌套形状。"""
    if not isinstance(values, (list, tuple, np.ndarray)) or np.shape(values) != (length,):
        raise ValueError(f"{name} 必须为长度 {length} 的有限数值向量")
    for value in values:
        try:
            finite = math.isfinite(value)
        except (TypeError, OverflowError):
            finite = False
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not finite:
            raise ValueError(f"{name} 必须为长度 {length} 的有限数值向量")


def validate_serpentine_trajectory(t: dict, num_uavs: int):
    """校验四机等长蛇形的圆角空间和每条直段达到巡航速度所需的加减速距离。"""
    if num_uavs != 4:
        raise ValueError("serpentine_equal 轨迹要求 num_uavs=4")
    require_finite_vector(t["center_xy_m"], 2, "center_xy_m")
    require_finite_vector(t["phase_offsets_fraction"], 4, "phase_offsets_fraction")
    if any(value < 0 or value >= 1 for value in t["phase_offsets_fraction"]):
        raise ValueError("phase_offsets_fraction 必须属于 [0,1)")
    for key in ("region_size_m", "scan_rows", "turn_radius_m", "turn_speed_m_s",
                "tangential_acceleration_m_s2"):
        require_positive(t[key], key, key == "scan_rows")
    if t["scan_rows"] < 2 or t["scan_rows"] % 2:
        raise ValueError("scan_rows 必须为不小于 2 的正偶数，才能闭合蛇形回路")
    for key in ("edge_margin_m", "cross_margin_m"):
        value = t[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须为有限非负数")
    if t["turn_speed_m_s"] > t["speed_m_s"]:
        raise ValueError("turn_speed_m_s 不能大于 speed_m_s")
    half = t["region_size_m"] / 2
    spacing = (half - 2 * t["cross_margin_m"]) / (t["scan_rows"] - 1)
    radius = t["turn_radius_m"]
    straight_lengths = (half - 2 * t["edge_margin_m"] - 2 * radius,
                        spacing - 2 * radius,
                        half - 2 * t["cross_margin_m"] - 2 * radius)
    if min(straight_lengths) <= 0:
        raise ValueError("蛇形轨迹的圆角裁剪后直段必须为正；请调整区域、边距、行数或转弯半径")
    speed, turn_speed = t["speed_m_s"], t["turn_speed_m_s"]
    acceleration_distance = (0.0 if speed == turn_speed else
                             (speed - turn_speed) * (speed + turn_speed) / t["tangential_acceleration_m_s2"])
    if not math.isfinite(acceleration_distance) or min(straight_lengths) < acceleration_distance:
        raise ValueError("蛇形直段不足以按 tangential_acceleration_m_s2 加速至巡航速度再减速至转弯速度")


def validate_trajectory_config(c: dict):
    """校验轨迹种类、有限空间坐标、高度，以及对应路径的物理可行性。"""
    t = c["trajectory"]
    # 缺字段的旧完整 Config 使用旧圆轨迹，不隐式迁移成新蛇形。
    kind = t.setdefault("kind", "circle")
    if kind not in ("circle", "serpentine_equal"):
        raise ValueError("trajectory.kind 必须为 circle/serpentine_equal")
    require_finite_vector(t["server_xyz_m"], 3, "server_xyz_m")
    altitude = t["altitude_m"]
    if isinstance(altitude, bool) or not isinstance(altitude, (int, float)) or not math.isfinite(altitude):
        raise ValueError("altitude_m 必须为有限数值")
    if altitude <= t["server_xyz_m"][2]:
        raise ValueError("UAV 必须高于地面服务器")
    require_positive(t["speed_m_s"], "speed_m_s")
    if kind == "serpentine_equal":
        validate_serpentine_trajectory(t, c["num_uavs"])
    else:
        for key in ("radius_min_m", "radius_max_m"):
            require_positive(t[key], key)
        offset = t["center_offset_m"]
        if (t["radius_max_m"] < t["radius_min_m"] or isinstance(offset, bool)
                or not isinstance(offset, (int, float)) or not math.isfinite(offset)):
            raise ValueError("轨迹半径/中心偏移不合法")


def validate_optimization_config(c: dict):
    """校验优化策略、切点、容差和参考资源比例。"""
    o = c["optimization"]
    o.setdefault("split_objective", "reference_score")
    o.setdefault("split_workers", 1)
    require_positive(o["split_workers"], "split_workers", True)
    if o["split_objective"] not in ("reference_score", "optimized_window"):
        raise ValueError("split_objective 必须为 reference_score/optimized_window")
    if o["strategy"] not in ("dynamic_opt", "fixed_opt", "fixed_fixed", "no_window"):
        raise ValueError("未知 strategy")
    if o["reference_recovery"] not in ("none", "rebuild"):
        raise ValueError("reference_recovery 必须为 none/rebuild")
    validate_split_point(o["fixed_split"])
    for key in ("window_epochs", "max_iterations"):
        require_positive(o[key], key, True)
    for key in ("relative_tolerance", "feasibility_tolerance", "positive_floor"):
        require_positive(o[key], key)
    if o["positive_floor"] * c["num_uavs"] >= 1:
        raise ValueError("数值正下界过大，无法分配带宽")
    for v in c["reference"].values():
        if not 0 < v <= 1:
            raise ValueError("参考资源比例必须属于 (0,1]")


def validate_snapshot_config(c: dict):
    """校验共同样本快照的长度、正整数计数和单调性。"""
    d = c["data"]
    fixed = d["fixed_epoch_counts"]
    if fixed is not None:
        if len(fixed) != c["training"]["epochs"]:
            raise ValueError("fixed_epoch_counts 长度必须等于训练轮数")
        for v in fixed:
            require_positive(v, "fixed_epoch_counts", True)
        if any(a > b for a, b in zip(fixed, fixed[1:])):
            raise ValueError("统一样本快照必须非减")


def validate_config(c: dict) -> dict:
    """校验形状、物理边界和研究协议；返回深拷贝，不检查输入文件是否存在。"""
    c = copy.deepcopy(c)
    validate_training_config(c)
    validate_data_config(c)
    validate_hardware_config(c)
    validate_trajectory_config(c)
    validate_optimization_config(c)
    validate_snapshot_config(c)
    return c


def migrate_config_updates(updates: dict) -> dict:
    """按输入字段识别旧模式；旧圆参数、旧初始比例均保留原语义。"""
    result = copy.deepcopy(updates)
    trajectory = result.get("trajectory", {})
    old_circle_fields = ("radius_min_m", "radius_max_m", "center_offset_m")
    if "kind" not in trajectory and any(key in trajectory for key in old_circle_fields):
        trajectory["kind"] = "circle"
    data = result.get("data", {})
    if "initial_samples_per_uav" not in data and "initial_fraction" in data:
        data["initial_samples_per_uav"] = None
    return result


def load_config(path: str | None = None, updates: dict | None = None) -> dict:
    """读取可选 JSON 和内存覆盖，返回验证后的 Config；相对路径相对 JSON。"""
    source = Path(path).resolve() if path else None
    file_updates = migrate_config_updates(json.loads(source.read_text(encoding="utf-8-sig"))) if source else {}
    memory_updates = migrate_config_updates(updates or {})
    num_uavs = memory_updates.get("num_uavs", file_updates.get("num_uavs", 4))
    c = default_config(num_uavs)
    if path:
        c = merge_config(c, file_updates)
        for group, key in (("data", "root"), ("training", "checkpoint")):
            if c[group][key]:
                p = Path(c[group][key]).expanduser()
                c[group][key] = str(p if p.is_absolute() else source.parent / p)
        p = Path(c["output_root"]).expanduser()
        c["output_root"] = str(p if p.is_absolute() else source.parent / p)
    return validate_config(merge_config(c, memory_updates))


def json_value(value: Any) -> Any:
    """将数值容器转为严格 JSON 数据；Tensor 由 checkpoint 接口单独保存。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能写入 JSON: {type(value)}")


def save_json(path: str | Path, value: Any) -> None:
    """I/O：原子替换单个 JSON 文件，拒绝 NaN/Infinity。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(p)


def smoke_config(config: dict):
    """生成四机四轮合成数据诊断配置，保留原 smoke 覆盖参数。"""
    return validate_config(merge_config(config, {"num_uavs": 4,
        "data": {"kind": "synthetic", "image_size": 32, "synthetic_per_uav": 4,
                 "synthetic_eval_size": 3, "samples_per_uav": None, "initial_fraction": 0.5,
                 "initial_samples_per_uav": None,
                 "fixed_epoch_counts": None},
        "training": {"epochs": 4, "batch_size": 2, "epoch_step_size": 0.001,
                     "checkpoint": None, "allow_random_init": True, "save_checkpoint": False},
        "uav": {"kappa": 1e-28, "compute_power_max_w": 0.3375, "power_max_w": 1.0,
                "cycles_per_flop": 0.5, "energy_cap_j": 200000.0},
        "budget": {"mode": "empirical"},
        "optimization": {"max_iterations": 15, "split_objective": "reference_score"}}))
