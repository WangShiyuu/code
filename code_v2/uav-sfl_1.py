r"""动态 UAV-SFL（小函数拆分版）：参数、环境、式(25)—(31)优化、函数式 ResNet18 与训练。

运行指令（Windows PowerShell）
-------------------------------
先进入目录，并选择本机已安装依赖的解释器：
    cd E:\Work1SFL\code\code_v2
    python -m venv .venv
    $sflPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
    & $sflPython -m pip install -r .\requirements.txt

然后按用途选择一条命令；以下各命令可独立运行，不必顺序全部执行：
    # 首次运行：四机、4 epoch 合成数据，检查优化与真实反向传播流程。
    & $sflPython -X utf8 -B .\uav-sfl_1.py smoke
    # 查看参数 / 验证五切点梯度、约束和尾批步长。
    & $sflPython -X utf8 -B .\uav-sfl_1.py config
    & $sflPython -X utf8 -B .\uav-sfl_1.py check
    # 准备真实 CBCL 数据清单 / 计算五个切点的画像。
    & $sflPython -X utf8 -B .\uav-sfl_1.py prepare
    & $sflPython -X utf8 -B .\uav-sfl_1.py profile
    # 仅进行资源和能耗仿真。
    & $sflPython -X utf8 -B .\uav-sfl_1.py simulate
    # 使用本目录的 ImageNet 预训练权重进行 CBCL 联合优化与训练。
    & $sflPython -X utf8 -B .\uav-sfl_1.py train --checkpoint .\resnet18-f37072fd.pth

以上命令均不依赖示例 JSON；直接修改 default_config() 即可调整默认参数。
新电脑没有上述解释器时，先用 python -m venv .venv 创建环境，再运行
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
后续将 $sflPython 设置为 (Resolve-Path '.\.venv\Scripts\python.exe').Path。

默认四机等周期闭合蛇形轨迹，每机在物理时刻0已有500个样本。
默认 strategy='dynamic_opt'、split_objective='optimized_window'。巡逻采样按固定秒数入库；
每个切分组合按相同的预测窗口分别求资源，比较通过原约束复核的累计能耗。
reference_recovery='rebuild' 用于可选 reference_score 策略的参考资源恢复。
train 默认计划 120 epoch，实际能否完成取决于约束和求解状态；smoke 只做诊断。
结果位于本文件同目录的 output/，先查看 summary.json 中的 completed/failure，
再查看 history.csv 和 decisions.jsonl。退出码 0/1/2 分别表示成功/错误/任务未完成。

来源：同目录 notion_source.md，Notion 页面最后编辑于 2026-10-29T18:45:37Z。
入口：python uav-sfl_1.py {config,prepare,profile,simulate,train,check,smoke} --help
所有研究逻辑在本文件；不导入 code_v0/code_v1，也不定义有状态业务类。


重构阅读路线（独立文件，不依赖原 uav_sfl.py）：
  配置：validate_config -> 各分组 validate_*_config -> require_positive。
  主流程：execute_experiment -> prepare_experiment_inputs -> select_epoch_splits
          -> solve_epoch_candidates -> train_epoch_branches -> save_training_results/save_history_csv。
  命令行：main -> build_argument_parser/load_cli_config -> run_*_command。
  数据、模型、凸子问题及协议验证也按步骤拆成小函数，公开入口和计算协议保持一致。
  PowerShell 直接运行：python -X utf8 -B .\uav-sfl_1.py check

数据契约（数组第 0 维始终按 UAV 编号 0..N-1 排列）：
  Config: 嵌套 dict，唯一默认值见 default_config()，可用 JSON 局部覆盖。
  Data: {records:list[dict], clients:list[list[int]], val:list[int], test:list[int],
         release_s: ndarray[Dmax], start_s:float, classes:list[str], ...}。
  Model: {params:dict[str,Tensor], buffers:dict[str,Tensor]}；BN 统计量固定。
  Profile: {client_cycles: ndarray[N,5] 或共享 ndarray[5],
            server_cycles,activation_bits,gradient_bits,prefix_bits,
            client_flops,server_flops: ndarray[5], metadata:dict}。
  Resources: {frequency_hz,power_w,up_hz,down_hz: ndarray[N]}。
  Problem: {epoch:int,count:int,gain:ndarray[N],split:ndarray[N],next_split:array|None,
            remaining_j:ndarray[N],aggregate:bool,final:bool}。
  SolveResult: {ok:bool,status:str,resources:Resources|None,metrics:dict|None,
                trace:list[dict],initialization:list[dict],wall_s:float}。
  RunResult: {summary:dict,history:list[dict],final_model:Model|None,
              best_model:Model|None,branches:list[Model]|None,output_dir:str}。

公开主入口不修改调用方的配置/模型/资源；文件读写、打印集中在 I/O 和入口函数。
分组校验、清单组装和 checkpoint 恢复辅助函数操作主入口新建的局部字典。
单位：m、s、Hz、W、J、bit、cycle；CVXPY 内部采用频率比例、带宽比例、Mbit/s。
物理时间来自式(13)，不等于本机训练或求解的墙钟时间。
默认硬件、轨迹、预算是可运行场景假设，非设备标定结果；详见 README。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from contextlib import contextmanager
import hashlib
from itertools import product
from functools import lru_cache, partial
import io
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import time
import warnings
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath
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


# 1. 参数与数据结构 ------------------------------------------------------------
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
        "output_root": str(ROOT / "output"),
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


def validate_training_config(c: dict):
    """校验机器数量、种子和训练预算，并更新副本中的实际轮数。"""
    for key in ("num_uavs",):
        require_positive(c[key], key, True)
    if isinstance(c["seed"], bool) or not isinstance(c["seed"], int) or c["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    for key in ("epochs", "batch_size", "torch_threads"):
        require_positive(c["training"][key], key, True)
    device = c["training"]["device"]
    if (not isinstance(device, str) or device.startswith("auto:")
            or re.fullmatch(r"[a-z][a-z0-9_]*(?::(?:0|[1-9][0-9]*))?", device) is None):
        raise ValueError("training.device 必须是有效设备字符串，例如 auto、cpu、cuda 或 cuda:0")
    if c["training"]["aggregation_period"] != 2 or c["training"]["epochs"] % 2:
        raise ValueError("本研究 K=2，epochs 必须是正偶数")
    require_positive(c["training"]["epoch_step_size"], "epoch_step_size")
    if c["budget"]["mode"] not in ("empirical", "theoretical"):
        raise ValueError("budget.mode 必须是 empirical/theoretical")
    c["training"]["epochs"] = training_budget(c)["epochs"]


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


def resolve_training_device(requested: str) -> str:
    """选择本机张量设备并解析实际 CUDA 索引。"""
    if requested == "cpu":
        return requested
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError("当前 Python 环境未安装 PyTorch，请使用安装了 PyTorch 的环境运行。") from exc

    if requested not in ("auto", "cuda") and not requested.startswith("cuda:"):
        device = torch.device(requested)
        return "cpu" if device.type == "cpu" else str(device)
    if not torch.cuda.is_available():
        if requested == "auto":
            return "cpu"
        raise RuntimeError(f"请求的设备 {requested} 不可用。请用 nvidia-smi 检查显卡和驱动，"
                           "并确认当前 Python 已安装支持 CUDA 的 PyTorch。也可指定 --device auto 或 cpu。")
    count = torch.cuda.device_count()
    index = int(requested.split(":", 1)[1]) if requested.startswith("cuda:") else torch.cuda.current_device()
    if not 0 <= index < count:
        raise ValueError(f"请求的设备 {requested} 不可用。当前可见 CUDA 设备数为 {count}。")
    return f"cuda:{index}"


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


# 2. 原始数据、独立场景划分、按物理时间释放 --------------------------------------
def index_cbcl_images(images: Any) -> dict[str, str]:
    """为图片 ZIP 建立不重复的文件名到成员路径映射。"""
    members = {}
    for member in images.namelist():
        name = PurePosixPath(member.replace("\\", "/")).name
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            if name in members:
                raise ValueError(f"重复图片文件名: {name}")
            members[name] = member
    return members


def cbcl_object_record(obj: Any, oid: int, name: str, member: str,
                       width: int, height: int) -> dict | None:
    """解析一个标注对象；删除项或无效多边形返回 None，未知类别仍报错。"""
    label = (obj.findtext("name") or "").strip().lower()
    if (obj.findtext("deleted") or "0").strip() == "1" or not label:
        return None
    if label not in CLASSES:
        raise ValueError(f"未知类别 {label!r}: {name}")
    points = [(float(pt.findtext("x")), float(pt.findtext("y"))) for pt in obj.findall("polygon/pt")]
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    xs, ys = zip(*points)
    box = [max(0, math.floor(min(xs) - 1)), max(0, math.floor(min(ys) - 1)),
           min(width, math.ceil(max(xs))), min(height, math.ceil(max(ys)))]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return {"id": f"{name}:{oid}", "scene": name, "member": member,
            "bbox": box, "label": CLASSES.index(label)}


def read_cbcl_records(root: str) -> tuple[list[dict], dict]:
    """读取两份 ZIP，返回有效区域记录和审计计数；不解压、不复制样本。"""
    from PIL import Image
    root = Path(root)
    records, skipped = [], 0
    with zipfile.ZipFile(root / "Images.zip") as images, zipfile.ZipFile(root / "Annotations.zip") as annotations:
        members = index_cbcl_images(images)
        seen = set()
        for member in sorted(x for x in annotations.namelist() if x.lower().endswith(".xml")):
            tree = ET.fromstring(annotations.read(member))
            name = PurePosixPath((tree.findtext("filename") or "").strip().replace("\\", "/")).name
            if name not in members or name in seen:
                raise ValueError(f"标注图片缺失或重复: {name}")
            seen.add(name)
            with images.open(members[name]) as stream, Image.open(stream) as im:
                width, height = im.size
            for oid, obj in enumerate(tree.findall("object")):
                record = cbcl_object_record(obj, oid, name, members[name], width, height)
                if record is None:
                    skipped += 1
                else:
                    records.append(record)
    if not records:
        raise ValueError("没有有效 CBCL 区域")
    return records, {"images": len(members), "scenes": len(seen), "regions": len(records), "skipped": skipped}


def partition_clients(indices: list[int], records: list[dict], config: dict) -> tuple[list[list[int]], list[int]]:
    """返回等量且不重叠的客户端序列和未使用索引；支持 IID/容量约束 Dirichlet。"""
    rng = np.random.default_rng(config["seed"] + 1)
    n, d = config["num_uavs"], config["data"]
    size = d["samples_per_uav"] or len(indices) // n
    if size < 1 or size * n > len(indices):
        raise ValueError("真实训练区域不足以满足每机样本数，禁止复制补齐")
    order = rng.permutation(indices).tolist()
    selected, unused = order[:size * n], order[size * n:]
    if d["partition"] == "iid":
        return [selected[i * size:(i + 1) * size] for i in range(n)], unused
    # 每类的设备偏好采自 Dirichlet；逐样本在尚未填满的设备中归一化采样。
    # 等容量约束会改变最终经验分布，因此保存各机实际类别计数供报告。
    preferences = rng.dirichlet([d["dirichlet_alpha"]] * n, len(CLASSES))
    clients = [[] for _ in range(n)]
    room = np.full(n, size)
    for index in selected:
        weights = preferences[records[index]["label"]] * (room > 0)
        weights = weights / weights.sum()
        uid = int(rng.choice(n, p=weights))
        clients[uid].append(index)
        room[uid] -= 1
    return [rng.permutation(ids).tolist() for ids in clients], unused


def patrol_radii(config: dict) -> np.ndarray:
    trajectory = config["trajectory"]
    return np.linspace(trajectory["radius_min_m"], trajectory["radius_max_m"], config["num_uavs"])


def initial_sample_count(capacity: int, config: dict) -> int:
    """确定首轮每机样本数：显式固定数优先，None 回退比例；不截断超容量请求。"""
    require_positive(capacity, "capacity", True)
    d = config["data"]
    requested = d.get("initial_samples_per_uav")
    if requested is not None:
        require_positive(requested, "initial_samples_per_uav", True)
        if requested > capacity:
            raise ValueError(f"initial_samples_per_uav={requested} 超过每机真实样本容量 {capacity}")
        return requested
    fraction = d["initial_fraction"]
    require_positive(fraction, "initial_fraction")
    if fraction > 1:
        raise ValueError("initial_fraction 必须属于 (0,1]")
    return max(1, math.floor(capacity * fraction))


def sample_release_schedule(capacity: int, config: dict) -> tuple[np.ndarray, dict]:
    """返回每机共同释放表；默认时刻 0 已有固定初始数，其余按原巡逻计划采集。

    rate_per_s 显式使用旧恒速语义：全部样本从 1/rate 开始逐个到达，
    初始数只决定首轮等待到第几个样本，不把这些样本提前移至时刻 0。
    """
    d = config["data"]
    initial = initial_sample_count(capacity, config)
    if d["rate_per_s"] is not None:
        release = np.arange(1, capacity + 1, dtype=float) / d["rate_per_s"]
        return release, {"mode": "constant_rate", "initial_count": initial, "rate_per_s": d["rate_per_s"]}

    # 规划区间取首轮开始至末轮开始，预留末段用完整样本训练。
    # 不读取未来实耗时；对同一巡逻计划，各种资源策略共用相同物理时间表。
    target_s = (config["training"]["epochs"] - 1) * d["patrol_epoch_s"] * d["patrol_collection_fraction"]
    release = np.zeros(capacity)
    new_samples = capacity - initial
    batch = d["samples_per_capture"]
    captures = (new_samples + batch - 1) // batch
    interval = d["sample_interval_s"]
    if new_samples:
        if interval is None:
            interval = target_s / captures
        capture_number = np.arange(new_samples) // batch + 1
        release[initial:] = capture_number * interval
    return release, {"mode": "periodic_capture", "initial_count": initial, "target_s": target_s,
                     "sample_interval_s": interval, "samples_per_capture": batch,
                     "captures": captures,
                     "full_release_s": float(release[-1])}


def split_cbcl_scenes(records: list[dict], config: dict) -> tuple[list[int], list[int], list[int]]:
    """先打乱独立场景，再返回训练、验证、测试区域索引，避免场景泄漏。"""
    d = config["data"]
    scenes = sorted({r["scene"] for r in records})
    scenes = np.random.default_rng(config["seed"]).permutation(scenes).tolist()
    nv, nt = max(1, round(len(scenes) * d["val_fraction"])), max(1, round(len(scenes) * d["test_fraction"]))
    if nv + nt >= len(scenes):
        raise ValueError("独立场景不足，无法划分训练/验证/测试")
    val_scenes, test_scenes = set(scenes[:nv]), set(scenes[nv:nv + nt])
    val = [i for i, r in enumerate(records) if r["scene"] in val_scenes]
    test = [i for i, r in enumerate(records) if r["scene"] in test_scenes]
    train = [i for i, r in enumerate(records) if r["scene"] not in val_scenes | test_scenes]
    return train, val, test


def synthetic_data_indices(config: dict) -> tuple[list[dict], list[int], list[int], list[int], dict]:
    """生成诊断样本记录与互不重叠的训练、验证、测试索引。"""
    d, n = config["data"], config["num_uavs"]
    train_size, eval_size = n * d["synthetic_per_uav"], d["synthetic_eval_size"]
    records = [{"id": f"synthetic_{i}", "scene": f"synthetic_{i}", "label": i % len(CLASSES)}
               for i in range(train_size + 2 * eval_size)]
    train = list(range(train_size))
    val = list(range(train_size, train_size + eval_size))
    test = list(range(train_size + eval_size, len(records)))
    return records, train, val, test, {"synthetic": True, "regions": len(records)}


def assemble_data_manifest(config: dict, records: list[dict], clients: list[list[int]],
                           val: list[int], test: list[int], unused: list[int], audit: dict,
                           release: np.ndarray, plan: dict) -> dict:
    """组装清单及来源审计，并计算覆盖分组、顺序、释放表和预处理的指纹。"""
    d = config["data"]
    result = {"kind": d["kind"], "records": records, "clients": clients, "val": val, "test": test,
              "unused_train": unused, "release_s": release, "start_s": float(release[plan["initial_count"] - 1]),
              "release_plan": plan,
              "capacity": len(clients[0]), "classes": CLASSES.copy(), "audit": audit,
              "class_counts": [np.bincount([records[i]["label"] for i in ids], minlength=len(CLASSES)) for ids in clients],
              "image_size": d["image_size"], "root": d["root"], "seed": config["seed"]}
    if d["kind"] == "cbcl":
        result["source_files"] = {name: {"bytes": (Path(d["root"]) / name).stat().st_size,
                                        "mtime_ns": (Path(d["root"]) / name).stat().st_mtime_ns}
                                  for name in ("Images.zip", "Annotations.zip")}
    result["fingerprint"] = hashlib.sha256(json.dumps(result, default=json_value, sort_keys=True).encode()).hexdigest()
    return result


def prepare_data(config: dict) -> dict:
    """I/O：准备 Data，按独立场景划分，再分配客户端并建立物理释放表。

    records 是区域记录表，clients[n]、val、test 保存对该表的整数索引。
    clients 的固定顺序表示采集顺序，训练只打乱快照副本；所有 UAV 共用
    release_s。这里只读取 ZIP/生成清单，不训练、不写裁剪缓存。
    """
    d = config["data"]
    if d["kind"] == "cbcl":
        records, audit = read_cbcl_records(d["root"])
        train, val, test = split_cbcl_scenes(records, config)
    else:
        records, train, val, test, audit = synthetic_data_indices(config)
    clients, unused = partition_clients(train, records, config)
    capacity = len(clients[0])
    release, plan = sample_release_schedule(capacity, config)
    if d["fixed_epoch_counts"] is not None and max(d["fixed_epoch_counts"]) > capacity:
        raise ValueError("统一快照超过真实训练容量")
    return assemble_data_manifest(config, records, clients, val, test, unused, audit, release, plan)


def epoch_snapshot(data: dict, physical_s: float, epoch: int, config: dict) -> dict:
    """纯函数：返回轮初固定样本数/各机索引；epoch 从 1 开始编号。

    physical_s 是仿真物理时刻。searchsorted(..., side='right') 包含恰在轮初
    到达的样本；返回的 indices 是新列表，所以之后释放新样本不会改变本轮。
    fixed_epoch_counts 仅供共同快照消融，会显式替代物理释放计数。
    """
    fixed = config["data"]["fixed_epoch_counts"]
    count = fixed[epoch - 1] if fixed is not None else int(np.searchsorted(data["release_s"], physical_s, side="right"))
    if not 1 <= count <= data["capacity"]:
        raise ValueError("当前释放表未满足启动条件或统一快照越界")
    return {"count": count, "indices": [ids[:count] for ids in data["clients"]],
            "time_s": physical_s, "mode": "matched_snapshot_ablation" if fixed else "physical_release"}


def load_cbcl_samples(data: dict, indices: list[int]) -> list[np.ndarray]:
    """从图片 ZIP 读取并裁剪一批 RGB 区域，保持原有缩放与像素精度。"""
    from PIL import Image
    size, samples = data["image_size"], []
    with zipfile.ZipFile(Path(data["root"]) / "Images.zip") as z:
        for i in indices:
            r = data["records"][i]
            with Image.open(io.BytesIO(z.read(r["member"]))) as im:
                crop = im.convert("RGB").crop(r["bbox"]).resize((size, size), Image.Resampling.BILINEAR)
                samples.append(np.asarray(crop, dtype=np.float32) / 255.0)
    return samples


def synthetic_samples(data: dict, indices: list[int], labels: list[int]) -> list[np.ndarray]:
    """以样本编号派生局部随机数，生成可学习且可复现的诊断图像。"""
    size, samples = data["image_size"], []
    for i, label in zip(indices, labels):
        rng = np.random.default_rng(np.random.SeedSequence([data["seed"], i]))
        pixels = rng.uniform(0, 0.2, (size, size, 3)).astype(np.float32)
        pixels[:, :, label % 3] += 0.6  # 仅供诊断的可学习合成模式。
        samples.append(pixels)
    return samples


def batch_tensors(samples: list[np.ndarray], labels: list[int], device: str) -> tuple[Any, Any]:
    """执行统一标准化和 NCHW 转换，返回设备上的图像与 int64 标签。"""
    import torch
    array = (np.stack(samples) - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return (torch.tensor(array.transpose(0, 3, 1, 2), device=device),
            torch.tensor(labels, dtype=torch.long, device=device))


def load_batch(data: dict, indices: list[int], device: str) -> tuple[Any, Any]:
    """I/O：读取指定索引，返回 float32 [B,3,H,W]、int64 [B]；无随机增强。"""
    labels = [data["records"][i]["label"] for i in indices]
    samples = load_cbcl_samples(data, indices) if data["kind"] == "cbcl" else synthetic_samples(data, indices, labels)
    return batch_tensors(samples, labels, device)


# 3. 函数式 ResNet18、画像、SGD -------------------------------------------------
def block_name(index: int) -> str:
    """0..7 残差块索引 -> torchvision 兼容参数前缀。"""
    return f"layer{index // 2 + 1}.{index % 2}"


def initialize_resnet_tensors(seed: int) -> tuple[dict, dict]:
    """按标准 ResNet18 键名和固定随机调用顺序构造 CPU 参数及 BN buffers。"""
    import torch
    params, buffers = {}, {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def conv(name, cin, cout, kernel):
        params[name + ".weight"] = torch.randn(cout, cin, kernel, kernel, generator=generator) * math.sqrt(2 / (cout * kernel * kernel))
    def bn(name, width):
        params[name + ".weight"], params[name + ".bias"] = torch.ones(width), torch.zeros(width)
        buffers[name + ".running_mean"], buffers[name + ".running_var"] = torch.zeros(width), torch.ones(width)
        buffers[name + ".num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    conv("conv1", 3, 64, 7)
    bn("bn1", 64)
    for i, (cin, cout, stride) in enumerate(BLOCKS):
        prefix = block_name(i)
        conv(prefix + ".conv1", cin, cout, 3)
        bn(prefix + ".bn1", cout)
        conv(prefix + ".conv2", cout, cout, 3)
        bn(prefix + ".bn2", cout)
        if stride == 2:
            conv(prefix + ".downsample.0", cin, cout, 1)
            bn(prefix + ".downsample.1", cout)
    width = len(CLASSES)
    params["fc.weight"] = (torch.rand(width, 512, generator=generator) * 2 - 1) / math.sqrt(512)
    params["fc.bias"] = (torch.rand(width, generator=generator) * 2 - 1) / math.sqrt(512)
    return params, buffers


def restore_model_checkpoint(params: dict, buffers: dict, path: str) -> dict:
    """恢复新建张量，核验键/形状/有限值，并返回权重来源；只修改新建字典。"""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "params" in checkpoint and "buffers" in checkpoint:
        state = {**checkpoint["params"], **checkpoint["buffers"]}
    else:
        state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): val for key, val in state.items()}
    expected = {**params, **buffers}
    # 官方早期 ResNet18 权重没有 BN num_batches_tracked。遵循 PyTorch
    # BatchNorm 的旧权重兼容规则，仅为这些计数项补零；固定 BN 不使用计数。
    # 权重、running_mean/running_var 等其他缺失项仍按错误处理。
    initialized_counters = [name for name in buffers
                            if name.endswith(".num_batches_tracked") and name not in state]
    state = {**state, **{name: buffers[name].clone() for name in initialized_counters}}
    missing = [name for name in expected if name not in state]
    unexpected = set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(f"checkpoint 键不匹配: missing={missing[:5]}, unexpected={sorted(unexpected)[:5]}")
    head_reset = False
    for name, original in expected.items():
        incoming = state[name]
        if incoming.shape != original.shape:
            if name.startswith("fc."):
                head_reset = True
                continue
            raise ValueError(f"checkpoint 形状不匹配: {name}")
        if not torch.isfinite(incoming).all():
            raise ValueError(f"checkpoint 非有限参数: {name}")
        (params if name in params else buffers)[name] = incoming.to(original.dtype).clone()
    provenance = {"kind": "local_checkpoint", "path": str(Path(path).resolve()),
                  "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "head_reinitialized": head_reset,
                  "legacy_bn_counters_initialized": initialized_counters}
    return provenance


def initialize_model(config: dict, require_pretrained: bool = False) -> tuple[dict, dict]:
    """返回 (Model, provenance)，构造可按残差块边界执行的函数式 ResNet18。

    params 保存可训练叶子 Tensor，buffers 保存固定 BN 统计量；标准键名
    兼容官方 state_dict。先构造局部随机初值，再按需恢复 checkpoint，最后
    移至设备。正式训练缺少 checkpoint 且未允许随机初始化时仍报错。
    """
    device = resolve_training_device(str(config["training"]["device"]))
    params, buffers = initialize_resnet_tensors(config["seed"])
    provenance = {"kind": "random_diagnostic", "seed": config["seed"], "head_reinitialized": True}
    path = config["training"]["checkpoint"]
    if path:
        provenance = restore_model_checkpoint(params, buffers, path)
    elif require_pretrained and not config["training"]["allow_random_init"]:
        raise ValueError("正式训练需要 training.checkpoint（标准 ResNet18 state_dict 或本程序模型）；诊断可显式 --allow-random-init")
    return {"params": {k: v.to(device).detach().requires_grad_(True) for k, v in params.items()},
            "buffers": {k: v.to(device) for k, v in buffers.items()}}, provenance


def frozen_bn(x: Any, model: dict, name: str) -> Any:
    """固定统计量的 BatchNorm；仿射参数仍求梯度，不改变 buffers。"""
    from torch.nn import functional as F
    p, b = model["params"], model["buffers"]
    return F.batch_norm(x, b[name + ".running_mean"], b[name + ".running_var"],
                        p[name + ".weight"], p[name + ".bias"], training=False, eps=1e-5)


def residual_block(x: Any, model: dict, index: int) -> Any:
    """一个完整残差块；输入/输出 Tensor，不修改 Model。"""
    from torch.nn import functional as F
    name, p = block_name(index), model["params"]
    stride = BLOCKS[index][2]
    identity = x
    if stride == 2:
        identity = frozen_bn(F.conv2d(x, p[name + ".downsample.0.weight"], stride=2), model, name + ".downsample.1")
    y = F.relu(frozen_bn(F.conv2d(x, p[name + ".conv1.weight"], stride=stride, padding=1), model, name + ".bn1"))
    y = frozen_bn(F.conv2d(y, p[name + ".conv2.weight"], padding=1), model, name + ".bn2")
    return F.relu(y + identity)


def validate_split_point(split: int) -> None:
    """切点沿用客户端残差块数编号，只允许整组边界。"""
    if isinstance(split, bool) or not isinstance(split, (int, np.integer)) or split not in SPLIT_POINTS:
        raise ValueError(f"split 必须是组边界 {SPLIT_POINTS} 之一，收到 {split}")


def forward_prefix(model: dict, images: Any, split: int) -> Any:
    """执行 stem + 前 split 个完整残差块，返回有计算图的激活 Tensor。"""
    from torch.nn import functional as F
    validate_split_point(split)
    x = F.conv2d(images, model["params"]["conv1.weight"], stride=2, padding=3)
    x = F.max_pool2d(F.relu(frozen_bn(x, model, "bn1")), 3, 2, 1)
    for i in range(split):
        x = residual_block(x, model, i)
    return x


def forward_suffix(model: dict, activation: Any, split: int) -> Any:
    """执行剩余残差块 + 池化/分类头，返回 logits [B,9]。"""
    from torch.nn import functional as F
    validate_split_point(split)
    x = activation
    for i in range(split, len(BLOCKS)):
        x = residual_block(x, model, i)
    x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    return F.linear(x, model["params"]["fc.weight"], model["params"]["fc.bias"])


def prefix_keys(model: dict, split: int) -> list[str]:
    """返回客户端可训练参数键；BN 固定 buffers 已预部署，不计重复通信。"""
    validate_split_point(split)
    prefixes = ["conv1.", "bn1."] + [block_name(i) + "." for i in range(split)]
    return [name for name in model["params"] if any(name.startswith(p) for p in prefixes)]


def split_gradients(model: dict, images: Any, labels: Any, split: int) -> tuple[dict, dict]:
    """显式模拟分割学习，返回 (全模型梯度字典, 批次统计)，不更新参数。

    images 为 float32 [B,3,H,W]，labels 为 int64 [B]，split 为 0/2/4/6/8。
    梯度字典键与 model['params'] 相同；loss 是批次平均交叉熵，correct 是
    正确样本数，通信量按本批实际 Tensor 大小计 bit，上传还包含标签。
    autograd.grad 直接返回梯度，不向参数 .grad 累积，函数间没有残留梯度状态。
    """
    import torch
    from torch.nn import functional as F
    left = prefix_keys(model, split)
    right = [name for name in model["params"] if name not in set(left)]
    boundary = forward_prefix(model, images, split)
    # boundary 保留客户端计算图；received 是服务器收到的独立叶子激活。
    # detach 阻断跨端自动求导，后面必须显式把 dLoss/dActivation 传回客户端。
    received = boundary.detach().clone().requires_grad_(True)
    logits = forward_suffix(model, received, split)
    loss = F.cross_entropy(logits, labels)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("批次损失非有限，停止本轮训练")
    # 先取得服务器后缀参数梯度及激活梯度，再用激活梯度求客户端前缀梯度。
    # 此时两端参数都没有更新，保持与完整模型同一更新前状态的梯度关系。
    server_grad = torch.autograd.grad(loss, [model["params"][name] for name in right] + [received])
    client_grad = torch.autograd.grad(boundary, [model["params"][name] for name in left], grad_outputs=server_grad[-1].detach())
    grads = dict(zip(left, client_grad)) | dict(zip(right, server_grad[:-1]))
    return grads, {"loss": float(loss.detach()), "correct": int((logits.argmax(1) == labels).sum()),
                   "samples": len(labels), "activation_bits": boundary.numel() * boundary.element_size() * 8 + labels.numel() * labels.element_size() * 8,
                   "gradient_bits": server_grad[-1].numel() * server_grad[-1].element_size() * 8}


def sgd_step(model: dict, gradients: dict, step: float) -> dict:
    """式(19)：返回新 Model；无 momentum/weight decay，原参数及 buffers 不被写入。"""
    import torch
    with torch.no_grad():
        params = {k: (v - step * gradients[k]).detach().requires_grad_(True) for k, v in model["params"].items()}
    if not all(bool(torch.isfinite(v).all()) for v in params.values()):
        raise FloatingPointError("SGD 产生非有限参数，停止本次任务")
    return {"params": params, "buffers": model["buffers"]}


def train_client_batch(model: dict, data: dict, batch: list[int], split: int,
                       epoch_count: int, config: dict) -> tuple[dict, dict, float]:
    """读取一个 batch，计算分割梯度，并按 batch 在冻结快照中的占比执行 SGD。"""
    x, y = load_batch(data, batch, str(model["params"]["conv1.weight"].device))
    gradients, metrics = split_gradients(model, x, y, split)
    # 每个样本对本轮累计步长的权重相同；例如 D=3、batch_size=2 时，
    # 两批使用 2*eta/3 和 eta/3，数据增长不会自动扩大整轮累计步长。
    step = config["training"]["epoch_step_size"] * len(batch) / epoch_count
    state = sgd_step(model, gradients, step)
    return state, metrics, step


def train_client_epoch(model: dict, data: dict, indices: list[int], split: int,
                       epoch: int, uid: int, config: dict) -> tuple[dict, dict]:
    """遍历一次冻结快照，返回 (新 Model, 本轮统计)，尾批按真实大小计算。

    indices 必须是当前 UAV 的轮初快照，函数不会查询中途新释放的样本。
    epoch 为 1 起始轮号，uid 为 0 起始设备号，两者决定独立可复现的 batch 顺序。
    loss/accuracy 按样本数加权；step_sum 应等于 epoch_step_size；通信量是
    本轮激活/标签和梯度的累计 bit，不包含由系统聚合阶段单独计量的前缀参数。
    """
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, uid]))
    order = rng.permutation(indices).tolist()
    state, loss_sum, correct, step_sum, up, down = model, 0.0, 0, 0.0, 0, 0
    batch_size = config["training"]["batch_size"]
    for offset in range(0, len(order), batch_size):
        batch = order[offset:offset + batch_size]
        state, metrics, step = train_client_batch(state, data, batch, split, len(order), config)
        loss_sum += metrics["loss"] * len(batch)
        correct += metrics["correct"]
        step_sum += step
        up += metrics["activation_bits"]
        down += metrics["gradient_bits"]
    return state, {"loss": loss_sum / len(order), "accuracy": correct / len(order),
                   "samples": len(order), "step_sum": step_sum, "activation_bits": up, "gradient_bits": down}


def aggregate_models(models: list[dict], counts: list[int]) -> dict:
    """式(5)：按 counts 加权平均完整模型，返回新 Model，不修改各分支。

    不同切点只是同一完整参数字典的执行分界，所以可以按同名参数直接平均。
    本研究各机样本数相同，权重退化为 1/N；保留 counts 接口便于核验通用情况。
    BN 统计量冻结且应完全一致，检查通过后共享 buffers，不对计数做浮点平均。
    """
    import torch
    if len(models) != len(counts) or not models or min(counts) <= 0:
        raise ValueError("聚合模型数/权重不合法")
    weights = np.asarray(counts, dtype=float) / sum(counts)
    for m in models[1:]:
        if m["params"].keys() != models[0]["params"].keys():
            raise ValueError("聚合模型结构不一致")
        if any(not torch.equal(v, m["buffers"][k]) for k, v in models[0]["buffers"].items()):
            raise ValueError("BN 固定统计量不一致")
    with torch.no_grad():
        params = {key: sum(float(w) * m["params"][key] for w, m in zip(weights, models)).detach().requires_grad_(True)
                  for key in models[0]["params"]}
    return {"params": params, "buffers": models[0]["buffers"]}


def evaluate_model(model: dict, data: dict, indices: list[int], config: dict) -> dict:
    """I/O + 推理：返回 loss/accuracy/macro_f1/confusion；测试集只在任务末调用。"""
    import torch
    from torch.nn import functional as F
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    loss_sum = 0.0
    device = str(model["params"]["conv1.weight"].device)
    with torch.no_grad():
        for offset in range(0, len(indices), config["training"]["batch_size"]):
            ids = indices[offset:offset + config["training"]["batch_size"]]
            x, y = load_batch(data, ids, device)
            logits = forward_suffix(model, forward_prefix(model, x, 0), 0)
            batch_loss = F.cross_entropy(logits, y, reduction="sum")
            if not bool(torch.isfinite(batch_loss)):
                raise FloatingPointError("评价损失非有限，模型不可用于评价")
            loss_sum += float(batch_loss)
            for truth, pred in zip(y.cpu().tolist(), logits.argmax(1).cpu().tolist()):
                confusion[truth, pred] += 1
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * np.diag(confusion), denominator, out=np.zeros(len(CLASSES)), where=denominator > 0)
    return {"loss": loss_sum / len(indices), "accuracy": float(np.trace(confusion) / len(indices)),
            "macro_f1": float(f1.mean()), "confusion": confusion, "samples": len(indices)}


def profile_split_point(model: dict, config: dict, x: Any, y: Any, split: int) -> dict:
    """计量一个切点的客户端/服务器前后向 FLOPs 以及激活、梯度、前缀位数。"""
    import torch
    from torch.nn import functional as F
    from torch.utils.flop_counter import FlopCounterMode
    left = prefix_keys(model, split)
    right = [key for key in model["params"] if key not in set(left)]
    with FlopCounterMode(display=False) as counter:
        boundary = forward_prefix(model, x, split)
    client_flops = counter.get_total_flops()
    received = boundary.detach().clone().requires_grad_(True)
    with FlopCounterMode(display=False) as counter:
        logits = forward_suffix(model, received, split)
        loss = F.cross_entropy(logits, y)
        gs = torch.autograd.grad(loss, [model["params"][key] for key in right] + [received])
    server_flops = counter.get_total_flops()
    with FlopCounterMode(display=False) as counter:
        torch.autograd.grad(boundary, [model["params"][key] for key in left], grad_outputs=gs[-1])
    client_flops += counter.get_total_flops()
    return {"client_flops": client_flops, "server_flops": server_flops,
                 "server_cycles": server_flops * config["server"]["cycles_per_flop"],
                 "activation_bits": boundary.numel() * boundary.element_size() * 8 + 64,
                 "gradient_bits": boundary.numel() * boundary.element_size() * 8,
                 "prefix_bits": sum(model["params"][key].numel() * model["params"][key].element_size() * 8 for key in left)}


def assemble_split_profile(rows: list[dict], config: dict) -> dict:
    """将各切点计量行转为数组，应用硬件周期换算并标记计量边界。"""
    profile = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
    conversion = np.asarray(config["uav"]["cycles_per_flop"], dtype=float)
    profile["client_cycles"] = (profile["client_flops"] * conversion if conversion.ndim == 0
                                else conversion[:, None] * profile["client_flops"])
    profile["metadata"] = {"split_points": list(SPLIT_POINTS),
                           "uav_cycles_per_flop": uav_parameter(config, "cycles_per_flop"),
                           "method": "supported_operator_FLOPs_times_configured_cycles_per_FLOP",
                           "hardware_measured": False, "image_size": config["data"]["image_size"], "dtype": "float32",
                           "labels": "int64", "prefix_payload": "trainable_parameters_only_frozen_BN_predeployed",
                           "exclusions": "unsupported operators, optimizer updates, memory/I/O/runtime overhead"}
    return profile


def profile_splits(model: dict, config: dict) -> dict:
    """统计五切点前后向 FLOPs 与真实 bit，乘显式硬件系数得到等效周期。

    仅覆盖 FlopCounterMode 支持算子，不含优化器、数据读取、内存搬运及
    未支持算子；结果并非硬件实测，metadata 始终记录此限制。
    """
    import torch
    size, device = config["data"]["image_size"], model["params"]["conv1.weight"].device
    x = torch.linspace(-1, 1, 3 * size * size, device=device).reshape(1, 3, size, size)
    y = torch.zeros(1, dtype=torch.long, device=device)
    rows = [profile_split_point(model, config, x, y, split) for split in SPLIT_POINTS]
    return assemble_split_profile(rows, config)


def grouped_profile(profile: dict) -> dict:
    """统一实测/旧版画像到五个组边界，并保留各列对应的残差块数。"""
    metadata = profile.get("metadata", {})
    points = metadata.get("split_points")
    if points is None:
        # 旧版九列画像按每个块后的位置排列，新版五列按组排列。
        points = range(len(BLOCKS) + 1) if np.shape(profile["client_cycles"])[-1:] == (9,) else SPLIT_POINTS
    points = tuple(points)
    if points not in (SPLIT_POINTS, tuple(range(len(BLOCKS) + 1))):
        raise ValueError("画像的切点顺序与 ResNet18 组边界不一致")
    columns = [points.index(cut) for cut in SPLIT_POINTS]
    result = dict(profile)
    fields = ("client_cycles", "server_cycles", "activation_bits", "gradient_bits", "prefix_bits")
    for key in fields + tuple(k for k in ("client_flops", "server_flops") if k in profile):
        values = np.asarray(profile[key], dtype=float)
        valid_shape = values.shape == (len(points),)
        if key == "client_cycles":
            valid_shape = values.ndim in (1, 2) and values.shape[-1] == len(points)
        if not valid_shape or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"画像 {key} 的长度必须与切点数一致且数值为有限正数")
        result[key] = values[..., columns]
    result["metadata"] = {**metadata, "split_points": list(SPLIT_POINTS)}
    return result


# 4. 给定轨迹、平均大尺度信道与原始成本 ------------------------------------------
def _patrol_line(start: np.ndarray, end: np.ndarray, speed: float,
                 turn_speed: float, acceleration: float) -> dict:
    """构造两端为转弯速度、可升至巡航速度的直线段及解析速度剖面。"""
    length = float(np.linalg.norm(end - start))
    ramp_distance = (speed ** 2 - turn_speed ** 2) / (2 * acceleration)
    if length <= 0 or length + 1e-10 < 2 * ramp_distance:
        raise ValueError("圆角后的每段直线必须为正且足以完成加速与减速")
    ramp_time = (speed - turn_speed) / acceleration
    cruise_distance = max(0.0, length - 2 * ramp_distance)
    cruise_time = cruise_distance / speed
    return {"kind": "line", "start_xy_m": start.tolist(), "end_xy_m": end.tolist(),
            "length_m": length, "duration_s": 2 * ramp_time + cruise_time,
            "direction_xy": ((end - start) / length).tolist(),
            "start_speed_m_s": turn_speed, "peak_speed_m_s": speed,
            "acceleration_m_s2": acceleration, "accel_distance_m": ramp_distance,
            "cruise_distance_m": cruise_distance, "accel_time_s": ramp_time,
            "cruise_time_s": cruise_time}

@lru_cache(maxsize=32)
def _serpentine_template(region_size: float, rows: int, edge: float, cross: float,
                          radius: float, speed: float, turn_speed: float,
                          acceleration: float) -> dict:
    """按轨迹几何数值缓存 NE 模板；缓存仅内部只读，公开接口返回独立副本。"""
    half = region_size / 2
    ys = np.linspace(cross, half - cross, rows)
    vertices = [np.array([edge, ys[0]])]
    for index, y in enumerate(ys):
        x = half - edge if index % 2 == 0 else edge
        vertices.append(np.array([x, y]))
        if index + 1 < rows:
            vertices.append(np.array([x, ys[index + 1]]))
    entries, exits, centers, signs = [], [], [], []
    raw_length = 0.0
    for index, vertex in enumerate(vertices):
        previous, following = vertices[index - 1], vertices[(index + 1) % len(vertices)]
        incoming, outgoing = vertex - previous, following - vertex
        raw_length += float(np.linalg.norm(outgoing))
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        if not np.isclose(np.dot(incoming, outgoing), 0.0, atol=1e-12):
            raise ValueError("蛇形闭合路线的相邻边必须垂直")
        entries.append(vertex - radius * incoming)
        exits.append(vertex + radius * outgoing)
        centers.append(entries[-1] + radius * outgoing)
        signs.append(float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]))
    segments, elapsed = [], 0.0
    # 周期起点为第一个圆角的出切点，默认 (20,40)，不是已切除的 (10,40)。
    for index in range(len(vertices)):
        corner = (index + 1) % len(vertices)
        line = _patrol_line(exits[index], entries[corner], speed, turn_speed, acceleration)
        arc = {"kind": "arc", "start_xy_m": entries[corner].tolist(),
               "end_xy_m": exits[corner].tolist(), "center_xy_m": centers[corner].tolist(),
               "radius_m": radius, "length_m": np.pi * radius / 2,
               "duration_s": np.pi * radius / (2 * turn_speed),
               "start_angle_rad": float(np.arctan2(*(entries[corner] - centers[corner])[::-1])),
               "sweep_angle_rad": signs[corner] * np.pi / 2, "speed_m_s": turn_speed}
        for segment in (line, arc):
            segment["start_time_s"] = elapsed
            elapsed += segment["duration_s"]
            segment["end_time_s"] = elapsed
            segments.append(segment)
    return {"kind": "serpentine_equal", "segments": segments,
            "vertices_xy_m": [v.tolist() for v in vertices],
            "start_xy_m": exits[0].tolist(), "raw_length_m": raw_length,
            "length_m": sum(s["length_m"] for s in segments), "period_s": elapsed,
            "turn_count": len(vertices)}

def _patrol_template(config: dict) -> dict:
    """读取几何值组成缓存键；位置、服务器、时间和其他配置不会污染缓存。"""
    t = config["trajectory"]
    rows = t.get("scan_rows", 6)
    if config["num_uavs"] != 4 or isinstance(rows, bool) or not isinstance(rows, (int, np.integer)) or rows < 2 or rows % 2:
        raise ValueError("serpentine_equal 要求四架 UAV 且 scan_rows 为至少 2 的偶数")
    values = [float(t.get(key, default)) for key, default in (
        ("region_size_m", 1000.0), ("edge_margin_m", 10.0), ("cross_margin_m", 40.0),
        ("turn_radius_m", 10.0), ("speed_m_s", 10.0), ("turn_speed_m_s", 5.0),
        ("tangential_acceleration_m_s2", 2.0))]
    size, edge, cross, radius, speed, turn_speed, acceleration = values
    if not np.isfinite(values).all() or min(size, radius, speed, turn_speed, acceleration) <= 0 or min(edge, cross) < 0:
        raise ValueError("蛇形区域、圆角、速度和加速度必须有限为正，边距必须有限非负")
    if turn_speed > speed:
        raise ValueError("turn_speed_m_s 不能大于 speed_m_s")
    straight_lengths = [size / 2 - 2 * edge - 2 * radius,
                        (size / 2 - 2 * cross) / (rows - 1) - 2 * radius,
                        size / 2 - 2 * cross - 2 * radius]
    if min(straight_lengths) <= 0:
        raise ValueError("扫描区域及行距不足以容纳圆角")
    return _serpentine_template(size, int(rows), edge, cross, radius, speed, turn_speed, acceleration)

def _patrol_frame(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回与服务器坐标独立的区域中心，以及四机的周期时间相位比例。"""
    t = config["trajectory"]
    center = np.asarray(t.get("center_xy_m", [0.0, 0.0]), dtype=float)
    phases = np.asarray(t.get("phase_offsets_fraction", [0.0, 0.25, 0.5, 0.75]), dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy_m 必须是有限二维坐标")
    if phases.shape != (4,) or not np.isfinite(phases).all() or np.any((phases < 0) | (phases >= 1)):
        raise ValueError("phase_offsets_fraction 必须为四个 [0,1) 内的有限数")
    return center, phases

def patrol_geometry(config: dict) -> dict:
    """返回独立轨迹几何；蛇形含模板线/弧段和共同周期，圆形含各机周长/周期。

    蛇形 segments 为相对区域中心的 NE 模板，周期从默认 (20,40) 向右出发。
    四机分别旋转 0/90/180/270 度，并按 phase_offsets_s 做时间错相。
    长度/覆盖属于给定轨迹模型；圆形分支保留旧版给定轨迹的数值行为。
    """
    kind = config["trajectory"].get("kind", "circle")
    if kind == "circle":
        radii = patrol_radii(config)
        lengths = 2 * np.pi * radii
        return {"kind": kind, "segments": [], "radii_m": radii,
                "length_m": lengths, "period_s": lengths / config["trajectory"]["speed_m_s"]}
    if kind != "serpentine_equal":
        raise ValueError(f"未知轨迹 kind: {kind}")
    result = copy.deepcopy(_patrol_template(config))
    center, phases = _patrol_frame(config)
    result.update(center_xy_m=center.tolist(), rotation_rad=(np.arange(4) * np.pi / 2).tolist(),
                  phase_offsets_s=(phases * result["period_s"]).tolist())
    return result

def _patrol_segment_position(segment: dict, time_s: float) -> np.ndarray:
    """按线段的加减速或圆弧的匀速，解析求出段内位置。"""
    elapsed = min(max(time_s, 0.0), segment["duration_s"])
    if segment["kind"] == "arc":
        angle = segment["start_angle_rad"] + segment["sweep_angle_rad"] * elapsed / segment["duration_s"]
        return np.asarray(segment["center_xy_m"]) + segment["radius_m"] * np.array([np.cos(angle), np.sin(angle)])
    v0, acceleration = segment["start_speed_m_s"], segment["acceleration_m_s2"]
    if elapsed < segment["accel_time_s"]:
        distance = v0 * elapsed + acceleration * elapsed ** 2 / 2
    elif elapsed < segment["accel_time_s"] + segment["cruise_time_s"]:
        distance = segment["accel_distance_m"] + segment["peak_speed_m_s"] * (elapsed - segment["accel_time_s"])
    else:
        remaining = segment["duration_s"] - elapsed
        distance = segment["length_m"] - v0 * remaining - acceleration * remaining ** 2 / 2
    return np.asarray(segment["start_xy_m"]) + distance * np.asarray(segment["direction_xy"])

def patrol_positions(physical_s: float, config: dict) -> np.ndarray:
    """返回标量物理时刻的 XYZ[N,3]；周期首尾及所有线/弧接点位置、速度连续。"""
    physical_s = float(physical_s)
    if not np.isfinite(physical_s):
        raise ValueError("physical_s 必须有限")
    n, t = config["num_uavs"], config["trajectory"]
    kind = t.get("kind", "circle")
    if kind == "circle":
        radius = patrol_radii(config)
        phase = 2 * np.pi * np.arange(n) / n
        angle = phase + t["speed_m_s"] * physical_s / radius
        return np.column_stack((radius * np.cos(angle) + t["center_offset_m"] * np.cos(phase),
                                radius * np.sin(angle) + t["center_offset_m"] * np.sin(phase),
                                np.full(n, t["altitude_m"])))
    if kind != "serpentine_equal":
        raise ValueError(f"未知轨迹 kind: {kind}")
    template = _patrol_template(config)
    center, phases = _patrol_frame(config)
    segments, period = template["segments"], template["period_s"]
    end_times = [segment["end_time_s"] for segment in segments]
    xyz = np.empty((4, 3), dtype=float)
    for uid, fraction in enumerate(phases):
        local_time = (physical_s + fraction * period) % period
        index = min(int(np.searchsorted(end_times, local_time, side="right")), len(segments) - 1)
        xy = _patrol_segment_position(segments[index], local_time - segments[index]["start_time_s"])
        x, y = xy
        rotated = ((x, y), (-y, x), (-x, -y), (y, -x))[uid]
        xyz[uid, :2], xyz[uid, 2] = np.asarray(rotated) + center, t["altitude_m"]
    return xyz


def channel_at(physical_s: float, config: dict) -> dict:
    """式(6)—(8)：在给定巡逻轨迹上计算确定性大尺度平均增益，无小尺度衰落。

    轨迹位置连续；实验主循环仍在轮初冻结信道，不在整轮内部积分。
    """
    t, c = config["trajectory"], config["channel"]
    xyz = patrol_positions(physical_s, config)
    delta = xyz - np.asarray(t["server_xyz_m"])
    distance = np.linalg.norm(delta, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(delta[:, 2] / distance, -1, 1)))
    probability = 1 / (1 + c["los_a"] * np.exp(-c["los_b"] * (elevation - c["los_a"])))
    gain = probability * c["beta_l"] * distance ** (-c["alpha_l"]) + (1 - probability) * c["beta_n"] * distance ** (-c["alpha_n"])
    result = {"xyz_m": xyz, "distance_m": distance, "elevation_deg": elevation, "los_probability": probability, "gain": gain}
    if t.get("kind", "circle") == "serpentine_equal":
        template = _patrol_template(config)
        _, phases = _patrol_frame(config)
        result.update(trajectory_kind="serpentine_equal", patrol_period_s=template["period_s"],
                      patrol_length_m=template["length_m"], phase_offsets_s=phases * template["period_s"])
    return result


def uav_parameter(config: dict, key: str) -> np.ndarray:
    """将共享标量或逐机列表统一为长度 N 的只读数值视图。"""
    return np.broadcast_to(np.asarray(config["uav"][key], dtype=float), (config["num_uavs"],))


def client_cycle_table(profile: dict, num_uavs: int) -> np.ndarray:
    """旧画像共享五列，新画像每机一行。"""
    values = np.asarray(profile["client_cycles"], dtype=float)
    if values.shape not in ((len(SPLIT_POINTS),), (num_uavs, len(SPLIT_POINTS))):
        raise ValueError("客户端周期画像必须为 [5] 或 [num_uavs,5]")
    return np.broadcast_to(values, (num_uavs, len(SPLIT_POINTS)))


def maximum_frequency(config: dict) -> np.ndarray:
    """式(9)：由计算功率上限恢复 f_max，返回 Hz。"""
    return (uav_parameter(config, "compute_power_max_w") / uav_parameter(config, "kappa")) ** (1 / 3)


def reference_resources(config: dict) -> dict:
    """返回独立参考配置；其可行性必须由 audit_resources 判定。"""
    n = config["num_uavs"]
    return {"frequency_hz": maximum_frequency(config) * config["reference"]["frequency_fraction"],
            "power_w": uav_parameter(config, "power_max_w") * config["reference"]["power_fraction"],
            "up_hz": np.full(n, config["radio"]["total_up_hz"] / n),
            "down_hz": np.full(n, config["radio"]["total_down_hz"] / n)}


def shannon_rate(bandwidth_hz: Any, power_w: Any, gain: Any, noise_psd: float) -> np.ndarray:
    """式(8)，返回 bit/s；带宽内噪声=N0*B。"""
    b, p, h = np.asarray(bandwidth_hz), np.asarray(power_w), np.asarray(gain)
    if np.any(b <= 0) or np.any(p <= 0) or np.any(h <= 0):
        raise ValueError("速率输入必须严格为正")
    return b * np.log1p(p * h / (noise_psd * b)) / np.log(2)


def make_problem(epoch: int, count: int, gain: Any, split: Any, next_split: Any,
                 remaining_j: Any, config: dict) -> dict:
    """构造当前 epoch 的独立 Problem，供成本计算与资源优化共用。

    count 为每机本轮冻结样本数 D_k；gain、split、remaining_j 均为长度 N。
    split 是本轮执行的切点，next_split 只决定聚合后新前缀的下发负载。
    非最终偶数轮必须传 next_split；奇数轮和最终轮必须传 None。这样可以在
    P2 求解前确定所有通信量，避免求解后才切换模型造成时延/能耗漏算。
    返回时复制输入数组，后续扣电量或切换切点不会修改这份问题快照。
    """
    n, epochs = config["num_uavs"], config["training"]["epochs"]
    aggregate, final = epoch % 2 == 0, epoch == epochs
    if not 1 <= epoch <= epochs or count < 1:
        raise ValueError("epoch/count 越界")
    s = np.asarray(split, dtype=int)
    q = None if next_split is None else np.asarray(next_split, dtype=int)
    if not np.array_equal(np.asarray(split), s) or (q is not None and not np.array_equal(np.asarray(next_split), q)):
        raise ValueError("切分编号必须为整数")
    if s.shape != (n,) or not np.isin(s, SPLIT_POINTS).all():
        raise ValueError("当前切分向量非法")
    if aggregate and not final and (q is None or q.shape != (n,) or not np.isin(q, SPLIT_POINTS).all()):
        raise ValueError("非最终聚合轮需要下一周期切分向量")
    if (not aggregate or final) and q is not None:
        raise ValueError("非聚合轮或最终轮不应传 next_split")
    gain, remaining = np.asarray(gain, dtype=float), np.asarray(remaining_j, dtype=float)
    if gain.shape != (n,) or remaining.shape != (n,) or not np.isfinite(gain).all() or np.any(gain <= 0) or not np.isfinite(remaining).all():
        raise ValueError("信道或剩余电量形状/数值不合法")
    return {"epoch": epoch, "count": count, "gain": gain.copy(), "split": s.copy(),
            "next_split": None if q is None else q.copy(), "remaining_j": remaining.copy(),
            "aggregate": aggregate, "final": final}


def loads(problem: dict, profile: dict) -> dict:
    """返回当前轮各机周期数、训练通信量、同步通信量；最终下发严格为零。"""
    s, d = problem["split"] // BLOCKS_PER_GROUP, problem["count"]
    zero = np.zeros(len(s))
    # 计算/激活/梯度画像按“单样本”给出，乘 D_k 后才是整轮负载。
    # prefix_bits 已是一份完整前缀的大小，只在聚合时传一次，不再乘 D_k。
    # upload 用当前 s；download 用 next_split，二者可能跨越不同残差块边界。
    return {"client_cycles": d * client_cycle_table(profile, len(s))[np.arange(len(s)), s],
            "server_cycles": d * profile["server_cycles"][s],
            "activation_bits": d * profile["activation_bits"][s], "gradient_bits": d * profile["gradient_bits"][s],
            "upload_bits": profile["prefix_bits"][s] if problem["aggregate"] else zero,
            "download_bits": profile["prefix_bits"][problem["next_split"] // BLOCKS_PER_GROUP] if problem["next_split"] is not None else zero}


def evaluate_resources(problem: dict, resources: dict, profile: dict, config: dict) -> dict:
    """用原式(10)—(17)计算真实模型成本，返回可直接记录的 metrics 字典。

    Resources 的 frequency_hz/power_w/up_hz/down_hz 均为长度 N 的物理量数组。
    duration_s、total_energy_j、server_energy_j 为标量；uav_energy_j、各链路
    速率及 components 中各能耗项为长度 N 的数组；stage_times 为阶段标量。
    此处的“真实”指原始数学模型值，用来核验凸上界解，不是硬件功耗实测。
    """
    l = loads(problem, profile)
    f, p = resources["frequency_hz"], resources["power_w"]
    e, u, c = config["server"], config["uav"], config["channel"]
    ru = shannon_rate(resources["up_hz"], p, problem["gain"], c["noise_psd_w_hz"])
    rd = shannon_rate(resources["down_hz"], e["downlink_power_w"], problem["gain"], c["noise_psd_w_hz"])
    tc, te = l["client_cycles"] / f, l["server_cycles"] / e["frequency_hz"]
    ta, tg, tu, td = l["activation_bits"] / ru, l["gradient_bits"] / rd, l["upload_bits"] / ru, l["download_bits"] / rd
    components = {"uav_compute_j": uav_parameter(config, "kappa") * l["client_cycles"] * f ** 2,
                  "activation_upload_j": p * ta, "prefix_upload_j": p * tu,
                  "server_compute_j": e["kappa"] * e["frequency_hz"] ** 2 * l["server_cycles"],
                  "gradient_download_j": e["downlink_power_w"] * tg,
                  "prefix_download_j": e["downlink_power_w"] * td}
    # UAV 电池只扣本地计算和两类上行；下行发射能耗属于服务器。
    # 聚合处理能耗为服务器侧一次性开销，不分摊进各 UAV 的训练电池。
    uav_energy = components["uav_compute_j"] + components["activation_upload_j"] + components["prefix_upload_j"]
    agg_time = e["aggregation_time_s"] if problem["aggregate"] else 0.0
    agg_energy = e["aggregation_energy_j"] if problem["aggregate"] else 0.0
    server_energy = float(sum(np.sum(components[k]) for k in ("server_compute_j", "gradient_download_j", "prefix_download_j")) + agg_energy)
    # 同步屏障使每阶段都要等待该阶段最慢分支，且各阶段最慢的 UAV 可以不同。
    # 因而要 sum(max(stage_n))；不能把四个阶段相加后只取一次 max。
    stage_times = {"split_training_s": float(np.max(tc + te + ta + tg)), "prefix_upload_s": float(np.max(tu)),
                   "aggregation_s": agg_time, "prefix_download_s": float(np.max(td))}
    return {"duration_s": sum(stage_times.values()), "total_energy_j": float(uav_energy.sum() + server_energy),
            "uav_energy_j": uav_energy, "server_energy_j": server_energy, "components": components,
            "aggregation_energy_j": agg_energy, "stage_times": stage_times,
            "branch_training_s": tc + te + ta + tg, "up_rate_bps": ru, "down_rate_bps": rd,
            "compute_power_w": uav_parameter(config, "kappa") * f ** 3, "loads": l}


def audit_resources(problem: dict, resources: dict, profile: dict, config: dict) -> dict:
    """原始约束核验，返回 ok、无量纲最大违约与带单位的物理残差。

    残差按“实际值 - 上界”定义：正数表示违约，负数表示尚有余量。为让 Hz、W、
    J、s 可以共用数值容差，分别除以对应上界，再取正部最大值。无效资源数组
    会直接返回 ok=False/reason，不计算可能含除零的速率；调用方先读 ok。
    求解器报告 optimal/optimal_inaccurate 后，也必须通过这一原式检查。
    """
    n, tol = config["num_uavs"], config["optimization"]["feasibility_tolerance"]
    if any(np.shape(v) != (n,) or not np.isfinite(v).all() or np.any(np.asarray(v) <= 0) for v in resources.values()):
        return {"ok": False, "reason": "nonpositive_or_invalid_resources"}
    m = evaluate_resources(problem, resources, profile, config)
    residuals = {"deadline_s": m["duration_s"] - config["task"]["epoch_deadline_s"],
                 "battery_j": m["uav_energy_j"] - problem["remaining_j"],
                 "frequency_hz": resources["frequency_hz"] - maximum_frequency(config),
                 "power_w": resources["power_w"] - uav_parameter(config, "power_max_w"),
                 "up_hz": float(np.sum(resources["up_hz"])) - config["radio"]["total_up_hz"],
                 "down_hz": float(np.sum(resources["down_hz"])) - config["radio"]["total_down_hz"]}
    normalized = [residuals["deadline_s"] / config["task"]["epoch_deadline_s"],
                  float(np.max(residuals["battery_j"] / np.maximum(1.0, problem["remaining_j"]))),
                  float(np.max(residuals["frequency_hz"] / maximum_frequency(config))),
                  float(np.max(residuals["power_w"] / uav_parameter(config, "power_max_w"))),
                  residuals["up_hz"] / config["radio"]["total_up_hz"], residuals["down_hz"] / config["radio"]["total_down_hz"]]
    violation = max(0.0, *normalized)
    return {"ok": bool(violation <= tol and np.isfinite(m["total_energy_j"])),
            "normalized_violation": violation, "residuals": residuals}


# 5. 单 epoch 凸子问题、可行初始化、逆二次迭代 ----------------------------------
def normalized_resource_variables(problem: dict, num_uavs: int,
                                  tangent: dict | None, fixed_power: float | None) -> dict:
    """创建频率、功率、带宽比例与归一化阶段时长变量。"""
    import cvxpy as cp
    # f=f_actual/f_max，p=p_actual/p_max，bu/bd=B_actual/B_total。
    f = cp.Variable(num_uavs, name="frequency_fraction")
    bu, bd = cp.Variable(num_uavs, name="up_fraction"), cp.Variable(num_uavs, name="down_fraction")
    p = cp.Variable(num_uavs, name="power_fraction") if tangent is not None else np.full(num_uavs, fixed_power)
    # 奇数轮仅训练；中间聚合轮还上传和下发；末轮没有下发。
    tau = cp.Variable(1 + int(problem["aggregate"]) + int(problem["next_split"] is not None),
                      nonneg=True, name="stage_s")
    return {"f": f, "bu": bu, "bd": bd, "p": p, "tau": tau}


def convex_rate_expressions(problem: dict, config: dict, variables: dict) -> dict:
    """用 Shannon perspective 构建 Mbit/s 速率及其倒数表达式。"""
    import cvxpy as cp
    bu, bd, p = variables["bu"], variables["bd"], variables["p"]
    radio, e = config["radio"], config["server"]
    noise, gain = config["channel"]["noise_psd_w_hz"], problem["gain"]
    pmax = uav_parameter(config, "power_max_w")
    # -rel_entr(b,b+a*p)=b*log(1+a*p/b)，对带宽和功率联合凹。
    ru = -(radio["total_up_hz"] / 1e6) * cp.rel_entr(bu, bu + cp.multiply(gain * pmax / (noise * radio["total_up_hz"]), p)) / np.log(2)
    rd = -(radio["total_down_hz"] / 1e6) * cp.rel_entr(bd, bd + gain * e["downlink_power_w"] / (noise * radio["total_down_hz"])) / np.log(2)
    invru, invrd = cp.inv_pos(ru), cp.inv_pos(rd)
    return {"ru": ru, "rd": rd, "invru": invru, "invrd": invrd}


def convex_energy_expressions(problem: dict, profile: dict, config: dict, load_data: dict,
                              variables: dict, rates: dict, tangent: dict | None,
                              fixed_power: float | None) -> dict:
    """构建计算/通信能量、变量边界，以及 Phase-I 或相切上界的尺度。"""
    import cvxpy as cp
    n, floor = config["num_uavs"], config["optimization"]["positive_floor"]
    f, bu, bd, p = (variables[key] for key in ("f", "bu", "bd", "p"))
    ru, invru, invrd = rates["ru"], rates["invru"], rates["invrd"]
    fmax = maximum_frequency(config)
    pmax, kappa = uav_parameter(config, "power_max_w"), uav_parameter(config, "kappa")
    ec = cp.multiply(kappa * load_data["client_cycles"] * fmax ** 2, cp.square(f))
    # 特征通信与模型同步共享资源，能量先合并载荷，时延仍分阶段约束。
    up_mbit = (load_data["activation_bits"] + load_data["upload_bits"]) / 1e6
    down_mbit = (load_data["gradient_bits"] + load_data["download_bits"]) / 1e6
    constraints = [f >= floor, f <= 1, bu >= floor, bd >= floor, cp.sum(bu) <= 1, cp.sum(bd) <= 1]
    if tangent is not None:
        m0 = evaluate_resources(problem, tangent, profile, config)
        ru0, p0 = m0["up_rate_bps"] / 1e6, tangent["power_w"]
        # 当前点 psi=1，E_up0/psi 与原上行能量相切；正凹函数的倒数为凸。
        psi = 2 * cp.sqrt(cp.multiply(1 / ru0, ru)) - cp.multiply(pmax / p0, p)
        eup0 = up_mbit * p0 / ru0
        eup = cp.multiply(eup0, cp.inv_pos(psi))
        constraints += [p >= floor, p <= 1, psi >= floor]
        scale, slack = max(1e-6, m0["total_energy_j"] / n), 0.0
    else:
        eup = cp.multiply(up_mbit * pmax * fixed_power, invru)
        scale = max(1e-6, float(np.sum(kappa * load_data["client_cycles"] * fmax ** 2)) / n)
        slack = cp.Variable(nonneg=True, name="phase_one_slack")
    edown = cp.multiply(config["server"]["downlink_power_w"] * down_mbit, invrd)
    return {"compute": ec, "uplink": eup, "downlink": edown, "constraints": constraints,
            "scale": scale, "slack": slack}


def build_convex_program(problem: dict, load_data: dict, config: dict, variables: dict,
                         rates: dict, energy: dict, tangent: dict | None) -> Any:
    """按原顺序加入各阶段时延、电量约束，返回能量最小化凸程序。"""
    import cvxpy as cp
    f, tau = variables["f"], variables["tau"]
    invru, invrd = rates["invru"], rates["invrd"]
    e, deadline = config["server"], config["task"]["epoch_deadline_s"]
    ec, eup, edown = energy["compute"], energy["uplink"], energy["downlink"]
    scale, slack = energy["scale"], energy["slack"]
    train_time = cp.multiply(load_data["client_cycles"] / maximum_frequency(config), cp.inv_pos(f)) + load_data["server_cycles"] / e["frequency_hz"]
    train_time += cp.multiply(load_data["activation_bits"] / 1e6, invru) + cp.multiply(load_data["gradient_bits"] / 1e6, invrd)
    constraints = list(energy["constraints"])
    constraints += [train_time / deadline <= tau[0]]
    if problem["aggregate"]:
        constraints += [cp.multiply(load_data["upload_bits"] / 1e6 / deadline, invru) <= tau[1]]
    if problem["next_split"] is not None:
        constraints += [cp.multiply(load_data["download_bits"] / 1e6 / deadline, invrd) <= tau[2]]
    agg_time = e["aggregation_time_s"] if problem["aggregate"] else 0.0
    battery_scale = np.maximum(scale, np.abs(problem["remaining_j"]))
    # Phase-I 放宽归一化时限和电量；后续仍以 audit_resources 判定是否可行。
    constraints += [cp.sum(tau) + agg_time / deadline <= 1 + slack,
                    cp.multiply(1 / battery_scale, ec + eup) <= problem["remaining_j"] / battery_scale + slack]
    # 后缀计算和聚合能量为常数，此处省略，原式成本统计时完整加回。
    objective = cp.sum(ec + eup + edown) / scale
    objective = objective if tangent is not None else slack + 1e-8 * objective
    return cp.Problem(cp.Minimize(objective), constraints)


def solve_convex_program(program: Any, solver: str) -> dict:
    """检查 DCP/求解器并执行求解，记录求解警告和原有计时范围。"""
    import cvxpy as cp
    if not program.is_dcp():
        raise RuntimeError("凸子问题没有通过 CVXPY DCP 校验")
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"求解器 {solver} 不可用；当前 {cp.installed_solvers()}")
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            options = {"max_iter": 300, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9} if solver == "CLARABEL" else {}
            program.solve(solver=solver, **options)
        notes = [str(w.message) for w in caught]
    except cp.error.SolverError as exc:
        return {"ok": False, "status": "solver_error", "message": str(exc), "wall_s": time.perf_counter() - start}
    return {"ok": False, "status": program.status, "warnings": notes, "dcp": True,
            "wall_s": time.perf_counter() - start}


def recover_convex_resources(variables: dict, config: dict, tangent: dict | None) -> dict:
    """将求解后的比例变量恢复为实际频率、功率和上下行带宽。"""
    p, radio = variables["p"], config["radio"]
    power_fraction = np.asarray(p.value if tangent is not None else p)
    return {"frequency_hz": np.asarray(variables["f"].value) * maximum_frequency(config),
            "power_w": power_fraction * uav_parameter(config, "power_max_w"),
            "up_hz": np.asarray(variables["bu"].value) * radio["total_up_hz"],
            "down_hz": np.asarray(variables["bd"].value) * radio["total_down_hz"]}



def convex_subproblem(problem: dict, profile: dict, config: dict,
                      tangent: dict | None = None, fixed_power: float | None = None) -> dict:
    """求解一次凸问题并返回资源/求解状态。

    tangent != None：式(31)，用相切点归一化后的 psi=2*sqrt(R/R0)-p/p0，
    上行上界=E_up0/psi，与式(29)—(31)完全等价，避免 y 的数量级病态。
    fixed_power != None：固定发射功率的凸 Phase-I，最小化时延/电量松弛；
    只用于寻找原问题可行点，找不到不构成全局不可行证明。

    tangent 为已经验证可行的 Resources；fixed_power 是 p_max 的比例，两者
    必须且只能提供一个。输出 ok 指原约束也已通过，而不只是求解器有数值解。
    成功时 resources 恢复为 Hz/W，epigraph_s 恢复为秒；status/warnings 保存
    求解器状态。失败结果可能没有 resources/audit，调用方需先检查 ok。
    """
    import cvxpy as cp
    if (tangent is None) == (fixed_power is None):
        raise ValueError("必须且只能指定 tangent 或 fixed_power")
    load_data = loads(problem, profile)
    variables = normalized_resource_variables(problem, config["num_uavs"], tangent, fixed_power)
    rates = convex_rate_expressions(problem, config, variables)
    energy = convex_energy_expressions(problem, profile, config, load_data, variables,
                                       rates, tangent, fixed_power)
    program = build_convex_program(problem, load_data, config, variables, rates, energy, tangent)
    result = solve_convex_program(program, config["optimization"]["solver"])
    if result["status"] not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or any(
            variables[key].value is None for key in ("f", "bu", "bd")):
        return result
    resources = recover_convex_resources(variables, config, tangent)
    audit = audit_resources(problem, resources, profile, config)
    result.update({"ok": audit["ok"], "resources": resources, "audit": audit,
                   "epigraph_s": np.asarray(variables["tau"].value) * config["task"]["epoch_deadline_s"],
                   "phase_slack": float(energy["slack"].value) if tangent is None else 0.0})
    return result


def feasible_initialization(problem: dict, profile: dict, config: dict, previous: dict | None = None) -> dict:
    """返回 {ok, resources, attempts}；失败还包含明确的 status。

    尝试顺序为上一可行配置、初始参考配置、最大频率/功率等分带宽，最后在若干
    固定功率比例下解凸 Phase-I。每个候选均使用当前 Problem 重新审计，不能
    因它上一轮可行就直接复用。attempts 记录全部尝试，便于区分初始化失败与
    后续凸迭代失败。此有限初始化搜索不能提供原非凸问题的不可行证明。
    """
    attempts = []
    for name, candidate in (("previous", previous), ("reference", reference_resources(config))):
        if candidate is None:
            continue
        audit = audit_resources(problem, candidate, profile, config)
        attempts.append({"method": name, "audit": audit})
        if audit["ok"]:
            return {"ok": True, "resources": copy.deepcopy(candidate), "attempts": attempts}
    maximum = reference_resources(config)
    maximum["frequency_hz"][:] = maximum_frequency(config)
    maximum["power_w"][:] = uav_parameter(config, "power_max_w")
    audit = audit_resources(problem, maximum, profile, config)
    attempts.append({"method": "maximum_equal_bandwidth", "audit": audit})
    if audit["ok"]:
        return {"ok": True, "resources": maximum, "attempts": attempts}
    for fraction in (1.0, 0.3, 0.1, 0.03, 0.01, 0.001):
        result = convex_subproblem(problem, profile, config, fixed_power=max(fraction, config["optimization"]["positive_floor"]))
        attempts.append({"method": "fixed_power_phase_one", "power_fraction": fraction,
                         **{k: v for k, v in result.items() if k != "resources"}})
        if result["ok"]:
            return {"ok": True, "resources": result["resources"], "attempts": attempts}
    return {"ok": False, "resources": None, "attempts": attempts, "status": "feasible_initialization_not_found"}


def solve_resources(problem: dict, profile: dict, config: dict, previous: dict | None = None,
                    *, retain_feasible: bool = False) -> dict:
    """式(30)/(31)交替求解，返回 SolveResult，接受步真实可行且真实目标不增。

    trace[0] 是已验证初值，此后每项记录真实能耗、相对下降、原约束审计及
    是否接受；并非所有日志项都会被执行。ok=True 时 resources/metrics 可用于
    本轮执行，但 status='iteration_limit_feasible' 只表示达到上限仍然可行。
    ok=False 时返回 resources=None，由上层尝试下一候选或报告任务未完成。
    retain_feasible 用于窗口能耗比较。迭代失败时保留最近通过原约束审计的点并记录原因。
    """
    start = time.perf_counter()
    initialization = feasible_initialization(problem, profile, config, previous)
    trace = []
    def result(ok, status, resources=None):
        return {"ok": ok, "status": status, "resources": resources,
                "metrics": evaluate_resources(problem, resources, profile, config) if resources is not None else None,
                "trace": trace, "initialization": initialization["attempts"], "wall_s": time.perf_counter() - start}
    if not initialization["ok"]:
        return result(False, "feasible_initialization_not_found")
    current = initialization["resources"]
    energy = evaluate_resources(problem, current, profile, config)["total_energy_j"]
    trace.append({"iteration": 0, "energy_j": energy, "audit": audit_resources(problem, current, profile, config)})
    for iteration in range(1, config["optimization"]["max_iterations"] + 1):
        proposed = convex_subproblem(problem, profile, config, tangent=current)
        if not proposed["ok"]:
            trace.append({"iteration": iteration, "accepted": False, "subproblem": proposed})
            if retain_feasible:
                return result(True, "subproblem_failed_previous_feasible", current)
            return result(False, "convex_subproblem_failed")
        candidate = proposed["resources"]
        # 停止条件和接受条件均依据原始总能量，而不是随切线更新的凸上界目标。
        # 分母防止能量接近零时出现除零；每步还独立通过了原始约束核验。
        metrics = evaluate_resources(problem, candidate, profile, config)
        new_energy = metrics["total_energy_j"]
        relative = (energy - new_energy) / max(abs(energy), 1e-12)
        entry = {"iteration": iteration, "energy_j": new_energy, "relative_decrease": relative,
                 "solver_status": proposed["status"], "audit": proposed["audit"],
                 "solver_wall_s": proposed["wall_s"], "warnings": proposed["warnings"]}
        if new_energy > energy:
            entry["accepted"] = False
            trace.append(entry)
            # 数值误差范围内保留前一个已验证的可行点；不能把升高的点写入执行状态。
            if -relative <= config["optimization"]["relative_tolerance"]:
                return result(True, "numerical_stationary_previous_feasible", current)
            if retain_feasible:
                return result(True, "objective_increased_previous_feasible", current)
            return result(False, "true_objective_increased")
        entry["accepted"] = True
        trace.append(entry)
        current, energy = candidate, new_energy
        if relative <= config["optimization"]["relative_tolerance"]:
            return result(True, "converged", current)
    return result(True, "iteration_limit_feasible", current)


# 6. 窗口切分、参考评分、离散可行性 --------------------------------------------
def reference_scores(count: int, gain: np.ndarray, resources: dict, profile: dict,
                     first_cycle: bool, config: dict) -> np.ndarray:
    """式(26)：返回 [N,5] 参考能耗评分矩阵，元素是某机某候选的周期评分 J。

    count 是决策时刻每机样本数，gain 是已完成 epoch 窗口的平均信道。
    预测周期固定使用 m=2*count。首周期模型已预部署，chi=0；后续周期
    chi=1，计入新前缀下发。与新切点无关的旧前缀上传和聚合常数在评分中
    省略，但 screen_split 的可行性检查仍包括这些成本。
    """
    u, e, noise = config["uav"], config["server"], config["channel"]["noise_psd_w_hz"]
    ru = shannon_rate(resources["up_hz"], resources["power_w"], gain, noise)[:, None]
    rd = shannon_rate(resources["down_hz"], e["downlink_power_w"], gain, noise)[:, None]
    m, chi = 2 * count, int(not first_cycle)
    compute = m * (uav_parameter(config, "kappa")[:, None] * resources["frequency_hz"][:, None] ** 2
                   * client_cycle_table(profile, config["num_uavs"])
                   + e["kappa"] * e["frequency_hz"] ** 2 * profile["server_cycles"])
    upload = resources["power_w"][:, None] * (m * profile["activation_bits"] + profile["prefix_bits"]) / ru
    download = e["downlink_power_w"] * (m * profile["gradient_bits"] + chi * profile["prefix_bits"]) / rd
    return compute + upload + download


def screen_split(candidate: np.ndarray, current: np.ndarray, epoch: int, count: int,
                 gain: np.ndarray, remaining_j: np.ndarray, reference: dict,
                 profile: dict, config: dict) -> dict:
    """式(25)：返回参考预测范围的可行性、各轮时长、累计每机能耗及拒绝原因。

    首周期检查 epoch=1,2；在 epoch=2 预选下一周期时，检查当前未执行的
    聚合轮 2 和未来训练轮 3,4。整个范围固定当前 count 和窗口信道，属于
    参考预测，不读取未来真实 D(t)/h(t)。电池约束按范围内累计能耗检查。
    """
    first = epoch == 1
    start = 1 if first else epoch + 1
    horizon = []
    if not first:
        horizon.append(make_problem(epoch, count, gain, current, candidate, remaining_j, config))
    for k in range(start, start + 2):
        q = candidate if k % 2 == 0 and k < config["training"]["epochs"] else None
        horizon.append(make_problem(k, count, gain, candidate, q, remaining_j, config))
    metrics = [evaluate_resources(p, reference, profile, config) for p in horizon]
    total_uav = sum(m["uav_energy_j"] for m in metrics)
    times = [m["duration_s"] for m in metrics]
    tol = config["optimization"]["feasibility_tolerance"]
    time_ok = max(times) <= config["task"]["epoch_deadline_s"] * (1 + tol)
    energy_ok = bool(np.all(total_uav <= remaining_j + tol * np.maximum(1, remaining_j)))
    return {"ok": bool(time_ok and energy_ok), "duration_s": times, "uav_energy_j": total_uav,
            "reason": "feasible" if time_ok and energy_ok else ("reference_deadline" if not time_ok else "reference_battery")}


def forecast_split_energy(cuts, *, config, profile, epoch, count, current, gain, remaining, reference):
    """在相同预测信道和样本数下逐轮优化并扣电，累计原公式的系统能耗。"""
    # 即使忽略全部通信，最快 CPU 也无法按时完成的切点可直接排除。
    tol = config["optimization"]["feasibility_tolerance"]
    max_clock = maximum_frequency(config) * (1 + tol)
    compute_seconds = count * (client_cycle_table(profile, config["num_uavs"]) / max_clock[:, None]
                               + profile["server_cycles"] / config["server"]["frequency_hz"])
    used_cuts = [cuts] if epoch == 1 else [current, cuts]
    if any(np.any(compute_seconds[np.arange(config["num_uavs"]), np.asarray(vector) // BLOCKS_PER_GROUP]
                  > config["task"]["epoch_deadline_s"] * (1 + tol)) for vector in used_cuts):
        return cuts, None, "compute_time_exceeds_deadline"
    battery = np.array(remaining, dtype=float)
    total = 0.0
    statuses = []
    last = 2 if epoch == 1 else epoch + 2
    for k in range(epoch, last + 1):
        active = current if k == epoch and epoch != 1 else cuts
        next_cuts = cuts if k % 2 == 0 and k < config["training"]["epochs"] else None
        problem = make_problem(k, count, gain, active, next_cuts, battery, config)
        result = solve_resources(problem, profile, config, reference, retain_feasible=True)
        if not result["ok"]:
            return cuts, None, f"epoch_{k}: {result['status']}"
        total += result["metrics"]["total_energy_j"]
        battery -= result["metrics"]["uav_energy_j"]
        statuses.append(result["status"])
    return cuts, total, ", ".join(statuses)


def select_splits(epoch: int, count: int, current: np.ndarray, gain_history: list[np.ndarray],
                  initial_gain: np.ndarray, remaining_j: np.ndarray, reference: dict,
                  profile: dict, config: dict) -> dict:
    """穷举组边界组合，以式(25)筛选并按式(26)的参考总能耗排列 P2 候选。

    scores_j 的列按 SPLIT_POINTS 排列。总分相同时优先沿用当前组合，
    其余同分组合按切点向量的字典序排列。
    """
    o, n = config["optimization"], config["num_uavs"]
    window = 1 if o["strategy"] == "no_window" else o["window_epochs"]
    gain = np.mean(gain_history[-window:], axis=0) if gain_history else initial_gain.copy()
    scores = reference_scores(count, gain, reference, profile, epoch == 1, config)
    current_key = tuple(current)
    if o["split_objective"] == "optimized_window":
        evaluate = partial(forecast_split_energy, config=config, profile=profile, epoch=epoch, count=count,
                           current=current, gain=gain, remaining=remaining_j, reference=reference)
        combinations = product(SPLIT_POINTS, repeat=n)
        if o["split_workers"] == 1:
            rows = list(map(evaluate, combinations))
        else:
            with ProcessPoolExecutor(max_workers=o["split_workers"]) as pool:
                rows = list(pool.map(evaluate, combinations, chunksize=4))
        ranked = sorted((energy, cuts != current_key, cuts) for cuts, energy, _ in rows if energy is not None)
        return {"ok": bool(ranked), "status": "optimized_window_complete" if ranked else "no_window_solution_found",
                "candidates": [np.array(cuts) for _, _, cuts in ranked], "scores_j": scores, "window_gain": gain,
                "trace": [{"vector": cuts, "forecast_energy_j": energy, "feasible": energy is not None,
                           "reason": status} for cuts, energy, status in rows],
                "combinations_evaluated": len(rows), "feasible_combinations": len(ranked)}
    uavs = np.arange(n)
    trace, feasible = [], []
    for cuts in product(SPLIT_POINTS, repeat=n):
        candidate = np.array(cuts, dtype=int)
        score = float(scores[uavs, candidate // BLOCKS_PER_GROUP].sum())
        audit = screen_split(candidate, current, epoch, count, gain, remaining_j, reference, profile, config)
        trace.append({"vector": cuts, "score_j": score, "feasible": audit["ok"], "reason": audit["reason"]})
        if audit["ok"]:
            feasible.append((score, cuts != current_key, cuts))
    feasible.sort()
    return {"ok": bool(feasible), "status": "exhaustive_complete" if feasible else "no_reference_feasible_split",
            "candidates": [np.array(cuts, dtype=int) for _, _, cuts in feasible], "scores_j": scores,
            "window_gain": gain, "trace": trace, "combinations_evaluated": len(trace),
            "feasible_combinations": len(feasible)}


def rebuild_reference(epoch: int, count: int, current: np.ndarray, window_gain: np.ndarray,
                      remaining_j: np.ndarray, profile: dict, config: dict) -> dict:
    """可选工程扩展（文稿默认关闭）：重建固定参考资源后重做 P1。

    用保留当前切分的聚合轮找可行配置，并将电量按参考 horizon 均分作为保守
    初始化预算。仍须通过完整式(25)筛选及当前真实状态 P2，不保证任务能完成。
    """
    first = epoch == 1
    aggregation_epoch = 2 if first else epoch
    q = current if aggregation_epoch < config["training"]["epochs"] else None
    horizon_size = 2 if first else 3
    problem = make_problem(aggregation_epoch, count, window_gain, current, q, remaining_j / horizon_size, config)
    initialized = feasible_initialization(problem, profile, config)
    if not initialized["ok"]:
        return initialized
    screen = screen_split(current, current, epoch, count, window_gain, remaining_j,
                          initialized["resources"], profile, config)
    return {**initialized, "ok": screen["ok"], "reference_screen": screen}


# 7. 联合仿真/训练流程与结果保存 -------------------------------------------------
def code_version() -> dict:
    """记录当前提交与实际源码哈希，Git 不可用时仍可追溯源码。"""
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    commit, dirty = None, None
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=5)
        commit = result.stdout.strip()
        result = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                 "--untracked-files=normal", "--", ".", ":(exclude)output/**"],
                                capture_output=True, text=True, check=True, timeout=5)
        dirty = bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": source_sha}


def make_output_directory(config: dict, label: str, version: dict | None = None) -> Path:
    """I/O：按北京时间创建唯一目录，不覆盖已有实验。"""
    version = code_version() if version is None else version
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    tag = version["git_commit"][:12] if version["git_commit"] else "src" + version["source_sha256"][:12]
    if version["git_dirty"]:
        tag += "_dirty_src" + version["source_sha256"][:8]
    output = Path(config["output_root"]) / f"output_{stamp}_{tag}_{label}_seed{config['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    return output


@contextmanager
def experiment_output(config: dict, label: str, output: str | Path | None = None, *,
                      training_device_requested: str | None = None):
    """为各入口保存参数、版本和运行状态，独占结果目录。"""
    version = code_version()
    directory = Path(output) if output else make_output_directory(config, label, version)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "config.json").exists():
        raise FileExistsError(f"实验目录已被使用: {directory}")
    now = lambda: datetime.now(timezone(timedelta(hours=8))).isoformat()
    metadata = {"mode": label, "status": "running", "started_at": now(), "finished_at": None,
                "output_dir": str(directory.resolve()), "code_version": version}
    if training_device_requested is not None:
        device = config["training"]["device"]
        metadata.update(training_device=device, training_device_requested=training_device_requested)
        print(f"模型执行设备：{device}（请求：{training_device_requested}）", file=sys.stderr, flush=True)
    # 排他创建运行记录，防止两个显式 --output 同时写入同一目录。
    with (directory / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    started = time.perf_counter()
    try:
        save_json(directory / "config.json", config)
        (directory / "uav_sfl_snapshot.py").write_bytes(Path(__file__).read_bytes())
        packages = {}
        for package in ("numpy", "Pillow", "torch", "cvxpy", "clarabel"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        save_json(directory / "runtime.json", {"python": sys.version, "platform": platform.platform(),
                                               "packages": packages, "code_version": version,
                                               "code_sha256": version["source_sha256"]})
        yield directory, metadata
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        error={"type": type(exc).__name__, "message": str(exc)})
        raise
    else:
        if metadata["status"] == "running":
            metadata["status"] = "completed"
    finally:
        metadata.update(finished_at=now(), wall_s=time.perf_counter() - started)
        save_json(directory / "run.json", metadata)


def checkpoint_payload(model: dict) -> dict:
    """返回 detached CPU 张量字典，供 torch.save；不改变训练状态。"""
    return {group: {k: v.detach().cpu().clone() for k, v in values.items()} for group, values in model.items()}


def run_experiment(config: dict, train: bool = False, data: dict | None = None,
                   profile: dict | None = None, output: str | Path | None = None,
                   label: str | None = None) -> dict:
    """算法1主入口，返回 RunResult，并保存可复核的参数、轨迹、资源及训练结果。

    train=False 只执行物理成本仿真；train=True 在资源通过原约束后执行真实 SGD。
    服务器并行分支在本地串行模拟，不把本机墙钟时间计作系统模型时间。

    data/profile 可传入准备好的 Data/Profile，否则在本次运行中生成；output
    是本次运行的新目录，已有 config.json 时拒绝覆盖。函数内部写日志和模型。
    返回的 summary/history 可用于进一步分析；Model 对象保留在内存返回值中。
    只有训练且全部轮次完成时 final_model 才非 None；best_model 仅按验证集选取。
    单轮可行不等于任务必然完成，调用者应先检查 summary['completed']。
    """
    config = validate_config(config)
    requested_device = config["training"]["device"]
    config["training"]["device"] = resolve_training_device(requested_device)
    with experiment_output(config, label or ("train" if train else "simulate"), output,
                           training_device_requested=requested_device) as (directory, metadata):
        result = execute_experiment(config, train, data, profile, directory, metadata["code_version"])
        summary = result["summary"]
        metadata.update(status=summary["status"], epochs_completed=summary["epochs_completed"],
                        epochs_planned=summary["epochs_planned"], failure=summary["failure"])
        return result


def prepare_experiment_inputs(config, train, data, profile, output):
    """准备并存档模型、数据和画像；校验外部传入的数据契约。"""
    base, provenance = initialize_model(config, require_pretrained=train)
    data = prepare_data(config) if data is None else data
    profile = profile_splits(base, config) if profile is None else profile
    if len(data["clients"]) != config["num_uavs"] or data["image_size"] != config["data"]["image_size"]:
        raise ValueError("传入 Data 与 Config 的 UAV 数/图像尺寸不一致")
    profile = grouped_profile(profile)
    client_cycle_table(profile, config["num_uavs"])
    save_json(output / "data_manifest.json", data)
    save_json(output / "split_profiles.json", profile)
    save_json(output / "initialization.json", provenance)
    return base, provenance, data, profile


def select_epoch_splits(epoch, count, current, gain_history, gain, remaining, reference, profile, config):
    """选择本轮候选；必要时重建参考资源并重新筛选，返回候选和参考资源。"""
    selection = select_splits(epoch, count, current, gain_history, gain, remaining,
                              reference, profile, config)
    if (not selection["ok"] and config["optimization"]["split_objective"] == "reference_score"
            and config["optimization"]["reference_recovery"] == "rebuild"):
        original_failure = selection
        recovery = rebuild_reference(epoch, count, current, selection["window_gain"], remaining, profile, config)
        if recovery["ok"]:
            reference = recovery["resources"]
            selection = select_splits(epoch, count, current, gain_history, gain, remaining,
                                      reference, profile, config)
        selection = {**selection, "reference_recovery": recovery, "original_reference_status": original_failure["status"]}

    return selection, reference


def solve_epoch_candidates(epoch, count, current, gain, remaining, reference, profile, config, selection):
    """依次尝试当前信道下的候选；成功后返回切点、问题和全部求解记录。"""
    strategy = config["optimization"]["strategy"]
    dynamic = strategy in ("dynamic_opt", "no_window")
    epochs = config["training"]["epochs"]
    choosing_current = dynamic and epoch == 1
    choosing_next = epoch % 2 == 0 and epoch < epochs
    queue = selection["candidates"] if selection else [current]
    attempts, solved, problem = [], None, None
    # 第三步：在真实当前信道下尝试候选并求 P2。参考筛选负责产生队列，
    # 不能代替实际时限/电池审计；任何候选都不可行时，在更新模型前停止。
    for cuts in queue:
        trial_current = cuts if choosing_current else current
        q = cuts if choosing_next else None
        problem = make_problem(epoch, count, gain, trial_current, q, remaining, config)
        if strategy == "fixed_fixed":
            resource = reference_resources(config)
            audit = audit_resources(problem, resource, profile, config)
            candidate = {"ok": audit["ok"], "status": "fixed_resources" if audit["ok"] else "fixed_resources_infeasible",
                         "resources": resource, "metrics": evaluate_resources(problem, resource, profile, config),
                         "trace": [], "audit": audit, "wall_s": 0.0}
        else:
            candidate = solve_resources(problem, profile, config, reference,
                                        retain_feasible=dynamic and config["optimization"]["split_objective"] == "optimized_window")
        attempts.append({"split": trial_current, "next_split": q, "solve": candidate})
        if candidate["ok"]:
            current = trial_current
            solved = candidate
            break

    return current, problem, solved, attempts


def train_epoch_branches(branches, data, indices, current, epoch, config, expected_loads):
    """训练所有 UAV 分支并核对通信载荷；全部成功后才返回新分支。"""
    next_branches, train_metrics = [], []
    for uid in range(config["num_uavs"]):
        new_model, stats = train_client_epoch(branches[uid], data, indices[uid], int(current[uid]), epoch, uid, config)
        # 真实 Tensor 通信载荷必须与物理模型一致（含标签和尾批）。
        if stats["activation_bits"] != expected_loads["activation_bits"][uid] or stats["gradient_bits"] != expected_loads["gradient_bits"][uid]:
            raise AssertionError("真实训练通信量与画像不一致")
        next_branches.append(new_model)
        train_metrics.append(stats)

    return next_branches, train_metrics


def save_history_csv(output: Path, history: list[dict]):
    """I/O：按原字段顺序将每轮物理成本、切点和验证准确率写入 CSV。"""
    fields = ["epoch", "start_s", "end_s", "count_per_uav", "duration_s", "energy_j", "server_j", "uav_j", "split", "next_split", "solver_status", "val_accuracy"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({"epoch": row["epoch"], "start_s": row["start_s"], "end_s": row["end_s"],
                             "count_per_uav": row["count_per_uav"], "duration_s": row["metrics"]["duration_s"],
                             "energy_j": row["metrics"]["total_energy_j"], "server_j": row["metrics"]["server_energy_j"],
                             "uav_j": json.dumps(row["metrics"]["uav_energy_j"].tolist()), "split": json.dumps(row["split"].tolist()),
                             "next_split": json.dumps(json_value(row["next_split"]) if row["next_split"] is not None else None),
                             "solver_status": row["solver_status"], "val_accuracy": row["validation"]["accuracy"] if row["validation"] else ""})


def save_training_results(config, data, output, summary, global_model, best_model, branches, history):
    """I/O：在训练结束后评估测试集，并按完成状态保存聚合模型或部分分支。"""
    import torch
    last_aggregation = summary["last_aggregation_epoch"]
    completed = summary["completed"]
    if last_aggregation:
        # 这里已离开训练循环，测试集从未参与选模。未完成时的“最近聚合模型”
        # 仍可供诊断，但不会被标记为完成任务的 final_model。
        summary["last_aggregated_test"] = evaluate_model(global_model, data, data["test"], config)
        summary["best_validation_model_test"] = evaluate_model(best_model, data, data["test"], config)
        if config["training"]["save_checkpoint"]:
            torch.save(checkpoint_payload(global_model), output / "last_aggregated_model.pt")
            torch.save(checkpoint_payload(best_model), output / "best_validation_model.pt")
    if not completed and history and config["training"]["save_checkpoint"]:
        torch.save({"branches": [checkpoint_payload(m) for m in branches], "epoch": len(history)}, output / "partial_branches.pt")


def execute_experiment(config, train, data, profile, output, version):
    """在已分配并存档的目录执行计算，输入/输出协议由 run_experiment 管理。"""
    import torch
    torch.set_num_threads(config["training"]["torch_threads"])
    start_wall = time.perf_counter()
    base, provenance, data, profile = prepare_experiment_inputs(config, train, data, profile, output)
    n, epochs = config["num_uavs"], config["training"]["epochs"]
    strategy = config["optimization"]["strategy"]
    dynamic = strategy in ("dynamic_opt", "no_window")
    current = np.full(n, config["optimization"]["fixed_split"], dtype=int)
    reference = reference_resources(config)
    remaining = uav_parameter(config, "energy_cap_j").copy()
    time_s, history, gain_history = data["start_s"], [], []
    branches = [base for _ in range(n)] if train else None  # 参数只读共享，SGD 返回新字典。
    global_model, best_model, best_val = base if train else None, None, None
    best_epoch, last_aggregation = None, 0
    failure, total_energy, processed = None, 0.0, 0
    try:
        for epoch in range(1, epochs + 1):
            # 第一步：只在轮初冻结 D_k、样本索引和当前链路状态。后续各批次
            # 共用本轮决策资源；本机求解/训练花费的墙钟时间不改变此物理快照。
            snapshot = epoch_snapshot(data, time_s, epoch, config)
            channel = channel_at(time_s, config)
            selection = None
            # 第二步：首轮确定当前切分；非最终偶数轮预选“下一周期”的切分。
            # 预选提前到当前聚合轮开始，是为了在 P2 求解前确定新前缀下发量。
            # 此时 current 仍用于本轮训练，直到该轮训练与聚合完成后才切换。
            if dynamic and (epoch == 1 or (epoch % 2 == 0 and epoch < epochs)):
                selection, reference = select_epoch_splits(
                    epoch, snapshot["count"], current, gain_history, channel["gain"], remaining, reference, profile, config)
                if not selection["ok"]:
                    failure = {"epoch": epoch, "reason": selection["status"], "selection": selection}
                    break
            # 非聚合轮和最终轮不下发新前缀，因此传 None；固定切分策略也需在
            # 中间聚合轮传 current，表示聚合后按同一切点重新下发参数。
            current, problem, solved, attempts = solve_epoch_candidates(
                epoch, snapshot["count"], current, channel["gain"], remaining, reference, profile, config, selection)
            if solved is None:
                failure = {"epoch": epoch, "reason": "no_current_feasible_candidate", "attempts": attempts}
                break
            metrics, train_metrics, validation = solved["metrics"], None, None
            training_wall = time.perf_counter()
            if train:
                # 第四步：每机从上一轮分支状态执行一次完整本地 epoch。
                # 新参数先放入 next_branches，所有分支完成后才接替 branches；
                # 无聚合的奇数轮保留各机独立参数，下一轮继续训练这些参数。
                next_branches, train_metrics = train_epoch_branches(
                    branches, data, snapshot["indices"], current, epoch, config, metrics["loads"])
                if problem["aggregate"]:
                    # 聚合完整参数，随后各分支从同一个新全局模型继续。参数只读
                    # 共享是安全的，因为下一个 sgd_step 会返回新参数而不原地改写。
                    # 最佳模型只看验证准确率；同分保持先前模型，测试集在末尾使用。
                    new_global = aggregate_models(next_branches, [snapshot["count"]] * n)
                    validation = evaluate_model(new_global, data, data["val"], config)
                    global_model, last_aggregation = new_global, epoch
                    next_branches = [new_global for _ in range(n)]
                    if best_val is None or validation["accuracy"] > best_val["accuracy"]:
                        best_model, best_val, best_epoch = new_global, validation, epoch
                branches = next_branches
            row = {"epoch": epoch, "start_s": time_s, "end_s": time_s + metrics["duration_s"],
                   "count_per_uav": snapshot["count"], "indices": snapshot["indices"], "split": current.copy(),
                   "next_split": problem["next_split"], "aggregate": problem["aggregate"], "channel": channel,
                   "resources": solved["resources"], "metrics": metrics,
                   "remaining_before_j": remaining.copy(), "remaining_after_j": remaining - metrics["uav_energy_j"],
                   "solver_status": solved["status"], "solver_wall_s": solved["wall_s"],
                   "training_wall_s": time.perf_counter() - training_wall if train else 0.0,
                   "train": train_metrics, "validation": validation}
            with (output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**row, "selection": selection, "attempts": attempts}, default=json_value, ensure_ascii=False, allow_nan=False) + "\n")
            history.append(row)
            if problem["aggregate"]:
                last_aggregation = epoch
            # 第五步：该轮成功后按模型时延推进物理时间、扣除 UAV 训练电量，
            # 再把本轮信道加入“已完成”窗口。新增样本直到下次 snapshot 才参与。
            # 保留最近一次实际可行资源作下次参考，且在聚合后才应用 next_split。
            remaining = row["remaining_after_j"].copy()
            time_s = row["end_s"]
            total_energy += metrics["total_energy_j"]
            processed += n * snapshot["count"]
            gain_history.append(channel["gain"].copy())
            reference = copy.deepcopy(solved["resources"])
            if problem["next_split"] is not None:
                current = problem["next_split"].copy()
            print(f"epoch {epoch}/{epochs} D={snapshot['count']} split={row['split'].tolist()} "
                  f"T={metrics['duration_s']:.3f}s E={metrics['total_energy_j']:.3f}J {solved['status']}", flush=True)
    except (ValueError, RuntimeError, FloatingPointError, AssertionError) as exc:
        failure = {"epoch": len(history) + 1, "reason": "execution_error", "type": type(exc).__name__, "message": str(exc)}
    completed = len(history) == epochs and failure is None
    summary = {"status": "completed" if completed else "incomplete", "completed": completed,
               "mode": "training" if train else "cost_simulation", "dataset_kind": data["kind"],
               "strategy": strategy, "epochs_completed": len(history), "epochs_planned": epochs,
               "aggregations_completed": sum(int(row["aggregate"]) for row in history),
               "reference_recovery": config["optimization"]["reference_recovery"],
               "budget": training_budget(config), "start_s": data["start_s"], "finish_s": time_s,
               "training_duration_s": time_s - data["start_s"], "variable_energy_j": total_energy,
               "system_energy_j": config["task"]["fixed_energy_j"] + total_energy,
               "uav_energy_j": sum((r["metrics"]["uav_energy_j"] for r in history), np.zeros(n)),
               "server_energy_j": sum(r["metrics"]["server_energy_j"] for r in history),
               "remaining_j": remaining, "processed_samples": processed,
               "distinct_training_samples": n * history[-1]["count_per_uav"] if history else 0,
               "best_validation_epoch": best_epoch, "best_validation": best_val,
               "last_aggregation_epoch": last_aggregation, "initialization": provenance,
               "data_fingerprint": data["fingerprint"], "hardware_profile_measured": profile["metadata"]["hardware_measured"],
               "failure": failure, "wall_s": time.perf_counter() - start_wall,
               "code_sha256": version["source_sha256"], "code_version": version,
               "comparison_note": "未完成任务不得将低能耗解释为节能；实时时间策略可能使用不同数据快照。"}
    if train:
        save_training_results(config, data, output, summary, global_model, best_model, branches, history)
    summary["wall_s"] = time.perf_counter() - start_wall
    save_history_csv(output, history)
    save_json(output / "summary.json", summary)
    return {"summary": summary, "history": history, "final_model": global_model if completed else None,
            "best_model": best_model, "branches": branches, "output_dir": str(output)}


# 8. 有意义的协议验证：五切点梯度、尾批、聚合通信、可行性与能量下降 ---------------
def _verification_config(config: dict):
    """返回协议验证所用的小规模独立配置。"""
    c = load_config(updates={"num_uavs": 2, "data": {"kind": "synthetic", "image_size": 32,
                            "synthetic_per_uav": 5, "synthetic_eval_size": 3,
                            "initial_samples_per_uav": None, "initial_fraction": 0.6},
                            "training": {"epochs": 4, "batch_size": 2, "allow_random_init": True, "save_checkpoint": False},
                            "uav": {"kappa": 1e-28, "compute_power_max_w": 0.3375, "power_max_w": 1.0,
                                    "cycles_per_flop": 0.5, "energy_cap_j": 200000.0},
                            "optimization": {"max_iterations": 15, "split_objective": "reference_score"},
                            "output_root": config["output_root"]})
    return c


def _verify_split_gradients(model: dict, x: Any, y: Any):
    """核对五个切点的梯度与完整网络梯度，并核对单步更新。"""
    import torch
    from torch.nn import functional as F
    names = list(model["params"])
    logits = forward_suffix(model, forward_prefix(model, x, 0), 0)
    grads = dict(zip(names, torch.autograd.grad(F.cross_entropy(logits, y), list(model["params"].values()))))
    errors = []
    for split in SPLIT_POINTS:
        split_grads, _ = split_gradients(model, x, y, split)
        errors.append(max(float((grads[name] - split_grads[name]).abs().max()) for name in names))
        for name in names:
            torch.testing.assert_close(grads[name], split_grads[name], rtol=2e-5, atol=2e-6)
        stepped = sgd_step(model, split_grads, 0.01)
        for name in names:
            torch.testing.assert_close(stepped["params"][name], model["params"][name] - 0.01 * grads[name], rtol=2e-5, atol=2e-6)
    return errors


def _verify_training_and_aggregation(model: dict, data: dict, c: dict):
    """核对尾批累计步长、输入不变性和按样本数加权聚合。"""
    import torch
    names = list(model["params"])
    original = {name: value.detach().clone() for name, value in model["params"].items()}
    updated, stats = train_client_epoch(model, data, data["clients"][0][:3], 4, 1, 0, c)
    assert math.isclose(stats["step_sum"], c["training"]["epoch_step_size"], rel_tol=1e-12)
    assert any(not torch.equal(updated["params"][name], model["params"][name]) for name in names)
    assert all(torch.equal(original[name], model["params"][name]) for name in names)
    average = aggregate_models([model, updated], [1, 3])
    for name in names:
        torch.testing.assert_close(average["params"][name], model["params"][name] * 0.25 + updated["params"][name] * 0.75)


def _verify_release_and_partitions(data: dict, c: dict):
    """核对轮初样本冻结、周期与旧版释放计划以及集合互斥。"""
    snap = epoch_snapshot(data, data["start_s"], 1, c)
    assert snap["count"] == 3
    later = epoch_snapshot(data, data["release_s"][3], 2, c)
    assert later["count"] == 4 and snap["count"] == 3
    # 检查首次采样、采样间隙、末批截断，以及长任务的自动间隔。
    patrol_config = default_config()
    release, plan = sample_release_schedule(5552, patrol_config)
    initial = plan["initial_count"]
    assert np.count_nonzero(release == 0) == initial
    np.testing.assert_allclose(np.diff(release[initial - 1:]), plan["sample_interval_s"])
    assert math.isclose(release[-1], plan["target_s"])
    manual = merge_config(patrol_config, {"data": {"initial_samples_per_uav": None,
                         "sample_interval_s": 5, "samples_per_capture": 2}})
    manual_release, _ = sample_release_schedule(6, manual)
    np.testing.assert_array_equal(manual_release, [0, 5, 5, 10, 10, 15])
    np.testing.assert_array_equal(np.searchsorted(manual_release, [4.9, 5, 9.9, 10, 20], side="right"),
                                  [1, 3, 3, 5, 6])
    assert np.searchsorted(release, 9 * 600, side="right") < 0.2 * len(release)
    legacy_release, _ = sample_release_schedule(5, merge_config(c, {"data": {"rate_per_s": 1.0}}))
    np.testing.assert_array_equal(legacy_release, [1, 2, 3, 4, 5])
    groups = [set(ids) for ids in data["clients"]] + [set(data["val"]), set(data["test"])]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])


def _verify_heterogeneous_resources(profile: dict, h: np.ndarray, c: dict):
    """核对逐机画像、硬件上限、旧画像兼容性和固定功率可行性。"""
    heterogeneous = load_config(updates={"num_uavs": 2,
        "uav": {"kappa": [1e-28, 1.4e-28], "compute_power_max_w": [0.1, 0.3375],
                "power_max_w": [0.3, 0.8], "cycles_per_flop": [1.0, 0.5], "energy_cap_j": [30.0, 40.0]}})
    hetero_profile = {**profile, "client_cycles": np.array([[2.0], [1.0]]) * profile["client_cycles"]}
    # 每机画像行与自身硬件对应；共享旧画像仍可直接读取。
    legacy = {key: np.arange(1.0, 10.0) for key in
              ("client_cycles", "server_cycles", "activation_bits", "gradient_bits", "prefix_bits")}
    np.testing.assert_array_equal(grouped_profile(legacy)["client_cycles"], [1, 3, 5, 7, 9])
    legacy["client_cycles"] = np.vstack([np.arange(1.0, 10.0), np.arange(1.0, 10.0) * 2])
    np.testing.assert_array_equal(grouped_profile(legacy)["client_cycles"], [[1, 3, 5, 7, 9], [2, 6, 10, 14, 18]])
    hetero_problem = make_problem(1, 3, h, np.array([4, 6]), None, np.array([30.0, 40.0]), heterogeneous)
    hetero_resource = reference_resources(heterogeneous)
    metrics = evaluate_resources(hetero_problem, hetero_resource, hetero_profile, heterogeneous)
    np.testing.assert_allclose(metrics["loads"]["client_cycles"],
                               3 * profile["client_cycles"][[2, 3]] * [2, 1])
    for key, bounds in (("frequency_hz", maximum_frequency(heterogeneous)),
                        ("power_w", uav_parameter(heterogeneous, "power_max_w"))):
        invalid = {k: v.copy() for k, v in hetero_resource.items()}
        invalid[key][0] = bounds[0] * 1.05
        audit = audit_resources(hetero_problem, invalid, hetero_profile, heterogeneous)
        assert not audit["ok"] and math.isclose(audit["normalized_violation"], 0.05)
    phase = convex_subproblem(hetero_problem, hetero_profile, heterogeneous, fixed_power=0.3)
    assert phase["ok"] and audit_resources(hetero_problem, phase["resources"], hetero_profile, heterogeneous)["ok"]
    assert reference_scores(3, h, hetero_resource, hetero_profile, True, heterogeneous).shape == (2, 5)


def _verify_stage_costs_and_solver(profile: dict, h: np.ndarray, remaining: np.ndarray, reference: dict, c: dict):
    """核对各轮阶段时延、同步载荷和求解后的可行性与能量下降。"""
    results = []
    for epoch, next_split in ((1, None), (2, np.array([8, 2])), (4, None)):
        problem = make_problem(epoch, 3, h, np.array([0, 4]), next_split, remaining, c)
        m = evaluate_resources(problem, reference, profile, c)
        mu = profile["prefix_bits"][problem["split"] // BLOCKS_PER_GROUP] / m["up_rate_bps"] if problem["aggregate"] else np.zeros(2)
        md = profile["prefix_bits"][next_split // BLOCKS_PER_GROUP] / m["down_rate_bps"] if next_split is not None else np.zeros(2)
        aggregate_time = c["server"]["aggregation_time_s"] if problem["aggregate"] else 0.0
        correct_duration = float(np.max(m["branch_training_s"]) + np.max(mu) + np.max(md) + aggregate_time)
        assert math.isclose(m["duration_s"], correct_duration, rel_tol=1e-12)
        if next_split is not None:
            wrong_combined_max = float(np.max(m["branch_training_s"] + mu + md) + aggregate_time)
            assert correct_duration > wrong_combined_max + 1e-3
        assert (m["loads"]["download_bits"].sum() == 0) == (next_split is None)
        if next_split is not None:
            np.testing.assert_array_equal(m["loads"]["download_bits"], profile["prefix_bits"][next_split // BLOCKS_PER_GROUP])
        solution = solve_resources(problem, profile, c, reference)
        assert solution["ok"], solution
        assert audit_resources(problem, solution["resources"], profile, c)["ok"]
        accepted = [r["energy_j"] for r in solution["trace"] if r.get("accepted", True)]
        assert all(new <= old for old, new in zip(accepted, accepted[1:]))
        results.append({"epoch": epoch, "status": solution["status"], "energy_j": solution["metrics"]["total_energy_j"]})
    return results


def _verify_infeasibility_and_battery(profile: dict, h: np.ndarray, remaining: np.ndarray, reference: dict, c: dict):
    """核对不可能任务被拒绝，以及收紧电池时的约束求解。"""
    # 低电池与不可能期限必须拒绝，不能静默返回参考资源。
    problem = make_problem(1, 3, h, np.array([0, 4]), None, np.zeros(2), c)
    assert not audit_resources(problem, reference, profile, c)["ok"]
    impossible = merge_config(c, {"task": {"epoch_deadline_s": 1e-6}})
    rejected = solve_resources(problem, profile, impossible)
    assert not rejected["ok"]
    # 仅收紧电量，让初始参考资源失效，验证电池约束确实进入凸求解。
    p = make_problem(1, 3, h, np.array([0, 4]), None, remaining, c)
    reference_energy = evaluate_resources(p, reference, profile, c)["uav_energy_j"]
    tight = make_problem(1, 3, h, np.array([0, 4]), None, reference_energy * 0.6, c)
    assert not audit_resources(tight, reference, profile, c)["ok"]
    tight_solution = solve_resources(tight, profile, c)
    assert tight_solution["ok"], tight_solution
    assert audit_resources(tight, tight_solution["resources"], profile, c)["ok"]


def _verify_reference_recovery_and_ranking(profile: dict, h: np.ndarray, remaining: np.ndarray, c: dict):
    """核对严格参考失败、显式参考恢复和全部切分组合排序。"""
    # 使用一个过慢的固定参考，严格 P1 应拒绝，显式重建可恢复参考可行性。
    slow = reference_resources(c)
    slow["frequency_hz"][:] = 1000.0
    selection = select_splits(2, 3, np.array([0, 4]), [h], h, remaining, slow, profile, c)
    assert not selection["ok"] and selection["status"] == "no_reference_feasible_split"
    rebuilt = rebuild_reference(2, 3, np.array([0, 4]), h, remaining, profile, c)
    assert rebuilt["ok"]
    selection = select_splits(2, 3, np.array([0, 4]), [h], h, remaining, rebuilt["resources"], profile, c)
    assert selection["ok"]
    assert selection["combinations_evaluated"] == len(SPLIT_POINTS) ** c["num_uavs"]
    assert {tuple(row["vector"]) for row in selection["trace"]} == set(product(SPLIT_POINTS, repeat=c["num_uavs"]))
    ranked = [tuple(v) for v in selection["candidates"]]
    feasible_rows = [row for row in selection["trace"] if row["feasible"]]
    assert selection["scores_j"].shape == (c["num_uavs"], len(SPLIT_POINTS))
    expected = sorted(feasible_rows, key=lambda row: (row["score_j"], row["vector"] != (0, 4), row["vector"]))
    assert ranked == [row["vector"] for row in expected]


def _verify_budget_and_channel(c: dict):
    """核对训练预算取整和信道计算的确定性。"""
    assert training_budget(default_config())["conditional_epochs"] == 100
    # 相同原始轨迹时刻必须产生完全相同的信道，不依赖调用顺序。
    np.testing.assert_array_equal(channel_at(20, c)["gain"], channel_at(20, c)["gain"])


def _verify_absolute_initial_samples(config: dict) -> dict:
    """独立核对默认500优先于比例，以及首次新增和全量释放的快照边界。"""
    release, plan = sample_release_schedule(5552, config)
    assert plan["initial_count"] == 500
    assert np.count_nonzero(release == 0) == 500
    assert release[499] == 0 < release[500]
    data = {"release_s": release, "capacity": 5552,
            "clients": [list(range(5552)) for _ in range(4)]}
    initial = epoch_snapshot(data, 0.0, 1, config)
    before = epoch_snapshot(data, float(np.nextafter(release[500], -np.inf)), 2, config)
    after = epoch_snapshot(data, float(release[500]), 2, config)
    final = epoch_snapshot(data, float(release[-1]), 120, config)
    assert initial["count"] == before["count"] == 500
    assert after["count"] == 501 and final["count"] == 5552
    assert all(len(indices) == 500 for indices in initial["indices"])
    full_release, full_plan = sample_release_schedule(500, config)
    np.testing.assert_array_equal(full_release, np.zeros(500))
    assert full_plan["initial_count"] == 500
    # 旧恒速模式不提前释放500个样本；启动等待仍由第500个到达时刻决定。
    rate_config = merge_config(config, {"data": {"rate_per_s": 2.0}})
    rate_release, rate_plan = sample_release_schedule(501, rate_config)
    assert rate_plan["initial_count"] == 500 and rate_release[499] == 250.0
    return {"initial_count": initial["count"], "capacity": 5552,
            "first_new_sample_s": float(release[500]), "full_release_s": float(release[-1])}

def _verify_initial_capacity_and_small_data(config: dict):
    """不足500的正式容量应拒绝；显式回退比例的小合成数据仍能准备。"""
    for capacity in (499, 0, -1, 1.5, True):
        try:
            sample_release_schedule(capacity, config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法或小于默认初始量的容量未被拒绝: {capacity!r}")
    small = load_config(updates={"data": {"kind": "synthetic", "initial_samples_per_uav": None,
        "initial_fraction": 0.6, "synthetic_per_uav": 5, "synthetic_eval_size": 3},
        "training": {"epochs": 4}})
    data = prepare_data(small)
    assert data["capacity"] == 5 and data["release_plan"]["initial_count"] == 3
    assert epoch_snapshot(data, data["start_s"], 1, small)["count"] == 3
    assert default_config(2)["trajectory"]["kind"] == "circle"

def _verify_serpentine_lengths_and_period(config: dict, geometry: dict) -> dict:
    """以独立解析值及黑箱积分核对四机闭合长度和共同周期。"""
    # 六段480m横扫、五段84m连接、420m闭合；12个R10圆角替换直角。
    expected_length = 3480.0 + 60.0 * math.pi
    # 直线峰值10、转弯5、切向加速度2：24次速度过渡额外耗时15s。
    expected_period = 348.0 + 12.0 * math.pi + 15.0
    assert math.isclose(geometry["raw_length_m"], 3720.0, rel_tol=1e-12)
    assert math.isclose(geometry["length_m"], expected_length, rel_tol=1e-12)
    assert math.isclose(geometry["period_s"], expected_period, rel_tol=1e-12)
    period = geometry["period_s"]
    for t in (0.0, 0.017 * period, 0.41 * period, period - 1e-4):
        np.testing.assert_allclose(patrol_positions(t, config), patrol_positions(t + period, config),
                                   rtol=0, atol=1e-8)
    np.testing.assert_allclose(patrol_positions(-1e-4, config),
                               patrol_positions(period - 1e-4, config), rtol=0, atol=1e-8)
    times = np.linspace(0.0, period, 4097)
    positions = np.stack([patrol_positions(float(t), config) for t in times])
    assert positions.shape == (4097, 4, 3) and np.isfinite(positions).all()
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=2).sum(axis=0)
    # 有限采样的折线略短于圆弧；容差只用于此独立数值积分。
    np.testing.assert_allclose(lengths, expected_length, rtol=0, atol=0.15)
    np.testing.assert_allclose(lengths, lengths[0], rtol=0, atol=0.02)
    return {"raw_length_m": 3720.0, "rounded_length_m": expected_length,
            "period_s": expected_period, "sampled_lengths_m": lengths.tolist()}

def _verify_serpentine_rotation_and_phase(config: dict, geometry: dict):
    """核对象限与同高边界、旋转关系，以及错相只打破同步的信道序列。"""
    period = geometry["period_s"]
    zero_phase = merge_config(config, {"trajectory": {"phase_offsets_fraction": [0.0] * 4}})
    phase = config["trajectory"]["phase_offsets_fraction"]
    assert phase == [0.0, 0.25, 0.5, 0.75]
    gains = []
    for t in np.linspace(0.0, period, 129):
        positions = patrol_positions(float(t), config)
        assert np.all(np.abs(positions[:, :2]) <= 500.0 + 1e-8)
        np.testing.assert_allclose(positions[:, 2], 100.0, rtol=0, atol=1e-10)
        base = patrol_positions(float(t), zero_phase)[0, :2]
        x, y = base
        rotations = np.array([[x, y], [-y, x], [-x, -y], [y, -x]])
        np.testing.assert_allclose(patrol_positions(float(t), zero_phase)[:, :2],
                                   rotations, rtol=0, atol=1e-8)
        for uid in range(4):
            shifted = patrol_positions(float(t + phase[uid] * period), zero_phase)
            np.testing.assert_allclose(positions[uid], shifted[uid], rtol=0, atol=1e-8)
        inverse_xy = np.array([[positions[0, 0], positions[0, 1]],
                               [positions[1, 1], -positions[1, 0]],
                               [-positions[2, 0], -positions[2, 1]],
                               [-positions[3, 1], positions[3, 0]]])
        assert np.all((inverse_xy[:, 0] >= 10.0 - 1e-8) & (inverse_xy[:, 0] <= 490.0 + 1e-8))
        assert np.all((inverse_xy[:, 1] >= 40.0 - 1e-8) & (inverse_xy[:, 1] <= 460.0 + 1e-8))
        for a in range(4):
            for b in range(a):
                assert np.linalg.norm(positions[a] - positions[b]) >= 50.0 - 1e-8
        gains.append(channel_at(float(t), config)["gain"])
    gains = np.stack(gains)
    for a in range(4):
        for b in range(a):
            assert not np.allclose(gains[:, a], gains[:, b], rtol=1e-7, atol=0)
    same_phase_gain = channel_at(0.173 * period, zero_phase)["gain"]
    np.testing.assert_allclose(same_phase_gain, same_phase_gain[0], rtol=1e-12, atol=0)

def _verify_serpentine_rounded_motion(config: dict, geometry: dict):
    """在全部线弧接点和周期接缝两侧检查位置与速度连续、转弯半径和速度。"""
    eps = 1e-4
    segments = geometry["segments"]
    np.testing.assert_allclose(segments[-1]["end_xy_m"], segments[0]["start_xy_m"], atol=1e-10)
    for uid, phase_s in enumerate(geometry["phase_offsets_s"]):
        for join_s in [0.0] + [segment["end_time_s"] for segment in segments]:
            t = join_s - phase_s
            before = patrol_positions(t - eps, config)[uid]
            at = patrol_positions(t, config)[uid]
            after = patrol_positions(t + eps, config)[uid]
            left_velocity, right_velocity = (at - before) / eps, (after - at) / eps
            assert np.isfinite(left_velocity).all() and np.isfinite(right_velocity).all()
            assert np.linalg.norm(left_velocity) <= 10.0 + 1e-5
            assert np.linalg.norm(right_velocity) <= 10.0 + 1e-5
            assert np.linalg.norm(left_velocity - right_velocity) < 0.01
    zero_phase = merge_config(config, {"trajectory": {"phase_offsets_fraction": [0.0] * 4}})
    arcs = [segment for segment in segments if "radius_m" in segment]
    assert len(arcs) == 12
    for arc in arcs:
        t = (arc["start_time_s"] + arc["end_time_s"]) / 2
        position = patrol_positions(t, zero_phase)[0, :2]
        velocity = (patrol_positions(t + eps, zero_phase)[0, :2]
                    - patrol_positions(t - eps, zero_phase)[0, :2]) / (2 * eps)
        radial = position - arc["center_xy_m"]
        assert math.isclose(float(np.linalg.norm(radial)), 10.0, rel_tol=1e-9)
        assert math.isclose(float(np.linalg.norm(velocity)), 5.0, rel_tol=1e-7)
        assert abs(float(np.dot(radial, velocity))) < 1e-5

def _verify_default_patrol_and_initial_samples() -> tuple[dict, dict]:
    """验证新默认场景；独立于用于原21项检查的两机圆轨迹配置。"""
    config = load_config()
    assert config["trajectory"]["kind"] == "serpentine_equal"
    geometry = patrol_geometry(config)
    trajectory_report = _verify_serpentine_lengths_and_period(config, geometry)
    _verify_serpentine_rotation_and_phase(config, geometry)
    _verify_serpentine_rounded_motion(config, geometry)
    release_report = _verify_absolute_initial_samples(config)
    _verify_initial_capacity_and_small_data(config)
    return trajectory_report, release_report


def verify_implementation(config: dict) -> dict:
    """依次执行小型 CPU 协议验证；失败抛 AssertionError，不作为实验准确率。"""
    import torch
    c = _verification_config(config)
    torch.set_num_threads(c["training"]["torch_threads"])
    model, _ = initialize_model(c)
    data = prepare_data(c)
    x, y = load_batch(data, data["clients"][0][:2], "cpu")
    errors = _verify_split_gradients(model, x, y)
    _verify_training_and_aggregation(model, data, c)
    _verify_release_and_partitions(data, c)
    profile = profile_splits(model, c)
    remaining = np.full(2, c["uav"]["energy_cap_j"])
    h = channel_at(data["start_s"], c)["gain"]
    reference = reference_resources(c)
    _verify_heterogeneous_resources(profile, h, c)
    results = _verify_stage_costs_and_solver(profile, h, remaining, reference, c)
    _verify_infeasibility_and_battery(profile, h, remaining, reference, c)
    _verify_reference_recovery_and_ranking(profile, h, remaining, c)
    _verify_budget_and_channel(c)
    trajectory_report, release_report = _verify_default_patrol_and_initial_samples()
    return {"passed": True, "verification_config": c, "split_gradient_max_abs_errors": errors,
            "checks": ["five_split_gradients_and_updates", "exhaustive_split_ranking", "input_immutability", "tail_batch_step_sum",
                       "weighted_aggregation", "frozen_epoch_snapshot", "periodic_capture_release", "legacy_rate_release", "disjoint_partitions",
                       "separate_stage_maxima", "new_prefix_payload", "final_zero_download",
                       "convex_original_feasibility", "accepted_energy_monotonicity", "infeasible_rejection",
                       "binding_battery_constraint", "strict_reference_failure_and_explicit_rebuild", "budget_rounding",
                       "per_uav_cycles_and_limits", "per_uav_phase_one", "legacy_and_grouped_profiles",
                       "serpentine_equal_length_period_and_wrap", "serpentine_rotation_phase_and_quadrants",
                       "serpentine_rounded_motion_continuity", "absolute_initial_500_snapshot_boundaries",
                       "initial_capacity_and_small_synthetic_compatibility"],
            "solver_cases": results, "trajectory_verification": trajectory_report,
            "initial_sample_verification": release_report}


def build_argument_parser():
    """构造七个子命令共用的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["config", "prepare", "profile", "simulate", "train", "check", "smoke"])
    parser.add_argument("--config", help="JSON 局部参数覆盖文件")
    parser.add_argument("--output", help="本次运行的新目录；不得覆盖已有 config.json")
    parser.add_argument("--checkpoint", help="ResNet18 本地预训练参数路径")
    parser.add_argument("--allow-random-init", action="store_true", help="显式允许随机初始化，仅用于诊断/指定的消融")
    parser.add_argument("--device", metavar="DEVICE", help="模型设备：auto、cpu、cuda 或 cuda:N；覆盖 training.device")
    return parser


def load_cli_config(args: argparse.Namespace):
    """读取 JSON 和显式命令行覆盖，保持原参数优先级。"""
    updates = {}
    if args.checkpoint or args.allow_random_init or args.device is not None:
        updates["training"] = {}
        if args.checkpoint:
            updates["training"]["checkpoint"] = str(Path(args.checkpoint).resolve())
        if args.allow_random_init:
            updates["training"]["allow_random_init"] = True
        if args.device is not None:
            updates["training"]["device"] = args.device
    return load_config(args.config, updates)


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


def run_experiment_command(args: argparse.Namespace, config: dict):
    """执行仿真、正式训练或 smoke，打印摘要并返回命令退出码。"""
    if args.command == "smoke":
        config = smoke_config(config)
    result = run_experiment(config, train=args.command != "simulate", output=args.output, label=args.command)
    report = {key: result["summary"][key] for key in ("status", "mode", "dataset_kind", "epochs_completed", "epochs_planned", "system_energy_j", "processed_samples")}
    report["failure_reason"] = result["summary"]["failure"]["reason"] if result["summary"]["failure"] else None
    print(json.dumps({"output_dir": str(Path(result["output_dir"]).resolve()), "summary": report}, default=json_value, ensure_ascii=False, indent=2))
    return 0 if result["summary"]["completed"] else 2


def run_preparation_command(args: argparse.Namespace, config: dict):
    """执行并存档 check、prepare 或 profile 的结果。"""
    requested_device = None
    if args.command == "profile":
        requested_device = config["training"]["device"]
        config["training"]["device"] = resolve_training_device(requested_device)
    with experiment_output(config, args.command, args.output, training_device_requested=requested_device) as (output, metadata):
        if args.command == "check":
            report = verify_implementation(config)
            save_json(output / "verification.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            data = prepare_data(config)
            save_json(output / "data_manifest.json", data)
            print(json.dumps({"output": str(output), "capacity_per_uav": data["capacity"], "audit": data["audit"],
                              "class_counts": data["class_counts"], "val": len(data["val"]), "test": len(data["test"])}, default=json_value, ensure_ascii=False, indent=2))
        else:
            import torch
            torch.set_num_threads(config["training"]["torch_threads"])
            model, _ = initialize_model(config)
            save_json(output / "split_profiles.json", profile_splits(model, config))
            print(f"模型画像已保存到 {output}")
    return 0


def main() -> int:
    """命令行入口：解析参数、加载配置，再分派到相应的小函数。"""
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.device is not None and args.command in ("check", "prepare"):
        parser.error(f"{args.command} 不接受 --device，请用 smoke 验证训练设备。")
    config = load_cli_config(args)
    if args.command == "config":
        print(json.dumps({"config": config, "training_budget": training_budget(config)}, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("simulate", "train", "smoke"):
        return run_experiment_command(args, config)
    return run_preparation_command(args, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, AssertionError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)
