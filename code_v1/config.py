"""连续飞行 UAV-SFL 场景的统一配置、参数校验和条件性训练预算。

运行命令：在本文件所在目录执行 ``python -X utf8 -B config.py``。
该命令读取本文件的 CONFIG，先打印完整配置 JSON，再打印条件性训练预算；
不写出文件、不裁剪图片、不生成环境，也不启动模型训练。

输入有两种方式：直接修改 CONFIG；或由调用方使用 ``load_config(JSON路径)``
加载局部覆盖。场景入口也支持 ``prepare_scenario.py --config 配置.json``。
JSON 只需提供需要改变的字段，但不能加入未知字段；JSON 内的相对路径以该
JSON 所在目录为基准。直接运行本文件不解析 JSON 命令行参数。

各模块共享本文件返回的配置和采集计数约定。时间单位为秒、坐标为米，频率和
带宽为 Hz，功率为 W，能量为 J；理论参数使用所选损失和模型参数的归一化尺度。
本模块仅依赖 Python 标准库，读取配置和环境回放不要求 NumPy、Pillow 或 PyTorch。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

# 所有默认输出路径锚定在 code_v1 内，不随启动命令的工作目录改变。
ROOT = Path(__file__).resolve().parent


def default_data_root():
    """返回原始 CBCL 压缩包目录的默认路径字符串，不读取或解压图片。

    从 code_v1 的各级父目录向外查找 ``DataSet/CBCL StreetScenes``，仅当
    Images.zip 和 Annotations.zip 都存在时采用该目录。找不到时返回本项目
    datasets 下的候选路径，供用户修改；返回路径并不代表数据已准备完成。
    """
    for parent in ROOT.parents:
        candidate = parent / "DataSet" / "CBCL StreetScenes"
        if all((candidate / name).is_file() for name in ("Images.zip", "Annotations.zip")):
            return str(candidate)
    return str(ROOT / "datasets" / "CBCL StreetScenes")


# 唯一默认参数表。load_config 会深拷贝它，避免一次 JSON 覆盖污染后续调用。
# 下列硬件、信道和理论数值是场景输入；实际设备或数据集上的有效性需另行标定。
CONFIG = {
    # 固定的飞行计划：无人机持续移动，训练及同步等待不改变既定航程。
    "mission": {
        # 无人机数量 N；duration_s 是整趟飞行的物理截止时间，单位秒。
        "num_uavs": 5, "duration_s": 600.0,
        # 工作区域的水平坐标边界，单位米；环境模块在此区域构造固定巡逻路线。
        "x_min_m": -500.0, "x_max_m": 500.0,
        "y_min_m": -500.0, "y_max_m": 500.0,
        # 固定飞行高度（米）和沿航线运动的速度（米/秒），不是训练决策变量。
        "altitude_m": 100.0, "speed_m_s": 10.0,
        # lane_spacing_m 控制巡逻线间距（米）；trace_interval_s 为物理时间
        # 轴上的环境观测间隔（秒），不是 epoch 时长、batch 时长或悬停时间。
        "lane_spacing_m": 100.0, "trace_interval_s": 0.1,
    },
    # 原始数据、按场景隔离的评价集以及各 UAV 固定采集序列的配置。
    "data": {
        # root 内放 Images.zip/Annotations.zip；output 保存裁剪 PNG 和 data.json。
        "root": default_data_root(), "output": str(ROOT / "output" / "data"),
        # image_size 是方形 RGB 裁剪的边长（像素）；val/test 是原始场景的
        # 划分比例，不保证裁剪后的样本数量也恰好按同一比例划分。
        "image_size": 64, "val_fraction": 0.15, "test_fraction": 0.15,
        # 数据划分与固定采集排列的随机种子，与初始化种子、信道种子分开使用。
        "seed": 42,
        # None：将实际训练池按 N 等量划分，不能均分的尾部不使用；正整数：
        # 每架 UAV 的统一容量 D_max。样本不放回复制，各机使用不同训练索引。
        "samples_per_uav": None,
        # 初始缓存 D0 已包含在 D_max 中。显式整数优先；None 时取
        # max(1, floor(D_max * initial_fraction))，随后检查 D0 不超过容量。
        "initial_samples_per_uav": None, "initial_fraction": 0.1,
        # 各机共用的采集速率 lambda，单位样本/秒；None 时按剩余容量除以
        # 任务时长推定，0 表示不再加入新样本。达到容量后累计数量保持不变。
        # epoch 只使用轮初快照；最后才到达且未进入快照的样本不参与本次训练。
        "sample_rate_per_s": None,
    },
    # 模型训练接口约定；准备数据或生成环境不会据此自动运行 100 个 epoch。
    "training": {
        # K_tot 必须为聚合间隔 K 的整数倍。epoch_step_size 是每个完整 epoch
        # 的累计步长 eta，不是每个 batch 都原样使用的学习率。
        # batch 的实际步长 = eta * 实际 batch 样本数 / 本 epoch 冻结样本数 D_k；
        # 尾批按实际大小归一化，因此新增样本不会增大单 epoch 的累计步长。
        "total_epochs": 100, "aggregation_period": 2, "epoch_step_size": 0.18,
        # 单次训练/评价的 batch 样本数上限；不是 epoch 的样本总数。
        "batch_size": 32, "eval_batch_size": 64,
        # 当前理论协议限定为无动量、无权重衰减的普通 SGD。
        "optimizer": "sgd", "momentum": 0.0, "weight_decay": 0.0,
        # constant 表示 eta 不随 epoch 调度；frozen 固定 BN 行为。
        # initialization_seed 用于各方案共享的模型初始化，非数据划分种子。
        "scheduler": "constant", "bn": "frozen", "initialization_seed": 42,
        # initialization_checkpoint=None 表示没有指定外部初始化权重路径；
        # device="auto" 交由训练调用方选择设备；workers 是数据加载进程数，0 为主进程。
        "initialization_checkpoint": None, "device": "auto", "workers": 0,
        # 可选的离线切分画像路径；None 表示尚未指定外部画像，并非画像已标定。
        "split_profile_path": None,
    },
    # 每架 UAV 共用的硬件参数和上行资源边界。
    "uav": {
        # 计算频率的下界、上界及切分评分参考值，单位 Hz。
        "f_min_hz": 2e8, "f_max_hz": 1.5e9, "f_ref_hz": 1e9,
        # kappa 为动态计算能耗系数，用于 E=kappa*C*f^2（J）；C 为等效周期数。
        # cycles_per_flop 将 FLOP 画像换算为周期数，需与实测硬件计量一致。
        # 此处的硬件 kappa 与 theory_budget 内无量纲的收敛系数 kappa 含义不同。
        "kappa": 1e-28, "cycles_per_flop": 0.5,
        # 上行发射功率下界、上界和切分评分参考值，单位 W；不是推进功率。
        "p_up_min_w": 0.01, "p_up_max_w": 1.0, "p_up_ref_w": 0.5,
    },
    # 地面服务器位置固定；UAV 的位置仍沿轨迹随物理时间变化。
    "server": {
        # 服务器三维坐标 [x,y,z]，单位米；无人机飞行高度必须高于此 z 坐标。
        "position_m": [0.0, 0.0, 0.0],
        # parallel_branches=None 时解析为 N 个独立后缀分支；显式值不得小于 N。
        # frequency_hz 是每个分支的固定计算频率（Hz），不是全部分支的共享总频率。
        "parallel_branches": None, "frequency_hz": 3e9,
        # 与 UAV 侧相同的 FLOP/周期换算及动态计算能耗系数，采用服务器自己的标定值。
        "cycles_per_flop": 0.03, "kappa": 1e-28,
        # downlink_power_w 为每条下行链路的固定功率（W）；总功率上限 None
        # 时由环境解析为 N * downlink_power_w，显式总上限须足够覆盖所有链路。
        "downlink_power_w": 2.0, "max_downlink_power_w": None,
        # 每次完整模型聚合的固定时延（秒）和计算功率（W）。
        "aggregation_time_s": 0.01, "aggregation_power_w": 10.0,
    },
    # 上、下行各自的总带宽及每条 UAV 链路的最低带宽，单位均为 Hz。
    # 每个方向都要求 N * 最低带宽 <= 总带宽，不将上下行预算相互替代。
    "radio": {
        "total_up_hz": 2e7, "total_down_hz": 2e7,
        "min_up_hz": 1e5, "min_down_hz": 1e5,
    },
    # 在共享物理时间网格上生成的信道观测参数；查询环境不会重新抽样。
    # 信道由观测时刻的移动 UAV 坐标与固定服务器坐标决定几何量，不能把
    # 服务器位置固定误解为 UAV 悬停。当前位置插值与最新已观测 CSI 分开查询。
    "channel": {
        # seed 固定整条环境随机过程；carrier_hz 为载波频率（Hz）；
        # noise_psd_w_hz 为噪声功率谱密度（W/Hz），带宽内噪声需再乘带宽。
        "seed": 42, "carrier_hz": 2.4e9, "noise_psd_w_hz": 3.981e-21,
        # los_a/los_b 为仰角采用度数时的 LoS 概率模型参数；损耗单位 dB。
        "los_a": 9.61, "los_b": 0.16, "los_loss_db": 1.0,
        # nlos_loss_db 是 NLoS 附加损耗；rician_k 为 LoS 莱斯因子，使用
        # 线性比值而非 dB。环境中的 NLoS 分支采用 K=0 的瑞利衰落。
        "nlos_loss_db": 20.0, "rician_k": 10.0,
    },
    # 周期边界的离散切分选择参数；连续资源仍使用执行时的当前 CSI。
    "selection": {
        # candidates 是 P0..P8 的非空子集；initial_split 必须包含在其中。
        "candidates": [f"P{i}" for i in range(9)], "initial_split": "P4",
        # 窗口长度是最近已观测 CSI 的条数，不是秒数或未来预测窗口。
        # max_sweeps 是候选坐标扫描次数；energy_tolerance_j 为能耗比较容差（J）。
        "channel_window_size": 20, "max_sweeps": 1, "energy_tolerance_j": 1e-9,
    },
    "energy": {
        # 每架 UAV 共用的电量参数，单位 J。None 表示尚未校准，不能解释成 0。
        # 仅准备数据/环境时可保留 None；执行训练电量核算前必须全部填全。
        # initial_energy_j 为任务初始电量；flight_energy_j 为既定航程的固定
        # 推进总能量，不随训练同步等待时长改变，不另加训练悬停能耗。
        "initial_energy_j": None, "flight_energy_j": None,
        # fixed_energy_j 预留识别、采集和基础运行等固定开销；reserve_energy_j
        # 为安全储备。两者不再包含上行两个字段已计入的能量，避免重复扣除。
        "fixed_energy_j": None, "reserve_energy_j": None,
    },
    "theory": {
        # 条件性归一化配置，不代表这些界已在具体网络或实际数据上验证。
        # L：逐样本损失梯度的 Lipschitz 常数；G：相关模型点处的样本梯度范数上界。
        # V：周期内按 UAV、epoch 和样本加权的参考梯度围绕周期经验均值的
        # 二阶离差上界，量纲为梯度范数平方，不是分类准确率的方差。
        # rho：周期参考经验平均梯度与固定总体梯度之间的范数偏差上界；
        # 需覆盖实际数据快照和相关模型点，IID 与历史复用不自动令 rho=0。
        "L": 1.0, "G": 1.0, "V": 0.02, "rho": 0.002,
        # delta：F(v0)-F_inf 的非负上界，属于损失尺度，不是置信概率。
        # epsilon：聚合模型平均期望梯度范数平方的目标上界，不是识别错误率，
        # 也不直接保证最后一个聚合模型的准确率。
        "delta": 0.10, "epsilon": 0.05,
    },
    # 共享场景输出目录，保存环境 CSV、场景 JSON 和巡逻路线 SVG；本文件不写出它们。
    "output": {"scenario_dir": str(ROOT / "output" / "scenario")},
}


def _merge(base, override, prefix=""):
    """将 JSON 局部配置递归合并到 base，原地修改 base，返回 None。

    override 必须为字典；只允许 base 已有的键，嵌套分组仍按字典递归。
    prefix 仅用于在错误信息中定位完整字段名。load_config 传入的是深拷贝，
    因而此处的原地修改不会改变全局 CONFIG。
    """
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
    """返回经过校验的独立配置字典，可选读取 UTF-8/带 BOM 的 JSON 覆盖。

    path=None 时使用 CONFIG；提供路径时，仅覆盖 JSON 中出现的已有字段。
    数据、输出、初始化权重及画像的相对路径均以该 JSON 所在目录为基准，
    路径字符串中的 ~ 会展开。此函数不检查图片内容、不启动数据准备，
    参数结构或数值不合法时由合并/校验步骤抛出 ValueError。
    """
    # 第一步：隔离本次配置，确保调用者修改返回值不会影响默认参数。
    config = copy.deepcopy(CONFIG)
    if path is not None:
        source = Path(path).expanduser().resolve()
        _merge(config, json.loads(source.read_text(encoding="utf-8-sig")))
        # 第二步：只解析规定的路径字段；None 保持其“未指定”的语义。
        path_fields = [("data", "root"), ("data", "output"), ("output", "scenario_dir"),
                       ("training", "initialization_checkpoint"), ("training", "split_profile_path")]
        for section, key in path_fields:
            value = config[section][key]
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{section}.{key} 需要非空路径字符串")
                resolved = Path(value).expanduser()
                config[section][key] = str((source.parent / resolved).resolve())
    # 第三步：字段和值都通过统一契约后才返回，避免各调用方自行猜测默认值。
    validate_config(config)
    return config


def _number(value, name, *, minimum=0.0, inclusive=False):
    """校验有限数值下界，成功返回 None，不接受布尔值或 NaN/无穷大。

    inclusive=False 要求 value > minimum；True 时允许等号。name 仅用于
    报错定位。参数单位由对应配置字段约定，此处不执行单位换算。
    """
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} 必须为有限数且 {relation} {minimum}")


def _integer(value, name, minimum=1):
    """校验不小于 minimum 的整数，成功返回 None；显式拒绝布尔值。

    默认适用于正样本数、epoch 数等；随机种子和 workers 使用 minimum=0。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数")


def validate_config(config):
    """只读校验统一配置的字段、类型、物理边界及跨字段约束，返回 None。

    配置分组和键必须与 CONFIG 的结构完全一致。检查包含飞行区域、数据比例、
    普通 SGD 协议、硬件资源、带宽/功率容量及路径隔离；发现冲突抛出 ValueError。
    energy 允许暂未校准的 None，仅在四项齐全时检查剩余训练电量。该函数不
    验证原始数据存在，也不证明理论参数界成立；充分预算条件由 theory_budget
    另行检查。校验不推进物理时钟，也不修改传入配置。
    """
    # 第一步：先检查完整结构，防止拼写错误静默地成为未使用的新参数。
    required = set(CONFIG)
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"配置顶层字段必须为 {sorted(required)}")
    if any(not isinstance(config[key], dict) for key in required):
        raise ValueError("各配置分组必须为字典")
    for section in required:
        if set(config[section]) != set(CONFIG[section]):
            raise ValueError(f"{section} 配置字段不完整或含未知字段")
    # 第二步：检查物理任务和采集参数；场景时长与训练 epoch 数是不同的量。
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
    # 第三步：保持理论对应的更新协议，禁止把 batch 学习率或动量设定混入。
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
    # 第四步：检查机载和服务器硬件，参考资源必须位于允许边界内。
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
    # 第五步：最低带宽应能同时分给所有 UAV；上下行分别检查。
    radio = config["radio"]
    for key, value in radio.items():
        _number(value, f"radio.{key}")
    for direction in ("up", "down"):
        if mission["num_uavs"] * radio[f"min_{direction}_hz"] > radio[f"total_{direction}_hz"]:
            raise ValueError(f"{direction}方向总带宽不足以分配最低链路带宽")
    # 第六步：检查信道和切分参数；只允许已有模型接口支持的 P0..P8。
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
    # 第七步：保留未知电量的 None，但已填写的值必须有效；不能把未知值当零。
    for key, value in config["energy"].items():
        if value is not None:
            _number(value, f"energy.{key}", inclusive=True)
    if all(v is not None for v in config["energy"].values()):
        training_energy_capacity(config)
    # 理论参数在此只检查数值域，收敛步长及误差底项约束留给预算函数。
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
    """从已准备的 data.json 字典解析共同采集计划，返回纯参数字典。

    manifest 中各客户端的索引是训练子集内的局部索引，必须等量、不重叠且
    恰好覆盖保留的训练池。真实容量只取清单，不重新划分、补齐或抽取样本；
    如配置显式给出 samples_per_uav，则必须与该容量一致。

    返回 num_uavs、max_samples_per_uav、initial_samples_per_uav（整数）、
    rate_samples_per_s（样本/秒）和 duration_s（秒），供环境和快照模块共用。
    清单内容/指纹的完整验证由数据或场景入口负责；本函数校验容量与采集契约。
    不修改 config、manifest 或固定采集排列；契约冲突时抛出 ValueError。
    """
    validate_config(config)
    count = config["mission"]["num_uavs"]
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("需要schema=1的data.json")
    clients, settings = manifest.get("clients"), manifest.get("settings", {})
    if (not isinstance(clients, list) or len(clients) != count
            or settings.get("uavs") != count or settings.get("partition") != "iid"):
        raise ValueError("数据清单必须与无人机数量一致且采用IID划分")
    # 第一步：验证 UAV 编号、样本数和局部索引，不从清单外虚构可采样数量。
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
    # 第二步：各索引恰好覆盖一次后，单机实际容量就是已准备清单的统一长度。
    maximum = sizes[0]
    explicit = config["data"]["samples_per_uav"]
    if explicit is not None and explicit != maximum:
        raise ValueError("配置的每机样本数与已准备清单不一致")
    if "samples" in manifest:
        rows = manifest["samples"]
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or sum(row.get("split") == "train" for row in rows) != len(flat)):
            raise ValueError("清单训练样本数与UAV索引不一致")
    # 第三步：显式 D0 优先；比例推定时至少保留一个可训练的初始样本。
    data = config["data"]
    initial = data["initial_samples_per_uav"]
    if initial is None:
        initial = max(1, math.floor(maximum * data["initial_fraction"]))
    if initial > maximum:
        raise ValueError("初始缓存不能超过每机样本上限")
    # 第四步：未填写 lambda 时，将 D_max-D0 个新样本均匀释放到任务末尾。
    # 环境这里只返回计划；是否进入某轮训练，仍由该轮开始时的快照决定。
    duration = float(config["mission"]["duration_s"])
    rate = data["sample_rate_per_s"]
    if rate is None:
        rate = (maximum - initial) / duration
    return dict(num_uavs=count, max_samples_per_uav=maximum,
                initial_samples_per_uav=initial, rate_samples_per_s=float(rate), duration_s=duration)


def available_samples_at(time_s, collection):
    """返回指定物理时刻每架 UAV 已可用的累计样本数，结果为整数。

    time_s 单位秒，必须位于闭区间 [0, duration_s]；collection 使用
    resolve_collection 的字段约定。按 min(D_max, D0+floor(lambda*t)) 计数，
    处理饱和容量和整数到达时刻的浮点边界。相同输入始终返回相同计数，
    不抽取图片、不推进时钟、不修改采集排列；返回计数也不代表该时刻已经
    开始新的 epoch，冻结训练快照是调用方的责任。
    """
    # 第一步：独立验证查询契约，供纯环境读取路径使用，不要求导入数据依赖。
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
    # 第二步：先判断已采满，避免很大但有限的 rate*time_s 乘积溢出；
    # rate=0 时不做除法，初始已满的情况由末尾 min 同样正确处理。
    if rate > 0 and time_s >= (maximum - initial) / rate:
        return maximum
    # 第三步：向正无穷方向修正一个浮点单位，避免理论上的整数到达边界
    # 因乘法舍入被 floor 少算一个；这不是随机采样或提前推进采集时钟。
    acquired = math.floor(math.nextafter(float(rate * time_s), math.inf))
    return min(maximum, initial + acquired)


def training_energy_capacity(config):
    """返回每架 UAV 可用于训练计算和通信的电量上限，单位 J。

    通常传入经 load_config/validate_config 校验的配置。四项 energy 必须
    全部已标定：初始电量减去固定飞行能量、固定基础开销和储备能量。
    未填写项目或剩余电量为负时抛出 ValueError；None 不能作为零参与核算。
    此函数不按训练时长增加推进/悬停开销，也不扣除服务器的能量。
    """
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
    """计算平均期望梯度精度对应的条件性充分 epoch 预算，返回字典。

    输入为已校验配置中的 theory 和 training 参数；先要求 0<K*L*eta<1/2，
    再要求误差底项 C 小于 epsilon。满足时将 A/(epsilon-C) 向上补足为
    聚合间隔 K 的整数倍，且至少包含一个完整聚合周期。

    sufficient_epochs 是推导出的预算；configured_epochs 仍是用户配置值，
    本函数不会自动改写它。configured_bound 报告实际配置轮数处的 A/Q+C。
    返回的 conditional=True 说明这里只做公式核算，不验证 L/G/V/rho/delta
    是否覆盖真实网络、数据流与自适应模型点；也不保证最终模型分类准确率。
    """
    p, t = config["theory"], config["training"]
    K, eta = t["aggregation_period"], t["epoch_step_size"]
    # 第一步：由累计步长计算无量纲周期尺度，保证漂移系数 kappa 小于 1。
    a = K * p["L"] * eta
    if not 0 < a < 0.5:
        raise ValueError("充分预算需要0 < K*L*eta < 1/2")
    kappa = (a / (1 - a)) ** 2
    # 第二步：计算衰减系数 A 和误差底项 C；KG^2 修正聚合索引，使平均
    # 覆盖训练后的 v1..vJ，而不把初始化 v0 当作已完成训练的输出。
    A = 2 * p["delta"] / (eta * (1 - kappa)) + K * p["G"] ** 2
    C = (kappa * p["V"] + 2 * p["G"] * p["rho"] * (kappa + a) + p["rho"] ** 2) / (1 - kappa)
    # 底项已达到目标时，仅增加 epoch 不能由这个充分界得到精度保证。
    if C >= p["epsilon"]:
        raise ValueError("误差底项不小于目标精度，不能推出有限充分预算")
    # 第三步：先按聚合周期向上取整，再乘回 K；max 保留至少一次聚合。
    # 不取近似平均轮数，也不向下取整，否则可能破坏充分性或聚合边界。
    epochs = K * max(1, math.ceil(A / (K * (p["epsilon"] - C))))
    return dict(a=a, kappa=kappa, A=A, C_infinity=C, sufficient_epochs=epochs,
                configured_epochs=t["total_epochs"], configured_bound=A / t["total_epochs"] + C,
                conditional=True)


if __name__ == "__main__":
    # 直接运行仅用于检查配置与条件性预算；不会生成文件或启动训练。
    current = load_config()
    print(json.dumps(current, ensure_ascii=False, indent=2, allow_nan=False))
    print("条件性训练预算：", theory_budget(current))
