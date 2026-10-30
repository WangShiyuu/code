"""动态 UAV-SFL：参数、环境、式(25)—(31)优化、函数式 ResNet18 与训练。

来源：同目录 notion_source.md，Notion 页面最后编辑于 2026-10-29T18:45:37Z。
入口：python uav_sfl.py {config,prepare,profile,simulate,train,check,smoke} --help
所有研究逻辑在本文件；不导入 code_v0/code_v1，也不定义有状态业务类。

数据契约（数组第 0 维始终按 UAV 编号 0..N-1 排列）：
  Config: 嵌套 dict，唯一默认值见 default_config()，可用 JSON 局部覆盖。
  Data: {records:list[dict], clients:list[list[int]], val:list[int], test:list[int],
         release_s: ndarray[Dmax], start_s:float, classes:list[str], ...}。
  Model: {params:dict[str,Tensor], buffers:dict[str,Tensor]}；BN 统计量固定。
  Profile: {client_cycles,server_cycles,activation_bits,gradient_bits,prefix_bits,
            client_flops,server_flops: ndarray[9], metadata:dict}。
  Resources: {frequency_hz,power_w,up_hz,down_hz: ndarray[N]}。
  Problem: {epoch:int,count:int,gain:ndarray[N],split:ndarray[N],next_split:array|None,
            remaining_j:ndarray[N],aggregate:bool,final:bool}。
  SolveResult: {ok:bool,status:str,resources:Resources|None,metrics:dict|None,
                trace:list[dict],initialization:list[dict],wall_s:float}。
  RunResult: {summary:dict,history:list[dict],final_model:Model|None,
              best_model:Model|None,branches:list[Model]|None,output_dir:str}。

核心函数不修改传入配置/参数/资源；文件读写、打印仅在 I/O 和入口函数。
单位：m、s、Hz、W、J、bit、cycle；CVXPY 内部采用频率比例、带宽比例、Mbit/s。
物理时间来自式(13)，不等于本机训练或求解的墙钟时间。
默认硬件、轨迹、预算是可运行场景假设，非设备标定结果；详见 README。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import heapq
import io
import json
import math
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


# 1. 参数与数据结构 ------------------------------------------------------------
def default_config() -> dict:
    """返回独立 Config。修改此函数即可集中设置本研究全部参数。"""
    return {
        "seed": 42,
        "num_uavs": 5,
        "data": {"kind": "cbcl", "root": str(ROOT.parents[1] / "DataSet" / "CBCL StreetScenes"),
                 "image_size": 64, "val_fraction": 0.1, "test_fraction": 0.1,
                 "samples_per_uav": None, "initial_fraction": 0.1, "rate_per_s": 1.0,
                 "synthetic_per_uav": 20, "synthetic_eval_size": 12,
                 "partition": "iid", "dirichlet_alpha": 0.5,
                 "fixed_epoch_counts": None},
        "training": {"epochs": 100, "aggregation_period": 2, "epoch_step_size": 0.01,
                     "batch_size": 32, "device": "cpu", "torch_threads": 4,
                     "checkpoint": None, "allow_random_init": False, "save_checkpoint": True},
        "budget": {"mode": "empirical", "A_bar": 3.85, "C_bar": 0.0115,
                   "epsilon": 0.05, "effective_bound_verified": False},
        "trajectory": {"server_xyz_m": [0.0, 0.0, 0.0], "altitude_m": 100.0,
                       "radius_min_m": 250.0, "radius_max_m": 450.0,
                       "center_offset_m": 150.0, "speed_m_s": 10.0},
        "channel": {"los_a": 9.61, "los_b": 0.16, "beta_l": 1e-4, "beta_n": 1e-6,
                    "alpha_l": 2.0, "alpha_n": 2.5, "noise_psd_w_hz": 3.981e-21},
        "uav": {"kappa": 1e-28, "compute_power_max_w": 0.3375, "power_max_w": 1.0,
                "cycles_per_flop": 0.5, "energy_cap_j": 200000.0},
        "server": {"frequency_hz": 3e9, "kappa": 1e-28, "cycles_per_flop": 0.03,
                   "downlink_power_w": 2.0, "aggregation_time_s": 0.01,
                   "aggregation_energy_j": 0.1},
        "radio": {"total_up_hz": 2e7, "total_down_hz": 2e7},
        "reference": {"frequency_fraction": 0.8, "power_fraction": 0.5},
        "optimization": {"strategy": "dynamic_opt", "fixed_split": 4,
                         "reference_recovery": "none",
                         "window_epochs": 5, "max_iterations": 50,
                         "relative_tolerance": 1e-5, "feasibility_tolerance": 2e-6,
                         "positive_floor": 1e-5, "solver": "CLARABEL",
                         "max_initial_candidates": 60000, "max_coordinate_sweeps": 100,
                         "score_tolerance_j": 1e-10},
        "task": {"epoch_deadline_s": 600.0, "fixed_energy_j": 0.0},
        "output_root": str(ROOT / "output"),
    }


def merge_config(base: dict, updates: dict) -> dict:
    """严格递归覆盖已知字段，返回新 dict；拼错字段立即报错。"""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"未知配置字段: {key}")
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


def validate_config(c: dict) -> dict:
    """校验形状、物理边界和研究协议；返回深拷贝，不检查输入文件是否存在。"""
    c = copy.deepcopy(c)
    def positive(x, name, integer=False):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f"{name} 必须为有限正数")
        if integer and not isinstance(x, int):
            raise ValueError(f"{name} 必须为整数")
    for key in ("num_uavs",):
        positive(c[key], key, True)
    if isinstance(c["seed"], bool) or not isinstance(c["seed"], int) or c["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    for key in ("epochs", "batch_size", "torch_threads"):
        positive(c["training"][key], key, True)
    if c["training"]["aggregation_period"] != 2 or c["training"]["epochs"] % 2:
        raise ValueError("本研究 K=2，epochs 必须是正偶数")
    positive(c["training"]["epoch_step_size"], "epoch_step_size")
    if c["budget"]["mode"] not in ("empirical", "theoretical"):
        raise ValueError("budget.mode 必须是 empirical/theoretical")
    c["training"]["epochs"] = training_budget(c)["epochs"]
    d = c["data"]
    if d["kind"] not in ("cbcl", "synthetic") or d["partition"] not in ("iid", "dirichlet"):
        raise ValueError("data.kind=cbcl/synthetic，partition=iid/dirichlet")
    positive(d["image_size"], "image_size", True)
    if d["image_size"] < 32:
        raise ValueError("ResNet18 image_size 至少 32")
    for key in ("rate_per_s", "dirichlet_alpha", "synthetic_per_uav", "synthetic_eval_size"):
        positive(d[key], key, key.startswith("synthetic"))
    if d["samples_per_uav"] is not None:
        positive(d["samples_per_uav"], "samples_per_uav", True)
    if not 0 < d["initial_fraction"] <= 1:
        raise ValueError("initial_fraction 必须属于 (0,1]")
    if min(d["val_fraction"], d["test_fraction"]) <= 0 or d["val_fraction"] + d["test_fraction"] >= 1:
        raise ValueError("验证/测试比例必须为正且和小于 1")
    for group in ("uav", "radio", "channel"):
        for key, val in c[group].items():
            if key == "energy_cap_j" and isinstance(val, list):
                if len(val) != c["num_uavs"]:
                    raise ValueError("energy_cap_j 数组长度必须为 N")
                for v in val:
                    positive(v, key)
            else:
                positive(val, key)
    for key, val in c["server"].items():
        if key in ("aggregation_time_s", "aggregation_energy_j"):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{key} 必须非负")
        else:
            positive(val, key)
    positive(c["task"]["epoch_deadline_s"], "epoch_deadline_s")
    if not math.isfinite(c["task"]["fixed_energy_j"]) or c["task"]["fixed_energy_j"] < 0:
        raise ValueError("fixed_energy_j 必须非负")
    t = c["trajectory"]
    if len(t["server_xyz_m"]) != 3 or not np.isfinite(t["server_xyz_m"]).all():
        raise ValueError("server_xyz_m 必须为有限三维坐标")
    if t["altitude_m"] <= t["server_xyz_m"][2]:
        raise ValueError("UAV 必须高于地面服务器")
    for key in ("speed_m_s", "radius_min_m", "radius_max_m"):
        positive(t[key], key)
    if t["radius_max_m"] < t["radius_min_m"] or not math.isfinite(t["center_offset_m"]):
        raise ValueError("轨迹半径/中心偏移不合法")
    o = c["optimization"]
    if o["strategy"] not in ("dynamic_opt", "fixed_opt", "fixed_fixed", "no_window"):
        raise ValueError("未知 strategy")
    if o["reference_recovery"] not in ("none", "rebuild"):
        raise ValueError("reference_recovery 必须为 none/rebuild")
    if isinstance(o["fixed_split"], bool) or o["fixed_split"] not in range(9):
        raise ValueError("fixed_split 必须为 0..8")
    for key in ("window_epochs", "max_iterations", "max_initial_candidates", "max_coordinate_sweeps"):
        positive(o[key], key, True)
    for key in ("relative_tolerance", "feasibility_tolerance", "positive_floor", "score_tolerance_j"):
        positive(o[key], key)
    if o["positive_floor"] * c["num_uavs"] >= 1:
        raise ValueError("数值正下界过大，无法分配带宽")
    for v in c["reference"].values():
        if not 0 < v <= 1:
            raise ValueError("参考资源比例必须属于 (0,1]")
    fixed = d["fixed_epoch_counts"]
    if fixed is not None:
        if len(fixed) != c["training"]["epochs"]:
            raise ValueError("fixed_epoch_counts 长度必须等于训练轮数")
        for v in fixed:
            positive(v, "fixed_epoch_counts", True)
        if any(a > b for a, b in zip(fixed, fixed[1:])):
            raise ValueError("统一样本快照必须非减")
    return c


def load_config(path: str | None = None, updates: dict | None = None) -> dict:
    """读取可选 JSON 和内存覆盖，返回验证后的 Config；相对路径相对 JSON。"""
    c = default_config()
    if path:
        source = Path(path).resolve()
        c = merge_config(c, json.loads(source.read_text(encoding="utf-8-sig")))
        for group, key in (("data", "root"), ("training", "checkpoint")):
            if c[group][key]:
                p = Path(c[group][key]).expanduser()
                c[group][key] = str(p if p.is_absolute() else source.parent / p)
        p = Path(c["output_root"]).expanduser()
        c["output_root"] = str(p if p.is_absolute() else source.parent / p)
    return validate_config(merge_config(c, updates or {}))


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
def read_cbcl_records(root: str) -> tuple[list[dict], dict]:
    """读取两份 ZIP，返回有效区域记录和审计计数；不解压、不复制样本。"""
    from PIL import Image
    root = Path(root)
    records, skipped = [], 0
    with zipfile.ZipFile(root / "Images.zip") as images, zipfile.ZipFile(root / "Annotations.zip") as annotations:
        members = {}
        for member in images.namelist():
            name = PurePosixPath(member.replace("\\", "/")).name
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                if name in members:
                    raise ValueError(f"重复图片文件名: {name}")
                members[name] = member
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
                label = (obj.findtext("name") or "").strip().lower()
                if (obj.findtext("deleted") or "0").strip() == "1" or not label:
                    skipped += 1
                    continue
                if label not in CLASSES:
                    raise ValueError(f"未知类别 {label!r}: {name}")
                points = [(float(pt.findtext("x")), float(pt.findtext("y"))) for pt in obj.findall("polygon/pt")]
                if len(points) < 3 or not np.isfinite(points).all():
                    skipped += 1
                    continue
                xs, ys = zip(*points)
                box = [max(0, math.floor(min(xs) - 1)), max(0, math.floor(min(ys) - 1)),
                       min(width, math.ceil(max(xs))), min(height, math.ceil(max(ys)))]
                if box[0] >= box[2] or box[1] >= box[3]:
                    skipped += 1
                    continue
                records.append({"id": f"{name}:{oid}", "scene": name, "member": members[name],
                                "bbox": box, "label": CLASSES.index(label)})
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


def prepare_data(config: dict) -> dict:
    """I/O：返回 Data；CBCL 按原始场景 80/10/10 隔离，再等量分配训练区域。"""
    d, n = config["data"], config["num_uavs"]
    if d["kind"] == "cbcl":
        records, audit = read_cbcl_records(d["root"])
        scenes = sorted({r["scene"] for r in records})
        scenes = np.random.default_rng(config["seed"]).permutation(scenes).tolist()
        nv, nt = max(1, round(len(scenes) * d["val_fraction"])), max(1, round(len(scenes) * d["test_fraction"]))
        if nv + nt >= len(scenes):
            raise ValueError("独立场景不足，无法划分训练/验证/测试")
        val_scenes, test_scenes = set(scenes[:nv]), set(scenes[nv:nv + nt])
        val = [i for i, r in enumerate(records) if r["scene"] in val_scenes]
        test = [i for i, r in enumerate(records) if r["scene"] in test_scenes]
        train = [i for i, r in enumerate(records) if r["scene"] not in val_scenes | test_scenes]
    else:
        train_size, eval_size = n * d["synthetic_per_uav"], d["synthetic_eval_size"]
        records = [{"id": f"synthetic_{i}", "scene": f"synthetic_{i}", "label": i % len(CLASSES)}
                   for i in range(train_size + 2 * eval_size)]
        train = list(range(train_size))
        val = list(range(train_size, train_size + eval_size))
        test = list(range(train_size + eval_size, len(records)))
        audit = {"synthetic": True, "regions": len(records)}
    clients, unused = partition_clients(train, records, config)
    capacity = len(clients[0])
    initial = max(1, math.floor(capacity * d["initial_fraction"]))
    release = np.arange(1, capacity + 1, dtype=float) / d["rate_per_s"]
    if d["fixed_epoch_counts"] is not None and max(d["fixed_epoch_counts"]) > capacity:
        raise ValueError("统一快照超过真实训练容量")
    # 指纹覆盖分组、样本顺序、释放表和预处理；原始包另留大小/mtime 以供审计。
    result = {"kind": d["kind"], "records": records, "clients": clients, "val": val, "test": test,
              "unused_train": unused, "release_s": release, "start_s": float(release[initial - 1]),
              "capacity": capacity, "classes": CLASSES.copy(), "audit": audit,
              "class_counts": [np.bincount([records[i]["label"] for i in ids], minlength=len(CLASSES)) for ids in clients],
              "image_size": d["image_size"], "root": d["root"], "seed": config["seed"]}
    if d["kind"] == "cbcl":
        result["source_files"] = {name: {"bytes": (Path(d["root"]) / name).stat().st_size,
                                        "mtime_ns": (Path(d["root"]) / name).stat().st_mtime_ns}
                                  for name in ("Images.zip", "Annotations.zip")}
    result["fingerprint"] = hashlib.sha256(json.dumps(result, default=json_value, sort_keys=True).encode()).hexdigest()
    return result


def epoch_snapshot(data: dict, physical_s: float, epoch: int, config: dict) -> dict:
    """纯函数：返回轮初固定样本数/各机索引。轮内新样本只在下一次调用中出现。"""
    fixed = config["data"]["fixed_epoch_counts"]
    count = fixed[epoch - 1] if fixed is not None else int(np.searchsorted(data["release_s"], physical_s, side="right"))
    if not 1 <= count <= data["capacity"]:
        raise ValueError("当前释放表未满足启动条件或统一快照越界")
    return {"count": count, "indices": [ids[:count] for ids in data["clients"]],
            "time_s": physical_s, "mode": "matched_snapshot_ablation" if fixed else "physical_release"}


def load_batch(data: dict, indices: list[int], device: str) -> tuple[Any, Any]:
    """I/O：读取指定索引，返回 float32 [B,3,H,W]、int64 [B]；无随机增强。"""
    import torch
    from PIL import Image
    size, samples = data["image_size"], []
    labels = [data["records"][i]["label"] for i in indices]
    if data["kind"] == "cbcl":
        with zipfile.ZipFile(Path(data["root"]) / "Images.zip") as z:
            for i in indices:
                r = data["records"][i]
                with Image.open(io.BytesIO(z.read(r["member"]))) as im:
                    crop = im.convert("RGB").crop(r["bbox"]).resize((size, size), Image.Resampling.BILINEAR)
                    samples.append(np.asarray(crop, dtype=np.float32) / 255.0)
    else:
        for i, label in zip(indices, labels):
            rng = np.random.default_rng(np.random.SeedSequence([data["seed"], i]))
            pixels = rng.uniform(0, 0.2, (size, size, 3)).astype(np.float32)
            pixels[:, :, label % 3] += 0.6  # 仅供诊断的可学习合成模式。
            samples.append(pixels)
    array = (np.stack(samples) - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return (torch.tensor(array.transpose(0, 3, 1, 2), device=device),
            torch.tensor(labels, dtype=torch.long, device=device))


# 3. 函数式 ResNet18、画像、SGD -------------------------------------------------
def block_name(index: int) -> str:
    """0..7 残差块索引 -> torchvision 兼容参数前缀。"""
    return f"layer{index // 2 + 1}.{index % 2}"


def initialize_model(config: dict, require_pretrained: bool = False) -> tuple[dict, dict]:
    """返回 Model 和初始化来源；无 nn.Module/全局 RNG/隐式权重下载。"""
    import torch
    params, buffers = {}, {}
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
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
    provenance = {"kind": "random_diagnostic", "seed": config["seed"], "head_reinitialized": True}
    path = config["training"]["checkpoint"]
    if path:
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
    elif require_pretrained and not config["training"]["allow_random_init"]:
        raise ValueError("正式训练需要 training.checkpoint（标准 ResNet18 state_dict 或本程序模型）；诊断可显式 --allow-random-init")
    device = config["training"]["device"]
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


def forward_prefix(model: dict, images: Any, split: int) -> Any:
    """执行 stem + 前 split 个完整残差块，返回有计算图的激活 Tensor。"""
    from torch.nn import functional as F
    if split not in range(9):
        raise ValueError("split 必须为 0..8")
    x = F.conv2d(images, model["params"]["conv1.weight"], stride=2, padding=3)
    x = F.max_pool2d(F.relu(frozen_bn(x, model, "bn1")), 3, 2, 1)
    for i in range(split):
        x = residual_block(x, model, i)
    return x


def forward_suffix(model: dict, activation: Any, split: int) -> Any:
    """执行剩余残差块 + 池化/分类头，返回 logits [B,9]。"""
    from torch.nn import functional as F
    x = activation
    for i in range(split, 8):
        x = residual_block(x, model, i)
    x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    return F.linear(x, model["params"]["fc.weight"], model["params"]["fc.bias"])


def prefix_keys(model: dict, split: int) -> list[str]:
    """返回客户端可训练参数键；BN 固定 buffers 已预部署，不计重复通信。"""
    prefixes = ["conv1.", "bn1."] + [block_name(i) + "." for i in range(split)]
    return [name for name in model["params"] if any(name.startswith(p) for p in prefixes)]


def split_gradients(model: dict, images: Any, labels: Any, split: int) -> tuple[dict, dict]:
    """显式 detach/传激活/返梯度；返回全模型梯度字典和批次统计，不更新参数。"""
    import torch
    from torch.nn import functional as F
    left = prefix_keys(model, split)
    right = [name for name in model["params"] if name not in set(left)]
    boundary = forward_prefix(model, images, split)
    received = boundary.detach().clone().requires_grad_(True)
    logits = forward_suffix(model, received, split)
    loss = F.cross_entropy(logits, labels)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("批次损失非有限，停止本轮训练")
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


def train_client_epoch(model: dict, data: dict, indices: list[int], split: int,
                       epoch: int, uid: int, config: dict) -> tuple[dict, dict]:
    """遍历一次冻结快照；返回新 Model 与加权统计，尾批步长按真实大小计算。"""
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, uid]))
    order = rng.permutation(indices).tolist()
    state, loss_sum, correct, step_sum, up, down = model, 0.0, 0, 0.0, 0, 0
    batch_size = config["training"]["batch_size"]
    for offset in range(0, len(order), batch_size):
        batch = order[offset:offset + batch_size]
        x, y = load_batch(data, batch, config["training"]["device"])
        gradients, metrics = split_gradients(state, x, y, split)
        step = config["training"]["epoch_step_size"] * len(batch) / len(order)
        state = sgd_step(state, gradients, step)
        loss_sum += metrics["loss"] * len(batch)
        correct += metrics["correct"]
        step_sum += step
        up += metrics["activation_bits"]
        down += metrics["gradient_bits"]
    return state, {"loss": loss_sum / len(order), "accuracy": correct / len(order),
                   "samples": len(order), "step_sum": step_sum, "activation_bits": up, "gradient_bits": down}


def aggregate_models(models: list[dict], counts: list[int]) -> dict:
    """式(5)：完整模型按样本数平均，返回新 Model；固定 BN buffers 必须一致。"""
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
    with torch.no_grad():
        for offset in range(0, len(indices), config["training"]["batch_size"]):
            ids = indices[offset:offset + config["training"]["batch_size"]]
            x, y = load_batch(data, ids, config["training"]["device"])
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


def profile_splits(model: dict, config: dict) -> dict:
    """统计九切点前后向 FLOPs 与真实 bit，乘显式硬件系数得到等效周期。

    FLOP 计数不是硬件实测。仅覆盖 FlopCounterMode 支持算子，不含优化器、
    数据读取、内存搬运及未支持算子；metadata 始终记录此限制。
    """
    import torch
    from torch.nn import functional as F
    from torch.utils.flop_counter import FlopCounterMode
    size, device = config["data"]["image_size"], config["training"]["device"]
    x = torch.linspace(-1, 1, 3 * size * size, device=device).reshape(1, 3, size, size)
    y = torch.zeros(1, dtype=torch.long, device=device)
    rows = []
    for split in range(9):
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
        rows.append({"client_flops": client_flops, "server_flops": server_flops,
                     "client_cycles": client_flops * config["uav"]["cycles_per_flop"],
                     "server_cycles": server_flops * config["server"]["cycles_per_flop"],
                     "activation_bits": boundary.numel() * boundary.element_size() * 8 + 64,
                     "gradient_bits": boundary.numel() * boundary.element_size() * 8,
                     "prefix_bits": sum(model["params"][key].numel() * model["params"][key].element_size() * 8 for key in left)})
    profile = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
    profile["metadata"] = {"method": "supported_operator_FLOPs_times_configured_cycles_per_FLOP",
                           "hardware_measured": False, "image_size": size, "dtype": "float32",
                           "labels": "int64", "prefix_payload": "trainable_parameters_only_frozen_BN_predeployed",
                           "exclusions": "unsupported operators, optimizer updates, memory/I/O/runtime overhead"}
    return profile


# 4. 给定轨迹、平均大尺度信道与原始成本 ------------------------------------------
def channel_at(physical_s: float, config: dict) -> dict:
    """式(6)—(8)：给定偏心圆巡逻轨迹的确定性平均功率增益；无小尺度随机衰落。"""
    n, t, c = config["num_uavs"], config["trajectory"], config["channel"]
    radius = np.linspace(t["radius_min_m"], t["radius_max_m"], n)
    phase = 2 * np.pi * np.arange(n) / n
    angle = phase + t["speed_m_s"] * physical_s / radius
    xyz = np.column_stack((radius * np.cos(angle) + t["center_offset_m"] * np.cos(phase),
                           radius * np.sin(angle) + t["center_offset_m"] * np.sin(phase),
                           np.full(n, t["altitude_m"])))
    delta = xyz - np.asarray(t["server_xyz_m"])
    distance = np.linalg.norm(delta, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(delta[:, 2] / distance, -1, 1)))
    probability = 1 / (1 + c["los_a"] * np.exp(-c["los_b"] * (elevation - c["los_a"])))
    gain = probability * c["beta_l"] * distance ** (-c["alpha_l"]) + (1 - probability) * c["beta_n"] * distance ** (-c["alpha_n"])
    return {"xyz_m": xyz, "distance_m": distance, "elevation_deg": elevation, "los_probability": probability, "gain": gain}


def maximum_frequency(config: dict) -> float:
    """式(9)：由计算功率上限恢复 f_max，返回 Hz。"""
    return (config["uav"]["compute_power_max_w"] / config["uav"]["kappa"]) ** (1 / 3)


def reference_resources(config: dict) -> dict:
    """返回独立参考配置；其可行性必须由 audit_resources 判定。"""
    n = config["num_uavs"]
    return {"frequency_hz": np.full(n, maximum_frequency(config) * config["reference"]["frequency_fraction"]),
            "power_w": np.full(n, config["uav"]["power_max_w"] * config["reference"]["power_fraction"]),
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
    """构造单 epoch Problem；聚合轮必须先确定下一周期切分，末轮必须为 None。"""
    n, epochs = config["num_uavs"], config["training"]["epochs"]
    aggregate, final = epoch % 2 == 0, epoch == epochs
    if not 1 <= epoch <= epochs or count < 1:
        raise ValueError("epoch/count 越界")
    s = np.asarray(split, dtype=int)
    q = None if next_split is None else np.asarray(next_split, dtype=int)
    if not np.array_equal(np.asarray(split), s) or (q is not None and not np.array_equal(np.asarray(next_split), q)):
        raise ValueError("切分编号必须为整数")
    if s.shape != (n,) or np.any((s < 0) | (s > 8)):
        raise ValueError("当前切分向量非法")
    if aggregate and not final and (q is None or q.shape != (n,) or np.any((q < 0) | (q > 8))):
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
    s, d = problem["split"], problem["count"]
    zero = np.zeros(len(s))
    return {"client_cycles": d * profile["client_cycles"][s], "server_cycles": d * profile["server_cycles"][s],
            "activation_bits": d * profile["activation_bits"][s], "gradient_bits": d * profile["gradient_bits"][s],
            "upload_bits": profile["prefix_bits"][s] if problem["aggregate"] else zero,
            "download_bits": profile["prefix_bits"][problem["next_split"]] if problem["next_split"] is not None else zero}


def evaluate_resources(problem: dict, resources: dict, profile: dict, config: dict) -> dict:
    """用原式(10)—(17)重算时延、能耗和载荷；各同步阶段分别取 max 后相加。"""
    l = loads(problem, profile)
    f, p = resources["frequency_hz"], resources["power_w"]
    e, u, c = config["server"], config["uav"], config["channel"]
    ru = shannon_rate(resources["up_hz"], p, problem["gain"], c["noise_psd_w_hz"])
    rd = shannon_rate(resources["down_hz"], e["downlink_power_w"], problem["gain"], c["noise_psd_w_hz"])
    tc, te = l["client_cycles"] / f, l["server_cycles"] / e["frequency_hz"]
    ta, tg, tu, td = l["activation_bits"] / ru, l["gradient_bits"] / rd, l["upload_bits"] / ru, l["download_bits"] / rd
    components = {"uav_compute_j": u["kappa"] * l["client_cycles"] * f ** 2,
                  "activation_upload_j": p * ta, "prefix_upload_j": p * tu,
                  "server_compute_j": e["kappa"] * e["frequency_hz"] ** 2 * l["server_cycles"],
                  "gradient_download_j": e["downlink_power_w"] * tg,
                  "prefix_download_j": e["downlink_power_w"] * td}
    uav_energy = components["uav_compute_j"] + components["activation_upload_j"] + components["prefix_upload_j"]
    agg_time = e["aggregation_time_s"] if problem["aggregate"] else 0.0
    agg_energy = e["aggregation_energy_j"] if problem["aggregate"] else 0.0
    server_energy = float(sum(np.sum(components[k]) for k in ("server_compute_j", "gradient_download_j", "prefix_download_j")) + agg_energy)
    stage_times = {"split_training_s": float(np.max(tc + te + ta + tg)), "prefix_upload_s": float(np.max(tu)),
                   "aggregation_s": agg_time, "prefix_download_s": float(np.max(td))}
    return {"duration_s": sum(stage_times.values()), "total_energy_j": float(uav_energy.sum() + server_energy),
            "uav_energy_j": uav_energy, "server_energy_j": server_energy, "components": components,
            "aggregation_energy_j": agg_energy, "stage_times": stage_times,
            "branch_training_s": tc + te + ta + tg, "up_rate_bps": ru, "down_rate_bps": rd,
            "compute_power_w": u["kappa"] * f ** 3, "loads": l}


def audit_resources(problem: dict, resources: dict, profile: dict, config: dict) -> dict:
    """原始约束核验，返回 ok、无量纲最大违约与物理残差；不得用 solver 状态替代。"""
    n, tol = config["num_uavs"], config["optimization"]["feasibility_tolerance"]
    if any(np.shape(v) != (n,) or not np.isfinite(v).all() or np.any(np.asarray(v) <= 0) for v in resources.values()):
        return {"ok": False, "reason": "nonpositive_or_invalid_resources"}
    m = evaluate_resources(problem, resources, profile, config)
    residuals = {"deadline_s": m["duration_s"] - config["task"]["epoch_deadline_s"],
                 "battery_j": m["uav_energy_j"] - problem["remaining_j"],
                 "frequency_hz": resources["frequency_hz"] - maximum_frequency(config),
                 "power_w": resources["power_w"] - config["uav"]["power_max_w"],
                 "up_hz": float(np.sum(resources["up_hz"])) - config["radio"]["total_up_hz"],
                 "down_hz": float(np.sum(resources["down_hz"])) - config["radio"]["total_down_hz"]}
    normalized = [residuals["deadline_s"] / config["task"]["epoch_deadline_s"],
                  float(np.max(residuals["battery_j"] / np.maximum(1.0, problem["remaining_j"]))),
                  float(np.max(residuals["frequency_hz"])) / maximum_frequency(config),
                  float(np.max(residuals["power_w"])) / config["uav"]["power_max_w"],
                  residuals["up_hz"] / config["radio"]["total_up_hz"], residuals["down_hz"] / config["radio"]["total_down_hz"]]
    violation = max(0.0, *normalized)
    return {"ok": bool(violation <= tol and np.isfinite(m["total_energy_j"])),
            "normalized_violation": violation, "residuals": residuals}


# 5. 单 epoch 凸子问题、可行初始化、逆二次迭代 ----------------------------------
def convex_subproblem(problem: dict, profile: dict, config: dict,
                      tangent: dict | None = None, fixed_power: float | None = None) -> dict:
    """求解一次凸问题并返回资源/求解状态。

    tangent != None：式(31)，用相切点归一化后的 psi=2*sqrt(R/R0)-p/p0，
    上行上界=E_up0/psi，与式(29)—(31)完全等价，避免 y 的数量级病态。
    fixed_power != None：固定发射功率的凸 Phase-I，最小化时延/电量松弛；
    只用于寻找原问题可行点，找不到不构成全局不可行证明。
    """
    import cvxpy as cp
    if (tangent is None) == (fixed_power is None):
        raise ValueError("必须且只能指定 tangent 或 fixed_power")
    n, o, u, e, radio = config["num_uavs"], config["optimization"], config["uav"], config["server"], config["radio"]
    l, floor = loads(problem, profile), o["positive_floor"]
    f = cp.Variable(n, name="frequency_fraction")
    bu, bd = cp.Variable(n, name="up_fraction"), cp.Variable(n, name="down_fraction")
    p = cp.Variable(n, name="power_fraction") if tangent is not None else np.full(n, fixed_power)
    tau = cp.Variable(1 + int(problem["aggregate"]) + int(problem["next_split"] is not None), nonneg=True, name="stage_s")
    noise, gain, fmax = config["channel"]["noise_psd_w_hz"], problem["gain"], maximum_frequency(config)
    # -rel_entr(b,b+a*p)=b*log(1+a*p/b) 是联合凹的 Shannon perspective。
    ru = -(radio["total_up_hz"] / 1e6) * cp.rel_entr(bu, bu + cp.multiply(gain * u["power_max_w"] / (noise * radio["total_up_hz"]), p)) / np.log(2)
    rd = -(radio["total_down_hz"] / 1e6) * cp.rel_entr(bd, bd + gain * e["downlink_power_w"] / (noise * radio["total_down_hz"])) / np.log(2)
    invru, invrd = cp.inv_pos(ru), cp.inv_pos(rd)
    ec = cp.multiply(u["kappa"] * l["client_cycles"] * fmax ** 2, cp.square(f))
    up_mbit = (l["activation_bits"] + l["upload_bits"]) / 1e6
    down_mbit = (l["gradient_bits"] + l["download_bits"]) / 1e6
    constraints = [f >= floor, f <= 1, bu >= floor, bd >= floor, cp.sum(bu) <= 1, cp.sum(bd) <= 1]
    if tangent is not None:
        m0 = evaluate_resources(problem, tangent, profile, config)
        ru0 = m0["up_rate_bps"] / 1e6
        p0 = tangent["power_w"]
        psi = 2 * cp.sqrt(cp.multiply(1 / ru0, ru)) - cp.multiply(u["power_max_w"] / p0, p)
        eup0 = up_mbit * p0 / ru0
        eup = cp.multiply(eup0, cp.inv_pos(psi))
        constraints += [p >= floor, p <= 1, psi >= floor]
        scale = max(1e-6, m0["total_energy_j"] / n)
        slack = 0.0
    else:
        eup = cp.multiply(up_mbit * u["power_max_w"] * fixed_power, invru)
        scale = max(1e-6, float(np.sum(u["kappa"] * l["client_cycles"] * fmax ** 2)) / n)
        slack = cp.Variable(nonneg=True, name="phase_one_slack")
    edown = cp.multiply(e["downlink_power_w"] * down_mbit, invrd)
    train_time = cp.multiply(l["client_cycles"] / fmax, cp.inv_pos(f)) + l["server_cycles"] / e["frequency_hz"]
    train_time += cp.multiply(l["activation_bits"] / 1e6, invru) + cp.multiply(l["gradient_bits"] / 1e6, invrd)
    deadline = config["task"]["epoch_deadline_s"]
    constraints += [train_time / deadline <= tau[0]]
    if problem["aggregate"]:
        constraints += [cp.multiply(l["upload_bits"] / 1e6 / deadline, invru) <= tau[1]]
    if problem["next_split"] is not None:
        constraints += [cp.multiply(l["download_bits"] / 1e6 / deadline, invrd) <= tau[2]]
    agg_time = e["aggregation_time_s"] if problem["aggregate"] else 0.0
    battery_scale = np.maximum(scale, np.abs(problem["remaining_j"]))
    constraints += [cp.sum(tau) + agg_time / deadline <= 1 + slack,
                    cp.multiply(1 / battery_scale, ec + eup) <= problem["remaining_j"] / battery_scale + slack]
    objective = cp.sum(ec + eup + edown) / scale
    objective = objective if tangent is not None else slack + 1e-8 * objective
    program = cp.Problem(cp.Minimize(objective), constraints)
    if not program.is_dcp():
        raise RuntimeError("凸子问题没有通过 CVXPY DCP 校验")
    if o["solver"] not in cp.installed_solvers():
        raise RuntimeError(f"求解器 {o['solver']} 不可用；当前 {cp.installed_solvers()}")
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            options = {"max_iter": 300, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9} if o["solver"] == "CLARABEL" else {}
            program.solve(solver=o["solver"], **options)
        notes = [str(w.message) for w in caught]
    except cp.error.SolverError as exc:
        return {"ok": False, "status": "solver_error", "message": str(exc), "wall_s": time.perf_counter() - start}
    result = {"ok": False, "status": program.status, "warnings": notes, "dcp": True,
              "wall_s": time.perf_counter() - start}
    if program.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or any(v.value is None for v in (f, bu, bd)):
        return result
    power_fraction = np.asarray(p.value if tangent is not None else p)
    resources = {"frequency_hz": np.asarray(f.value) * fmax, "power_w": power_fraction * u["power_max_w"],
                 "up_hz": np.asarray(bu.value) * radio["total_up_hz"], "down_hz": np.asarray(bd.value) * radio["total_down_hz"]}
    audit = audit_resources(problem, resources, profile, config)
    result.update({"ok": audit["ok"], "resources": resources, "audit": audit, "epigraph_s": np.asarray(tau.value) * deadline,
                   "phase_slack": float(slack.value) if tangent is None else 0.0})
    return result


def feasible_initialization(problem: dict, profile: dict, config: dict, previous: dict | None = None) -> dict:
    """返回已按原约束验证的初值；参考/最大资源失败后做固定功率凸 Phase-I。"""
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
    maximum["power_w"][:] = config["uav"]["power_max_w"]
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


def solve_resources(problem: dict, profile: dict, config: dict, previous: dict | None = None) -> dict:
    """式(30)/(31)交替求解，返回 SolveResult；接受步均真实可行且真实目标不增。"""
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
            return result(False, "convex_subproblem_failed")
        candidate = proposed["resources"]
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
    """式(26)：返回 [N,9] 周期候选评分；chi 仅控制周期前新前缀下发项。"""
    u, e, noise = config["uav"], config["server"], config["channel"]["noise_psd_w_hz"]
    ru = shannon_rate(resources["up_hz"], resources["power_w"], gain, noise)[:, None]
    rd = shannon_rate(resources["down_hz"], e["downlink_power_w"], gain, noise)[:, None]
    m, chi = 2 * count, int(not first_cycle)
    compute = m * (u["kappa"] * resources["frequency_hz"][:, None] ** 2 * profile["client_cycles"] + e["kappa"] * e["frequency_hz"] ** 2 * profile["server_cycles"])
    upload = resources["power_w"][:, None] * (m * profile["activation_bits"] + profile["prefix_bits"]) / ru
    download = e["downlink_power_w"] * (m * profile["gradient_bits"] + chi * profile["prefix_bits"]) / rd
    return compute + upload + download


def screen_split(candidate: np.ndarray, current: np.ndarray, epoch: int, count: int,
                 gain: np.ndarray, remaining_j: np.ndarray, reference: dict,
                 profile: dict, config: dict) -> dict:
    """式(25)：检验前一未执行聚合轮 + 下一两轮，含全部同步开销和累计电量。"""
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


def select_splits(epoch: int, count: int, current: np.ndarray, gain_history: list[np.ndarray],
                  initial_gain: np.ndarray, remaining_j: np.ndarray, reference: dict,
                  profile: dict, config: dict) -> dict:
    """返回候选队列、逐机枚举日志；只读已完成 epoch 信道，不读取未来状态。

    优先保留可行原切分；否则按评分用堆枚举组合寻找初值。之后逐机枚举九点，
    严格下降且参考可行才接受，同分保留原值。枚举达上限会明确报告，非全局证书。
    """
    o, n = config["optimization"], config["num_uavs"]
    window = 1 if o["strategy"] == "no_window" else o["window_epochs"]
    gain = np.mean(gain_history[-window:], axis=0) if gain_history else initial_gain.copy()
    scores = reference_scores(count, gain, reference, profile, epoch == 1, config)
    # 任一设备最短分割训练本身已超时，则枚举 9**N 组合也不能恢复参考可行性。
    # 此下界只针对固定参考配置，不能被解释为原联合优化问题的不可行证书。
    ru = shannon_rate(reference["up_hz"], reference["power_w"], gain, config["channel"]["noise_psd_w_hz"])
    rd = shannon_rate(reference["down_hz"], config["server"]["downlink_power_w"], gain, config["channel"]["noise_psd_w_hz"])
    train_times = count * (profile["client_cycles"][None, :] / reference["frequency_hz"][:, None]
                          + profile["server_cycles"][None, :] / config["server"]["frequency_hz"]
                          + profile["activation_bits"][None, :] / ru[:, None]
                          + profile["gradient_bits"][None, :] / rd[:, None])
    lower_bound = float(train_times.min(axis=1).max())
    if epoch != 1:
        lower_bound = max(lower_bound, float(train_times[np.arange(n), current].max()))
    if lower_bound > config["task"]["epoch_deadline_s"] * (1 + o["feasibility_tolerance"]):
        return {"ok": False, "status": "reference_training_time_lower_bound_exceeded", "candidates": [],
                "scores_j": scores, "window_gain": gain, "trace": [], "initial_combinations": 0,
                "reference_time_lower_bound_s": lower_bound}
    trace, feasible = [], {}
    def score(s):
        return float(scores[np.arange(n), s].sum())
    def check(s):
        audit = screen_split(s, current, epoch, count, gain, remaining_j, reference, profile, config)
        if audit["ok"]:
            feasible[tuple(int(v) for v in s)] = score(s)
        return audit
    chosen = current.copy()
    initial_audit = check(chosen)
    searched = 0
    if not initial_audit["ok"]:
        orders = np.argsort(scores, axis=1, kind="stable")
        ranks = (0,) * n
        decode = lambda r: np.array([orders[i, r[i]] for i in range(n)], dtype=int)
        heap, visited = [(score(decode(ranks)), ranks)], {ranks}
        found = False
        while heap and searched < o["max_initial_candidates"]:
            _, ranks = heapq.heappop(heap)
            trial = decode(ranks)
            audit = check(trial)
            searched += 1
            if audit["ok"]:
                chosen, found = trial, True
                break
            for uid in range(n):
                if ranks[uid] < 8:
                    child = list(ranks)
                    child[uid] += 1
                    child = tuple(child)
                    if child not in visited:
                        visited.add(child)
                        heapq.heappush(heap, (score(decode(child)), child))
        if not found:
            return {"ok": False, "status": "reference_search_limit" if heap else "no_reference_feasible_split",
                    "candidates": [], "scores_j": scores, "window_gain": gain, "trace": trace, "initial_combinations": searched}
    converged = False
    for sweep in range(o["max_coordinate_sweeps"]):
        improved = False
        for uid in range(n):
            best, best_score = chosen.copy(), score(chosen)
            for cut in range(9):
                trial = chosen.copy()
                trial[uid] = cut
                audit = check(trial)
                trial_score = score(trial)
                better = audit["ok"] and trial_score < best_score - o["score_tolerance_j"]
                trace.append({"sweep": sweep, "uav": uid, "candidate": cut, "vector": trial.tolist(),
                              "score_j": trial_score, "feasible": audit["ok"], "reason": audit["reason"],
                              "improves_incumbent": bool(better)})
                if better:
                    best, best_score = trial, trial_score
            if not np.array_equal(best, chosen):
                trace.append({"sweep": sweep, "uav": uid, "accepted_vector": best.tolist(), "score_j": best_score})
                chosen, improved = best, True
        if not improved:
            converged = True
            break
    # 主候选为坐标搜索结果，后续按评分排列已发现的可行组合，供当前信道 P2 重试。
    ordered = [tuple(chosen.tolist())] + [key for key in sorted(feasible, key=lambda key: (feasible[key], key)) if key != tuple(chosen)]
    return {"ok": True, "status": "coordinate_converged" if converged else "coordinate_limit",
            "candidates": [np.array(key, dtype=int) for key in ordered], "scores_j": scores,
            "window_gain": gain, "trace": trace, "initial_combinations": searched,
            "reference_initial_audit": initial_audit}


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
def make_output_directory(config: dict, label: str) -> Path:
    """I/O：按北京时间创建唯一目录，不覆盖已有实验。"""
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(config["output_root"]) / f"{label}_{stamp}_seed{config['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def checkpoint_payload(model: dict) -> dict:
    """返回 detached CPU 张量字典，供 torch.save；不改变训练状态。"""
    return {group: {k: v.detach().cpu().clone() for k, v in values.items()} for group, values in model.items()}


def run_experiment(config: dict, train: bool = False, data: dict | None = None,
                   profile: dict | None = None, output: str | Path | None = None) -> dict:
    """算法1主入口，返回 RunResult，并保存可复核的参数、轨迹、资源及训练结果。

    train=False 只执行物理成本仿真；train=True 在资源通过原约束后执行真实 SGD。
    服务器并行分支在本地串行模拟，不把本机墙钟时间计作系统模型时间。
    """
    import torch
    config = validate_config(config)
    torch.set_num_threads(config["training"]["torch_threads"])
    output = Path(output) if output else make_output_directory(config, "train" if train else "simulate")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError(f"实验目录已被使用: {output}")
    save_json(output / "config.json", config)
    start_wall = time.perf_counter()
    base, provenance = initialize_model(config, require_pretrained=train)
    data = prepare_data(config) if data is None else data
    profile = profile_splits(base, config) if profile is None else profile
    if len(data["clients"]) != config["num_uavs"] or data["image_size"] != config["data"]["image_size"]:
        raise ValueError("传入 Data 与 Config 的 UAV 数/图像尺寸不一致")
    for key in ("client_cycles", "server_cycles", "activation_bits", "gradient_bits", "prefix_bits"):
        if np.shape(profile[key]) != (9,) or not np.isfinite(profile[key]).all() or np.any(np.asarray(profile[key]) <= 0):
            raise ValueError(f"画像 {key} 必须包含九个有限正数")
    save_json(output / "data_manifest.json", data)
    save_json(output / "split_profiles.json", profile)
    save_json(output / "initialization.json", provenance)
    n, epochs = config["num_uavs"], config["training"]["epochs"]
    strategy = config["optimization"]["strategy"]
    dynamic = strategy in ("dynamic_opt", "no_window")
    current = np.full(n, config["optimization"]["fixed_split"], dtype=int)
    reference = reference_resources(config)
    remaining = np.broadcast_to(np.asarray(config["uav"]["energy_cap_j"], dtype=float), (n,)).copy()
    time_s, history, gain_history = data["start_s"], [], []
    branches = [base for _ in range(n)] if train else None  # 参数只读共享，SGD 返回新字典。
    global_model, best_model, best_val = base if train else None, None, None
    best_epoch, last_aggregation = None, 0
    failure, total_energy, processed = None, 0.0, 0
    try:
        for epoch in range(1, epochs + 1):
            snapshot = epoch_snapshot(data, time_s, epoch, config)
            channel = channel_at(time_s, config)
            selection = None
            if dynamic and (epoch == 1 or (epoch % 2 == 0 and epoch < epochs)):
                selection = select_splits(epoch, snapshot["count"], current, gain_history, channel["gain"], remaining,
                                          reference, profile, config)
                if not selection["ok"] and config["optimization"]["reference_recovery"] == "rebuild":
                    original_failure = selection
                    recovery = rebuild_reference(epoch, snapshot["count"], current, selection["window_gain"], remaining, profile, config)
                    if recovery["ok"]:
                        reference = recovery["resources"]
                        selection = select_splits(epoch, snapshot["count"], current, gain_history, channel["gain"], remaining,
                                                  reference, profile, config)
                    selection = {**selection, "reference_recovery": recovery, "original_reference_status": original_failure["status"]}
                if not selection["ok"]:
                    failure = {"epoch": epoch, "reason": selection["status"], "selection": selection}
                    break
                if epoch == 1:
                    current = selection["candidates"][0].copy()
            if epoch % 2 == 0 and epoch < epochs:
                queue = selection["candidates"] if dynamic else [current.copy()]
            else:
                queue = [None]
            attempts, solved, problem = [], None, None
            for q in queue:
                problem = make_problem(epoch, snapshot["count"], channel["gain"], current, q, remaining, config)
                if strategy == "fixed_fixed":
                    resource = reference_resources(config)
                    audit = audit_resources(problem, resource, profile, config)
                    candidate = {"ok": audit["ok"], "status": "fixed_resources" if audit["ok"] else "fixed_resources_infeasible",
                                 "resources": resource, "metrics": evaluate_resources(problem, resource, profile, config),
                                 "trace": [], "audit": audit, "wall_s": 0.0}
                else:
                    candidate = solve_resources(problem, profile, config, reference)
                attempts.append({"next_split": q, "solve": candidate})
                if candidate["ok"]:
                    solved = candidate
                    break
            if solved is None:
                failure = {"epoch": epoch, "reason": "no_current_feasible_candidate", "attempts": attempts}
                break
            metrics, train_metrics, validation = solved["metrics"], None, None
            training_wall = time.perf_counter()
            if train:
                next_branches, train_metrics = [], []
                for uid in range(n):
                    new_model, stats = train_client_epoch(branches[uid], data, snapshot["indices"][uid], int(current[uid]), epoch, uid, config)
                    # 真实 Tensor 通信载荷必须与物理模型一致（含标签和尾批）。
                    if stats["activation_bits"] != metrics["loads"]["activation_bits"][uid] or stats["gradient_bits"] != metrics["loads"]["gradient_bits"][uid]:
                        raise AssertionError("真实训练通信量与画像不一致")
                    next_branches.append(new_model)
                    train_metrics.append(stats)
                if problem["aggregate"]:
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
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "comparison_note": "未完成任务不得将低能耗解释为节能；实时时间策略可能使用不同数据快照。"}
    if train and last_aggregation:
        summary["last_aggregated_test"] = evaluate_model(global_model, data, data["test"], config)
        summary["best_validation_model_test"] = evaluate_model(best_model, data, data["test"], config)
        if config["training"]["save_checkpoint"]:
            torch.save(checkpoint_payload(global_model), output / "last_aggregated_model.pt")
            torch.save(checkpoint_payload(best_model), output / "best_validation_model.pt")
    if train and not completed and history and config["training"]["save_checkpoint"]:
        torch.save({"branches": [checkpoint_payload(m) for m in branches], "epoch": len(history)}, output / "partial_branches.pt")
    summary["wall_s"] = time.perf_counter() - start_wall
    save_json(output / "summary.json", summary)
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
    return {"summary": summary, "history": history, "final_model": global_model if completed else None,
            "best_model": best_model, "branches": branches, "output_dir": str(output)}


# 8. 有意义的协议验证：九切点梯度、尾批、聚合通信、可行性与能量下降 ---------------
def verify_implementation(config: dict) -> dict:
    """小型 CPU 验证；返回检查结果，失败抛 AssertionError。不作为实验准确率。"""
    import torch
    from torch.nn import functional as F
    c = load_config(updates={"num_uavs": 2, "data": {"kind": "synthetic", "image_size": 32,
                            "synthetic_per_uav": 5, "synthetic_eval_size": 3, "initial_fraction": 0.6},
                            "training": {"epochs": 4, "batch_size": 2, "allow_random_init": True, "save_checkpoint": False},
                            "optimization": {"max_iterations": 15}, "output_root": config["output_root"]})
    torch.set_num_threads(c["training"]["torch_threads"])
    model, _ = initialize_model(c)
    data = prepare_data(c)
    x, y = load_batch(data, data["clients"][0][:2], "cpu")
    names = list(model["params"])
    logits = forward_suffix(model, forward_prefix(model, x, 0), 0)
    grads = dict(zip(names, torch.autograd.grad(F.cross_entropy(logits, y), list(model["params"].values()))))
    errors = []
    for split in range(9):
        split_grads, _ = split_gradients(model, x, y, split)
        errors.append(max(float((grads[name] - split_grads[name]).abs().max()) for name in names))
        for name in names:
            torch.testing.assert_close(grads[name], split_grads[name], rtol=2e-5, atol=2e-6)
        stepped = sgd_step(model, split_grads, 0.01)
        for name in names:
            torch.testing.assert_close(stepped["params"][name], model["params"][name] - 0.01 * grads[name], rtol=2e-5, atol=2e-6)
    original = {name: value.detach().clone() for name, value in model["params"].items()}
    updated, stats = train_client_epoch(model, data, data["clients"][0][:3], 4, 1, 0, c)
    assert math.isclose(stats["step_sum"], c["training"]["epoch_step_size"], rel_tol=1e-12)
    assert any(not torch.equal(updated["params"][name], model["params"][name]) for name in names)
    assert all(torch.equal(original[name], model["params"][name]) for name in names)
    average = aggregate_models([model, updated], [1, 3])
    for name in names:
        torch.testing.assert_close(average["params"][name], model["params"][name] * 0.25 + updated["params"][name] * 0.75)
    snap = epoch_snapshot(data, data["start_s"], 1, c)
    assert snap["count"] == 3
    later = epoch_snapshot(data, data["start_s"] + 1.1, 2, c)
    assert later["count"] == 4 and snap["count"] == 3
    groups = [set(ids) for ids in data["clients"]] + [set(data["val"]), set(data["test"])]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    profile = profile_splits(model, c)
    remaining = np.full(2, c["uav"]["energy_cap_j"])
    h = channel_at(data["start_s"], c)["gain"]
    reference = reference_resources(c)
    results = []
    for epoch, next_split in ((1, None), (2, np.array([8, 1])), (4, None)):
        problem = make_problem(epoch, 3, h, np.array([0, 4]), next_split, remaining, c)
        m = evaluate_resources(problem, reference, profile, c)
        mu = profile["prefix_bits"][problem["split"]] / m["up_rate_bps"] if problem["aggregate"] else np.zeros(2)
        md = profile["prefix_bits"][next_split] / m["down_rate_bps"] if next_split is not None else np.zeros(2)
        aggregate_time = c["server"]["aggregation_time_s"] if problem["aggregate"] else 0.0
        correct_duration = float(np.max(m["branch_training_s"]) + np.max(mu) + np.max(md) + aggregate_time)
        assert math.isclose(m["duration_s"], correct_duration, rel_tol=1e-12)
        if next_split is not None:
            wrong_combined_max = float(np.max(m["branch_training_s"] + mu + md) + aggregate_time)
            assert correct_duration > wrong_combined_max + 1e-3
        assert (m["loads"]["download_bits"].sum() == 0) == (next_split is None)
        if next_split is not None:
            np.testing.assert_array_equal(m["loads"]["download_bits"], profile["prefix_bits"][next_split])
        solution = solve_resources(problem, profile, c, reference)
        assert solution["ok"], solution
        assert audit_resources(problem, solution["resources"], profile, c)["ok"]
        accepted = [r["energy_j"] for r in solution["trace"] if r.get("accepted", True)]
        assert all(new <= old for old, new in zip(accepted, accepted[1:]))
        results.append({"epoch": epoch, "status": solution["status"], "energy_j": solution["metrics"]["total_energy_j"]})
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
    # 使用一个过慢的固定参考，严格 P1 应拒绝，显式重建可恢复参考可行性。
    slow = reference_resources(c)
    slow["frequency_hz"][:] = 1000.0
    selection = select_splits(2, 3, np.array([0, 4]), [h], h, remaining, slow, profile, c)
    assert not selection["ok"] and selection["status"] == "reference_training_time_lower_bound_exceeded"
    rebuilt = rebuild_reference(2, 3, np.array([0, 4]), h, remaining, profile, c)
    assert rebuilt["ok"]
    selection = select_splits(2, 3, np.array([0, 4]), [h], h, remaining, rebuilt["resources"], profile, c)
    assert selection["ok"]
    assert training_budget(default_config())["conditional_epochs"] == 100
    # 相同原始轨迹时刻必须产生完全相同的信道，不依赖调用顺序。
    np.testing.assert_array_equal(channel_at(20, c)["gain"], channel_at(20, c)["gain"])
    return {"passed": True, "split_gradient_max_abs_errors": errors,
            "checks": ["nine_split_gradients_and_updates", "input_immutability", "tail_batch_step_sum",
                       "weighted_aggregation", "frozen_epoch_snapshot", "disjoint_partitions",
                       "separate_stage_maxima", "new_prefix_payload", "final_zero_download",
                       "convex_original_feasibility", "accepted_energy_monotonicity", "infeasible_rejection",
                       "binding_battery_constraint", "strict_reference_failure_and_explicit_rebuild", "budget_rounding"],
            "solver_cases": results}


def main() -> int:
    """命令行适配层；成功返回 0，协议验证/运行错误返回 1，任务未完成返回 2。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["config", "prepare", "profile", "simulate", "train", "check", "smoke"])
    parser.add_argument("--config", help="JSON 局部参数覆盖文件")
    parser.add_argument("--output", help="本次运行的新目录；不得覆盖已有 config.json")
    parser.add_argument("--checkpoint", help="ResNet18 本地预训练参数路径")
    parser.add_argument("--allow-random-init", action="store_true", help="显式允许随机初始化，仅用于诊断/指定的消融")
    args = parser.parse_args()
    updates = {}
    if args.checkpoint or args.allow_random_init:
        updates["training"] = {}
        if args.checkpoint:
            updates["training"]["checkpoint"] = str(Path(args.checkpoint).resolve())
        if args.allow_random_init:
            updates["training"]["allow_random_init"] = True
    config = load_config(args.config, updates)
    if args.command == "config":
        print(json.dumps({"config": config, "training_budget": training_budget(config)}, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("simulate", "train", "smoke"):
        if args.command == "smoke":
            config = validate_config(merge_config(config, {"num_uavs": 2,
                "data": {"kind": "synthetic", "image_size": 32, "synthetic_per_uav": 4,
                         "synthetic_eval_size": 3, "samples_per_uav": None, "initial_fraction": 0.5,
                         "fixed_epoch_counts": None},
                "training": {"epochs": 4, "batch_size": 2, "epoch_step_size": 0.001,
                             "checkpoint": None, "allow_random_init": True, "save_checkpoint": False},
                "budget": {"mode": "empirical"}, "optimization": {"max_iterations": 15}}))
        result = run_experiment(config, train=args.command != "simulate", output=args.output)
        report = {key: result["summary"][key] for key in ("status", "mode", "dataset_kind", "epochs_completed", "epochs_planned", "system_energy_j", "processed_samples")}
        report["failure_reason"] = result["summary"]["failure"]["reason"] if result["summary"]["failure"] else None
        print(json.dumps({"output_dir": str(Path(result["output_dir"]).resolve()), "summary": report}, default=json_value, ensure_ascii=False, indent=2))
        return 0 if result["summary"]["completed"] else 2
    output = Path(args.output) if args.output else make_output_directory(config, args.command)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        report = verify_implementation(config)
        save_json(output / "verification.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        data = prepare_data(config)
        save_json(output / "data_manifest.json", data)
        save_json(output / "config.json", config)
        print(json.dumps({"output": str(output), "capacity_per_uav": data["capacity"], "audit": data["audit"],
                          "class_counts": data["class_counts"], "val": len(data["val"]), "test": len(data["test"])}, default=json_value, ensure_ascii=False, indent=2))
    else:
        import torch
        torch.set_num_threads(config["training"]["torch_threads"])
        model, _ = initialize_model(config)
        save_json(output / "split_profiles.json", profile_splits(model, config))
        print(f"模型画像已保存到 {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, AssertionError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)
