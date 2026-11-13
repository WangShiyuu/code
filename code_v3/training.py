"""数据清单、函数式 ResNet18、梯度与训练；计算画像和训练结果也在此生成。"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .config import (
    BLOCKS,
    CLASSES,
    MEAN,
    SPLIT_POINTS,
    STD,
    json_value,
    uav_parameter,
    validate_split_point,
)


# 1. 原始数据、场景划分、物理时间释放与批次读取



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


def sample_release_schedule(capacity: int, config: dict) -> tuple[np.ndarray, dict]:
    """返回每机共同可用的第 j 个样本的物理时刻，以及可复算的巡逻计划。"""
    d = config["data"]
    initial = max(1, math.floor(capacity * d["initial_fraction"]))
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


# 2. 函数式 ResNet18、梯度、SGD 与聚合



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
    params, buffers = initialize_resnet_tensors(config["seed"])
    provenance = {"kind": "random_diagnostic", "seed": config["seed"], "head_reinitialized": True}
    path = config["training"]["checkpoint"]
    if path:
        provenance = restore_model_checkpoint(params, buffers, path)
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
    x, y = load_batch(data, batch, config["training"]["device"])
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


# 3. 五切点计算与通信画像



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
    size, device = config["data"]["image_size"], config["training"]["device"]
    x = torch.linspace(-1, 1, 3 * size * size, device=device).reshape(1, 3, size, size)
    y = torch.zeros(1, dtype=torch.long, device=device)
    rows = [profile_split_point(model, config, x, y, split) for split in SPLIT_POINTS]
    return assemble_split_profile(rows, config)


# 4. 多分支训练与模型结果保存



def checkpoint_payload(model: dict) -> dict:
    """返回 detached CPU 张量字典，供 torch.save；不改变训练状态。"""
    return {group: {k: v.detach().cpu().clone() for k, v in values.items()} for group, values in model.items()}


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
