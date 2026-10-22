"""用途：准备 CBCL 九类区域分类数据，供连续飞行中的分割联邦训练读取。

运行位置：先将终端工作目录切换到 code_v1，再执行以下命令。
    python -X utf8 -B data.py
    python -X utf8 -B data.py --help
    python -X utf8 -B data.py --config config_override.json
其中 config_override.json 是用户提供的统一 JSON 覆盖配置；未指定时读取
config.py 中的默认配置。命令行可覆盖数据路径、输出目录、UAV 数及统一容量。

输入：data.root 下的 Images.zip、Annotations.zip，以及统一配置中的图像尺寸、
场景划分种子、UAV 数量、样本容量和采集参数。原始数据包保持只读。
输出：data.output/data.json 和 data.output/crops/*.png；不生成训练权重、
不执行训练，也不通过重复区域补足样本数量。

处理顺序：按原始街景隔离训练/验证/测试，再在训练区域中随机选择和等量分配。
同一训练街景的不同区域可以分给不同 UAV，但同一区域只分配一次。
client.indices 的固定排列是采集顺序，训练只能打乱其快照副本。

后续训练调用 load_data 取得五元组，调用 AcquisitionSnapshots 查询物理时刻的
累计样本前缀；在 epoch 开始时冻结快照，数据模块自身不推进训练或物理时钟。
依赖：数据准备需要 NumPy、Pillow；只有 Dataset 实际读取张量时才导入 PyTorch。
使用 -B 可避免产生字节码缓存，-X utf8 用于统一终端的中文编码。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from config import load_config, validate_config, resolve_collection, available_samples_at


# 类别顺序同时决定整数标签、模型分类头及清单校验口径，不能随意重新排序。
CLASSES = ("bicycle", "building", "car", "pedestrian", "road", "sidewalk", "sky", "store", "tree")
# 读取训练张量时使用固定 RGB 归一化参数，数据准备阶段只保存 RGB PNG。
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DATA_PROTOCOL = "code_v1_scene_holdout_iid_equal_v1"


def write_json(path, value):
    """将可 JSON 序列化的值写成 UTF-8 文件，返回 None。

    自动创建父目录，保留中文和缩进；拒绝 NaN/Infinity，防止下游读取到
    不符合标准 JSON 的数值。清单的原子发布由 prepare_data 另行完成。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value, name, minimum=1):
    """校验整数下界并返回原值；name 用于解释出错的配置字段。

    显式排除 bool，因为 Python 将布尔值视为整数子类，而容量、尺寸等字段
    不应把 True/False 当成 1/0 使用。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是至少 {minimum} 的整数")
    return value


def _basename(name):
    """统一 ZIP/XML 内的路径分隔符，返回不含目录的文件名。

    图片包和标注可能分别使用 Windows 或 POSIX 路径，匹配时只比较文件名。
    重名文件随后由 read_cbcl 拒绝，避免错误地关联标注。
    """
    return PurePosixPath(name.replace("\\", "/")).name


def read_cbcl(data_root):
    """从原始包建立区域样本记录，返回 (records, report)。

    data_root 必须包含 Images.zip 和 Annotations.zip。本函数不整包解压、
    不写缓存；每个有效 polygon 的外接矩形构成一个分类样本。
    records 保留区域编号、所属街景、类别、ZIP 成员路径及裁剪框，report
    区分原图数量、有效街景数量、有效区域数和跳过的无效标注数。
    """
    root = Path(data_root).expanduser().resolve()
    for name in ("Images.zip", "Annotations.zip"):
        if not (root / name).is_file():
            raise ValueError(f"CBCL 缺少 {root / name}，请修改 data.root")
    records, skipped = [], Counter()
    with zipfile.ZipFile(root / "Images.zip") as images, zipfile.ZipFile(root / "Annotations.zip") as annotations:
        # 先构造文件名到 ZIP 成员的唯一映射，禁止同名图片造成标注歧义。
        members = {}
        for member in images.namelist():
            if PurePosixPath(member).suffix.lower() in (".jpg", ".jpeg", ".png"):
                name = _basename(member)
                if name in members:
                    raise ValueError(f"ZIP 中存在重复图片文件名：{name}")
                members[name] = member
        seen = set()
        # 排序后的标注读取顺序保持稳定，避免 ZIP 内部顺序影响可复现划分。
        xmls = sorted(name for name in annotations.namelist() if name.lower().endswith(".xml"))
        for xml_name in xmls:
            annotation = ET.fromstring(annotations.read(xml_name))
            filename = _basename((annotation.findtext("filename") or "").strip())
            if filename not in members or filename in seen:
                raise ValueError(f"标注图片缺失或重复：{xml_name}: {filename}")
            seen.add(filename)
            # 此处只读取原图尺寸以裁剪边界；真正的区域解码在准备缓存时进行。
            with images.open(members[filename]) as stream, Image.open(stream) as image:
                width, height = image.size
            for object_id, obj in enumerate(annotation.findall("object")):
                label = (obj.findtext("name") or "").strip().lower()
                if (obj.findtext("deleted") or "0").strip() == "1" or not label:
                    skipped["deleted_or_empty"] += 1
                    continue
                if label not in CLASSES:
                    raise ValueError(f"未知 CBCL 类别 {label!r}：{xml_name}")
                try:
                    points = [(float(pt.findtext("x")), float(pt.findtext("y"))) for pt in obj.findall("polygon/pt")]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"标注坐标损坏：{xml_name}，对象 {object_id}") from exc
                if len(points) < 3 or not all(math.isfinite(v) for pt in points for v in pt):
                    skipped["invalid_polygon"] += 1
                    continue
                xs, ys = zip(*points)
                # 标注坐标从 1 开始；PIL 裁剪从 0 开始且不包含右/下端点。
                # 外接框同时限制在原图范围内，避免输出越界区域或零面积样本。
                box = [max(0, math.floor(min(xs) - 1)), max(0, math.floor(min(ys) - 1)),
                       min(width, math.ceil(max(xs))), min(height, math.ceil(max(ys)))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    skipped["empty_box"] += 1
                    continue
                records.append(dict(sample_id=f"{filename}:object{object_id}", scene_id=filename,
                                    label=CLASSES.index(label), class_name=label,
                                    member=members[filename], bbox=box, annotation=xml_name))
    if not records:
        raise ValueError("CBCL 中没有可用的多边形区域")
    # usable_samples 计的是区域，不等同于独立原始街景图像数量。
    report = dict(dataset="CBCL StreetScenes", task="polygon_bounding_box_classification",
                  images=len(members), annotation_xmls=len(xmls), usable_samples=len(records),
                  usable_scenes=len({row["scene_id"] for row in records}), skipped=dict(skipped))
    return records, report


def split_scenes(records, seed, val_fraction=0.15, test_fraction=0.15):
    """按原始场景划分 holdout，返回 train/val/test 三个区域记录列表。

    seed 控制场景排列；默认按街景数量约 70/15/15 分割，不强求区域样本数
    也满足同一比例。同一 scene_id 的全部区域始终进入同一个集合，防止
    模型在训练中看到验证或测试区域所属的同一原图。
    """
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (val_fraction, test_fraction)):
        raise ValueError("验证和测试比例必须是有限数")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("验证和测试比例必须大于零且总和小于一")
    # 先固定候选场景顺序再随机排列；此步骤与 UAV 数量、容量选择相互独立。
    scenes = sorted({row["scene_id"] for row in records})
    if len(scenes) < 3:
        raise ValueError("至少需要三个有效场景隔离训练、验证和测试")
    scenes = np.random.default_rng(seed).permutation(scenes).tolist()
    n_val = max(1, round(len(scenes) * val_fraction))
    n_test = max(1, round(len(scenes) * test_fraction))
    if n_val + n_test >= len(scenes):
        raise ValueError("当前比例使训练场景为空")
    # 未被选作验证或测试的街景归入训练池，不对单个区域另做随机 holdout。
    mapping = {scene: "val" for scene in scenes[:n_val]}
    mapping.update({scene: "test" for scene in scenes[n_val:n_val + n_test]})
    return {split: [row for row in records if mapping.get(row["scene_id"], "train") == split]
            for split in ("train", "val", "test")}


def resolve_sizes(available, num_uavs, samples_per_uav=None):
    """解析每机统一容量，返回长度为 num_uavs 的相同整数列表。

    available 是隔离 holdout 后训练池的区域数。未给统一容量时只保留
    N*floor(pool/N) 个区域，其余随后排除；显式容量必须是单个正整数，
    且所有 UAV 的总需求不得超过训练池，绝不复制样本补足。
    """
    _positive_integer(available, "训练池大小")
    _positive_integer(num_uavs, "num_uavs")
    size = available // num_uavs if samples_per_uav is None else _positive_integer(samples_per_uav, "samples_per_uav")
    if size < 1 or size * num_uavs > available:
        raise ValueError(f"{available} 个训练样本无法为 {num_uavs} 架 UAV 各分配 {size} 个不同样本")
    return [size] * num_uavs


def partition_uavs(sample_count, num_uavs, seed):
    """对已经选中的训练区域做 IID 等量分配，返回每机索引列表。

    索引相对于最终训练子集，而非包含 holdout 的 manifest.samples 全表。
    sample_count 必须已能被 num_uavs 整除；一次随机排列被切成互不重叠的
    等长片段，各片段的顺序同时定义该 UAV 的固定采集顺序。
    """
    _positive_integer(sample_count, "sample_count")
    _positive_integer(num_uavs, "num_uavs")
    if sample_count % num_uavs:
        raise ValueError("分配前必须先剔除不能均分的训练尾部")
    # 随机 IID 分配不人为制造类别偏斜，也不保证有限样本的类别比例完全相同。
    order = np.random.default_rng(seed).permutation(sample_count).tolist()
    size = sample_count // num_uavs
    if not size:
        raise ValueError("每架 UAV 的训练序列必须非空")
    return [order[uav * size:(uav + 1) * size] for uav in range(num_uavs)]


def _fingerprint(manifest):
    """返回数据核心契约的 SHA-256 指纹，用于发现清单内容变化。

    指纹绑定 schema、预处理设置、类别顺序、样本记录和 UAV 索引排列；
    因此更改采集顺序也会改变指纹。它不是原始大 ZIP 的内容哈希，原始包
    的缓存匹配另使用 settings.source_files 内的大小和修改时间。
    """
    value = {key: manifest[key] for key in ("schema", "settings", "classes", "samples", "clients")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_manifest(manifest):
    """校验完整清单的数据契约；成功返回 None，失败抛出 ValueError。

    除 schema 1 和 v1 协议外，还检查指纹、归一化、样本唯一性、场景隔离、
    缓存相对路径、零起始 UAV 编号、训练局部索引及各层统计。此函数只校验
    清单，不读取 PNG 内容；实体缓存校验由 _validate_crops 负责。
    """
    try:
        # 旧 non-IID 或不等量清单不能仅凭相同 schema 编号进入新采集协议。
        if not isinstance(manifest, dict) or manifest.get("schema") != 1 or manifest.get("classes") != list(CLASSES):
            raise ValueError("仅支持 CBCL 九分类 schema 1 清单")
        settings = manifest["settings"]
        if settings.get("protocol") != DATA_PROTOCOL or settings.get("partition") != "iid":
            raise ValueError("清单不是 code_v1 的 IID 等量采集协议，请准备新的数据目录")
        num_uavs = _positive_integer(settings["uavs"], "settings.uavs")
        _positive_integer(settings["image_size"], "image_size", 32)
        if manifest.get("mean") != list(MEAN) or manifest.get("std") != list(STD):
            raise ValueError("清单归一化参数不一致")
        if manifest.get("fingerprint") != _fingerprint(manifest):
            raise ValueError("data.json 指纹不一致，请准备新的数据目录")
        # 逐区域检查：原图不能跨集合，同一区域和同一缓存图片不能重复使用。
        scenes, sample_ids, crops, counts = {}, set(), set(), Counter()
        sample_rows = manifest["samples"]
        if not isinstance(sample_rows, list):
            raise ValueError("samples 必须为列表")
        for row in sample_rows:
            sample_id, scene, split = row["sample_id"], row["scene_id"], row["split"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError("样本编号为空或重复")
            if not isinstance(scene, str) or not scene or split not in ("train", "val", "test"):
                raise ValueError("非法场景或数据集合")
            label = row["label"]
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(CLASSES) or CLASSES[label] != row["class_name"]:
                raise ValueError("类别和标签不一致")
            sample_ids.add(sample_id)
            if scenes.setdefault(scene, split) != split:
                raise ValueError("同一原始街景出现在不同集合，存在评价泄漏")
            raw = row["crop"]
            if not isinstance(raw, str):
                raise ValueError("缓存图片路径必须是字符串")
            # 同时识别两种路径风格，拒绝绝对路径、盘符和 .. 越出缓存目录。
            crop = PurePosixPath(raw.replace("\\", "/"))
            if (crop.is_absolute() or PureWindowsPath(raw).drive or ".." in crop.parts
                    or len(crop.parts) != 2 or crop.parts[0] != "crops" or crop.suffix.lower() != ".png"):
                raise ValueError("缓存图片必须是 crops 内的相对 PNG 路径")
            if str(crop) in crops:
                raise ValueError("多个样本引用相同缓存图片")
            crops.add(str(crop))
            counts[split] += 1
        clients = manifest["clients"]
        if not isinstance(clients, list) or len(clients) != num_uavs:
            raise ValueError("清单 UAV 数量不一致")
        # client.indices 指向过滤后的 train_rows；不是全 samples 的行号。
        indices, sizes = [], []
        train_rows = [row for row in sample_rows if row["split"] == "train"]
        for uav, client in enumerate(clients):
            if isinstance(client["uav_id"], bool) or not isinstance(client["uav_id"], int) or client["uav_id"] != uav:
                raise ValueError("UAV 编号必须从零顺序排列")
            own = client["indices"]
            if not isinstance(own, list) or not own or any(isinstance(i, bool) or not isinstance(i, int) for i in own):
                raise ValueError("UAV indices 必须为非空整数列表")
            if (isinstance(client["samples"], bool) or not isinstance(client["samples"], int)
                    or client["samples"] != len(own)):
                raise ValueError("UAV 样本数与索引不一致")
            if any(i < 0 or i >= len(train_rows) for i in own):
                raise ValueError("UAV 索引超出训练子集范围")
            expected = {name: sum(train_rows[i]["label"] == label for i in own)
                        for label, name in enumerate(CLASSES)}
            if client["class_counts"] != expected:
                raise ValueError("UAV 类别计数与训练索引不一致")
            indices.extend(own)
            sizes.append(len(own))
        # 完整覆盖且无重复，才可把这些固定序列安全解释为互不重复的采集流。
        if len(set(sizes)) != 1 or sorted(indices) != list(range(counts["train"])):
            raise ValueError("UAV 必须等量、不重叠且恰好覆盖保留训练集一次")
        if any(not counts[name] for name in ("train", "val", "test")):
            raise ValueError("训练、验证和测试集合均须非空")
        # 重新计算统计，避免读者从与样本本身不一致的汇总字段推断实验规模。
        if manifest["subset_counts"] != {name: counts[name] for name in ("train", "val", "test")}:
            raise ValueError("集合样本计数不一致")
        expected_scenes = {split: len({r["scene_id"] for r in sample_rows if r["split"] == split})
                           for split in ("train", "val", "test")}
        if manifest["scene_counts"] != expected_scenes:
            raise ValueError("集合场景计数不一致")
        expected_classes = {split: dict(Counter(r["class_name"] for r in sample_rows if r["split"] == split))
                            for split in ("train", "val", "test")}
        if manifest["class_counts"] != expected_classes:
            raise ValueError("集合类别计数不一致")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("数据清单缺少必要字段或字段类型不正确") from exc


def _read_manifest(path):
    """读取文件或数据目录内的 data.json，返回 (绝对路径, 已校验清单)。

    接受带 BOM 的 UTF-8 JSON；缺失、格式错误及契约不符均明确报错，
    不自动生成数据或回退到另一份可能改变实验划分的清单。
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "data.json"
    if not path.is_file():
        raise ValueError(f"未找到数据清单：{path}；请先运行 python data.py")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据清单不是有效 JSON：{path}") from exc
    validate_manifest(manifest)
    return path, manifest


def _validate_crops(root, manifest):
    """逐个校验清单引用的 PNG，成功返回 None。

    root 是数据缓存目录。除文件存在外，还要求解析后的实际路径留在该目录
    内，PNG 为 RGB 且尺寸与准备配置一致，并由 Pillow 检查文件结构；
    因此不能将一个不完整目录误认为可复用缓存。
    """
    size = manifest["settings"]["image_size"]
    for row in manifest["samples"]:
        path = root / row["crop"]
        try:
            # resolve 后再次检查目录范围，也拒绝指向外部文件的符号链接。
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"缓存图片缺失或位于缓存目录外：{path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (size, size):
                    raise ValueError(f"缓存图片尺寸、模式或格式不一致：{path}")
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"缓存图片损坏：{path}") from exc


def prepare_data(config):
    """按统一配置准备或复用数据，返回最终 data.json 的绝对 Path。

    config 提供 mission.num_uavs、data 路径/划分/尺寸/容量及采集设置。
    新数据先完成场景隔离和等量随机分配，再写 RGB PNG，最后发布清单；
    已有目录只有参数、原始包元信息、清单和 PNG 均通过检查才会复用。
    容量或初始采集配置冲突会在写图片前报错，函数不启动后续训练。
    """
    # 第一步：检查统一参数，并只把影响数据准备的设置写入缓存契约。
    # D0 和采集速率由 resolve_collection 单独核实，改变它们不必重新裁剪。
    validate_config(config)
    data = config["data"]
    root = Path(data["root"]).expanduser().resolve()
    destination = Path(data["output"]).expanduser().resolve()
    num_uavs = config["mission"]["num_uavs"]
    settings = dict(source_root=str(root), image_size=data["image_size"], uavs=num_uavs,
                    partition="iid", alpha=None, seed=data["seed"], train_samples=None,
                    samples_per_uav=None if data["samples_per_uav"] is None else [data["samples_per_uav"]] * num_uavs,
                    val_fraction=data["val_fraction"], test_fraction=data["test_fraction"],
                    protocol=DATA_PROTOCOL, acquisition_order="client.indices; immutable random permutation")
    sources = {name: root / name for name in ("Images.zip", "Annotations.zip")}
    for path in sources.values():
        if not path.is_file():
            raise ValueError(f"缺少 {path}，请修改 data.root")
    # 对约 2 GB 原始图像包使用文件元信息检查，避免每次读取整包计算内容哈希。
    settings["source_files"] = {name: dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                                for name, path in sources.items()}
    manifest_path = destination / "data.json"
    if manifest_path.is_file():
        # 第二步（缓存分支）：不覆盖旧结果，不静默接受参数不同或图片不完整的缓存。
        _, manifest = _read_manifest(manifest_path)
        if manifest["settings"] != settings:
            raise ValueError("此输出目录已有不同参数或原始数据的数据缓存，请选择新的 data.output")
        resolve_collection(config, manifest)
        _validate_crops(destination, manifest)
        return manifest_path
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("数据输出目录非空且没有完成清单，请选择新的 data.output")
    # 第二步（新建分支）：先隔离原图，再决定保留的训练容量，holdout 不参与补足。
    records, report = read_cbcl(root)
    subsets = split_scenes(records, data["seed"], data["val_fraction"], data["test_fraction"])
    available = len(subsets["train"])
    sizes = resolve_sizes(available, num_uavs, data["samples_per_uav"])
    order = np.random.default_rng(data["seed"] + 1).permutation(available).tolist()
    # 默认排除均分余数；显式较小容量还会排除多余训练区域，全部记录用于审计。
    selected, excluded = order[:sum(sizes)], order[sum(sizes):]
    excluded_ids = [subsets["train"][i]["sample_id"] for i in excluded]
    subsets["train"] = [subsets["train"][i] for i in selected]
    # 子种子分开控制训练区域选择和 UAV 分配，使采集序列可重复生成。
    client_indices = partition_uavs(sum(sizes), num_uavs, data["seed"] + 2)
    samples = [{**row, "split": split} for split in ("train", "val", "test") for row in subsets[split]]
    # 第三步：沿用 schema 1 的样本/客户端字段和 crops 相对路径，便于训练读取。
    # 按原始 ZIP 成员分组，同一街景只解码一次，再生成属于它的各区域缓存。
    grouped = defaultdict(list)
    for index, row in enumerate(samples):
        row["crop"] = f"crops/{index:07d}.png"
        grouped[row["member"]].append(row)
    clients = [dict(uav_id=uav, indices=indices, samples=len(indices),
                    class_counts={name: sum(subsets["train"][i]["label"] == label for i in indices)
                                  for label, name in enumerate(CLASSES)})
               for uav, indices in enumerate(client_indices)]
    manifest = dict(schema=1, settings=settings, classes=list(CLASSES), mean=list(MEAN), std=list(STD),
                    samples=samples, clients=clients, dataset_report=report,
                    split_protocol="custom scene-group holdout; training crops may belong to different UAVs",
                    subset_counts={split: len(rows) for split, rows in subsets.items()},
                    scene_counts={split: len({row["scene_id"] for row in rows}) for split, rows in subsets.items()},
                    class_counts={split: dict(Counter(row["class_name"] for row in rows)) for split, rows in subsets.items()},
                    allocation_report=dict(train_pool_samples=available, selected_train_samples=sum(sizes),
                                           excluded_train_samples=len(excluded), equal_split_remainder=available % num_uavs,
                                           excluded_sample_ids=excluded_ids,
                                           selection="random without replacement; excluded samples never enter holdout"))
    # 在任何图片写入之前检查清单和采集容量，避免非法 D0 留下大量半成品。
    manifest["fingerprint"] = _fingerprint(manifest)
    validate_manifest(manifest)
    resolve_collection(config, manifest)  # 在裁剪、创建输出前发现 D0 或容量配置冲突。
    # 第四步：只裁剪已选区域；被排除的训练区域不进入 PNG 清单或评价集合。
    (destination / "crops").mkdir(parents=True)
    with zipfile.ZipFile(sources["Images.zip"]) as archive:
        for member, rows in sorted(grouped.items()):
            with archive.open(member) as stream, Image.open(stream) as source:
                with source.convert("RGB") as rgb:
                    for row in rows:
                        with rgb.crop(tuple(row["bbox"])) as crop:
                            with crop.resize((data["image_size"], data["image_size"]), Image.Resampling.BILINEAR) as resized:
                                resized.save(destination / row["crop"])
    # 最后发布清单：训练端不会把尚未写完 PNG 的目录当作完整数据集。
    # 若中途失败，下次会拒绝残留非空目录，要求明确选择新的输出目录。
    write_json(destination / "data.pending.json", manifest)
    (destination / "data.pending.json").replace(manifest_path)
    return manifest_path


class CBCLDataset:
    """提供 PyTorch 可读取的区域数据集，构造时不导入 PyTorch。

    保存数据根目录、对应集合的记录和读取尺寸；实际访问样本时执行
    RGB、[0,1] 浮点缩放、HWC 到 CHW 转换及固定均值/标准差归一化。
    """
    def __init__(self, root, rows, image_size):
        """保存输入引用和读取尺寸；不预加载图片，也不推进采集状态。"""
        self.root, self.rows, self.image_size = Path(root), rows, image_size

    def __len__(self):
        """返回本 Dataset 所含区域数量，供 DataLoader 计算遍历长度。"""
        return len(self.rows)

    def __getitem__(self, index):
        """读取一个区域，返回 (归一化 CHW 张量, long 类型类别标签)。

        index 是当前 Dataset 的局部行号。PyTorch 在这里延迟导入，
        让只准备数据、检查清单或查询采集快照的调用不依赖 PyTorch。
        """
        import torch
        row = self.rows[int(index)]
        with Image.open(self.root / row["crop"]) as image:
            with image.convert("RGB") as rgb:
                if rgb.size != (self.image_size, self.image_size):
                    with rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR) as resized:
                        array = np.array(resized, dtype=np.float32) / 255.0
                else:
                    array = np.array(rgb, dtype=np.float32) / 255.0
        # Pillow/NumPy 的 HWC 图像转为模型使用的 CHW；均值/标准差按通道广播。
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean, std = tensor.new_tensor(MEAN)[:, None, None], tensor.new_tensor(STD)[:, None, None]
        return (tensor - mean) / std, torch.tensor(row["label"], dtype=torch.long)


def load_data(manifest_path, image_size=None):
    """保持训练读取五元组：train、val、test、每机局部索引、manifest。

    manifest_path 可指向数据目录或 data.json，image_size 可覆盖张量读取尺寸。
    先验证原始缓存尺寸和完整性，再构建 Dataset；索引返回副本，避免调用方
    打乱训练顺序时意外修改清单记录的固定采集序列。此处不导入 PyTorch。
    """
    manifest_path, manifest = _read_manifest(manifest_path)
    image_size = manifest["settings"]["image_size"] if image_size is None else image_size
    _positive_integer(image_size, "image_size", 32)
    _validate_crops(manifest_path.parent, manifest)
    # 过滤样本时保持清单内相对顺序，确保 UAV 局部索引与 train Dataset 一致。
    datasets = [CBCLDataset(manifest_path.parent, [r for r in manifest["samples"] if r["split"] == split], image_size)
                for split in ("train", "val", "test")]
    return (*datasets, [client["indices"][:] for client in manifest["clients"]], manifest)


class AcquisitionSnapshots:
    """按物理时间返回固定采集序列的累计前缀，不改变模型或时钟。

    内部使用不可变的索引元组保护释放顺序，每次查询返回新的列表。
    训练调用方在 epoch 开始时查询一次并持有该快照，期间新增样本留待
    下一 epoch；本类不根据训练完成进度重新定义数据到达时间。
    """
    def __init__(self, manifest_or_path, collection):
        """绑定已校验清单和 resolve_collection 返回的统一采集参数。

        输入可为清单字典或文件/目录路径。UAV 数和每机最大容量必须与清单
        一致；初始量、速率等时间采样规则由 available_samples_at 检查。
        """
        if isinstance(manifest_or_path, dict):
            manifest = manifest_or_path
            validate_manifest(manifest)
        else:
            _, manifest = _read_manifest(manifest_or_path)
        # 独立冻结清单中的序列，外部修改原始字典也不会改变既定采集顺序。
        self._indices = tuple(tuple(client["indices"]) for client in manifest["clients"])
        try:
            self.collection = dict(collection)
            num_uavs = _positive_integer(self.collection["num_uavs"], "collection.num_uavs")
            capacity = _positive_integer(self.collection["max_samples_per_uav"], "collection.max_samples_per_uav")
        except (KeyError, TypeError) as exc:
            raise ValueError("采集配置缺少必要字段或类型不正确") from exc
        if num_uavs != len(self._indices) or capacity != len(self._indices[0]):
            raise ValueError("采集配置与数据清单的 UAV 数量或容量不一致")
        available_samples_at(0.0, self.collection)

    def snapshot_indices(self, time_s, uav_id=None):
        """查询时刻 time_s 的可训练索引副本。

        time_s 单位为秒，以任务起点为零；样本数统一由配置模块的累计公式
        确定并受最大容量限制。uav_id 省略时返回所有 UAV 的索引列表，
        指定零起始编号时只返回该 UAV 的列表；重复查询不消耗数据。
        """
        count = available_samples_at(time_s, self.collection)
        if uav_id is None:
            return [list(indices[:count]) for indices in self._indices]
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or not 0 <= uav_id < len(self._indices):
            raise ValueError("uav_id 必须是有效的零起始 UAV 编号")
        return list(self._indices[uav_id][:count])


def main():
    """命令行入口：读取统一配置、应用显式覆盖，准备数据并打印规模。

    --help 仅显示参数帮助；--config 指向统一 JSON 覆盖文件，其他显式
    参数优先于配置文件。可预期的数据或文件错误转为简洁的非零退出信息，
    不自动清理或覆盖与当前配置不匹配的旧输出目录。
    """
    import argparse
    parser = argparse.ArgumentParser(description="CBCL场景隔离、IID等量分配及采集顺序准备")
    parser.add_argument("--config", help="统一JSON配置覆盖")
    parser.add_argument("--data-root", help="Images.zip和Annotations.zip所在目录")
    parser.add_argument("--output", help="数据清单和crops的输出目录")
    parser.add_argument("--num-uavs", "--uavs", dest="num_uavs", type=int)
    parser.add_argument("--samples-per-uav", type=int, help="所有UAV共同容量")
    args = parser.parse_args()
    try:
        # 命令行仅覆盖用户明确提供的值，未提供项沿用统一配置。
        config = load_config(args.config)
        if args.num_uavs is not None:
            config["mission"]["num_uavs"] = args.num_uavs
        for key, value in (("root", args.data_root), ("output", args.output)):
            if value is not None:
                config["data"][key] = str(Path(value).expanduser().resolve())
        if args.samples_per_uav is not None:
            config["data"]["samples_per_uav"] = args.samples_per_uav
        # 参数检查通过后才进入缓存复用或新建流程，不启动训练。
        validate_config(config)
        path = prepare_data(config)
        _, manifest = _read_manifest(path)
        print(f"数据清单：{path}\n集合样本数：{manifest['subset_counts']}\n"
              f"各 UAV 样本数：{[client['samples'] for client in manifest['clients']]}\n"
              f"排除训练池样本数：{manifest['allocation_report']['excluded_train_samples']}")
    except (ValueError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        parser.exit(2, f"数据准备失败：{exc}\n")


if __name__ == "__main__":
    main()
