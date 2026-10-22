"""CBCL 九类区域分类：按场景隔离、IID 等量分配及连续采集快照。

修改 config.py 的统一配置后运行 ``python data.py``。本模块只准备数据，
不启动训练；PyTorch 仅在 CBCLDataset.__getitem__ 读取张量时导入。
client.indices 的固定排列是采集顺序，训练只能打乱其快照副本。
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


CLASSES = ("bicycle", "building", "car", "pedestrian", "road", "sidewalk", "sky", "store", "tree")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DATA_PROTOCOL = "code_v1_scene_holdout_iid_equal_v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是至少 {minimum} 的整数")
    return value


def _basename(name):
    return PurePosixPath(name.replace("\\", "/")).name


def read_cbcl(data_root):
    """只读取原始 ZIP/XML；一个有效 polygon 外接框是一个分类样本。"""
    root = Path(data_root).expanduser().resolve()
    for name in ("Images.zip", "Annotations.zip"):
        if not (root / name).is_file():
            raise ValueError(f"CBCL 缺少 {root / name}，请修改 data.root")
    records, skipped = [], Counter()
    with zipfile.ZipFile(root / "Images.zip") as images, zipfile.ZipFile(root / "Annotations.zip") as annotations:
        members = {}
        for member in images.namelist():
            if PurePosixPath(member).suffix.lower() in (".jpg", ".jpeg", ".png"):
                name = _basename(member)
                if name in members:
                    raise ValueError(f"ZIP 中存在重复图片文件名：{name}")
                members[name] = member
        seen = set()
        xmls = sorted(name for name in annotations.namelist() if name.lower().endswith(".xml"))
        for xml_name in xmls:
            annotation = ET.fromstring(annotations.read(xml_name))
            filename = _basename((annotation.findtext("filename") or "").strip())
            if filename not in members or filename in seen:
                raise ValueError(f"标注图片缺失或重复：{xml_name}: {filename}")
            seen.add(filename)
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
    report = dict(dataset="CBCL StreetScenes", task="polygon_bounding_box_classification",
                  images=len(members), annotation_xmls=len(xmls), usable_samples=len(records),
                  usable_scenes=len({row["scene_id"] for row in records}), skipped=dict(skipped))
    return records, report


def split_scenes(records, seed, val_fraction=0.15, test_fraction=0.15):
    """按原始场景划分约 70/15/15，区域样本比例不必精确相同。"""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (val_fraction, test_fraction)):
        raise ValueError("验证和测试比例必须是有限数")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("验证和测试比例必须大于零且总和小于一")
    scenes = sorted({row["scene_id"] for row in records})
    if len(scenes) < 3:
        raise ValueError("至少需要三个有效场景隔离训练、验证和测试")
    scenes = np.random.default_rng(seed).permutation(scenes).tolist()
    n_val = max(1, round(len(scenes) * val_fraction))
    n_test = max(1, round(len(scenes) * test_fraction))
    if n_val + n_test >= len(scenes):
        raise ValueError("当前比例使训练场景为空")
    mapping = {scene: "val" for scene in scenes[:n_val]}
    mapping.update({scene: "test" for scene in scenes[n_val:n_val + n_test]})
    return {split: [row for row in records if mapping.get(row["scene_id"], "train") == split]
            for split in ("train", "val", "test")}


def resolve_sizes(available, num_uavs, samples_per_uav=None):
    """统一容量；默认只保留 N*floor(pool/N)，绝不复制样本补足。"""
    _positive_integer(available, "训练池大小")
    _positive_integer(num_uavs, "num_uavs")
    size = available // num_uavs if samples_per_uav is None else _positive_integer(samples_per_uav, "samples_per_uav")
    if size < 1 or size * num_uavs > available:
        raise ValueError(f"{available} 个训练样本无法为 {num_uavs} 架 UAV 各分配 {size} 个不同样本")
    return [size] * num_uavs


def partition_uavs(sample_count, num_uavs, seed):
    """随机不放回等量分配；返回的固定排列同时定义每机采集顺序。"""
    _positive_integer(sample_count, "sample_count")
    _positive_integer(num_uavs, "num_uavs")
    if sample_count % num_uavs:
        raise ValueError("分配前必须先剔除不能均分的训练尾部")
    order = np.random.default_rng(seed).permutation(sample_count).tolist()
    size = sample_count // num_uavs
    if not size:
        raise ValueError("每架 UAV 的训练序列必须非空")
    return [order[uav * size:(uav + 1) * size] for uav in range(num_uavs)]


def _fingerprint(manifest):
    value = {key: manifest[key] for key in ("schema", "settings", "classes", "samples", "clients")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_manifest(manifest):
    """校验 schema 1、场景隔离、固定采集顺序及 IID 等量覆盖。"""
    try:
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
        if len(set(sizes)) != 1 or sorted(indices) != list(range(counts["train"])):
            raise ValueError("UAV 必须等量、不重叠且恰好覆盖保留训练集一次")
        if any(not counts[name] for name in ("train", "val", "test")):
            raise ValueError("训练、验证和测试集合均须非空")
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
    size = manifest["settings"]["image_size"]
    for row in manifest["samples"]:
        path = root / row["crop"]
        try:
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"缓存图片缺失或位于缓存目录外：{path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (size, size):
                    raise ValueError(f"缓存图片尺寸、模式或格式不一致：{path}")
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"缓存图片损坏：{path}") from exc


def prepare_data(config):
    """按统一配置准备 PNG/清单；已有目录只在完整匹配并通过校验时复用。"""
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
    settings["source_files"] = {name: dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                                for name, path in sources.items()}
    manifest_path = destination / "data.json"
    if manifest_path.is_file():
        _, manifest = _read_manifest(manifest_path)
        if manifest["settings"] != settings:
            raise ValueError("此输出目录已有不同参数或原始数据的数据缓存，请选择新的 data.output")
        resolve_collection(config, manifest)
        _validate_crops(destination, manifest)
        return manifest_path
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("数据输出目录非空且没有完成清单，请选择新的 data.output")
    records, report = read_cbcl(root)
    subsets = split_scenes(records, data["seed"], data["val_fraction"], data["test_fraction"])
    available = len(subsets["train"])
    sizes = resolve_sizes(available, num_uavs, data["samples_per_uav"])
    order = np.random.default_rng(data["seed"] + 1).permutation(available).tolist()
    selected, excluded = order[:sum(sizes)], order[sum(sizes):]
    excluded_ids = [subsets["train"][i]["sample_id"] for i in excluded]
    subsets["train"] = [subsets["train"][i] for i in selected]
    client_indices = partition_uavs(sum(sizes), num_uavs, data["seed"] + 2)
    samples = [{**row, "split": split} for split in ("train", "val", "test") for row in subsets[split]]
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
    manifest["fingerprint"] = _fingerprint(manifest)
    validate_manifest(manifest)
    resolve_collection(config, manifest)  # 在裁剪、创建输出前发现 D0 或容量配置冲突。
    (destination / "crops").mkdir(parents=True)
    with zipfile.ZipFile(sources["Images.zip"]) as archive:
        for member, rows in sorted(grouped.items()):
            with archive.open(member) as stream, Image.open(stream) as source:
                with source.convert("RGB") as rgb:
                    for row in rows:
                        with rgb.crop(tuple(row["bbox"])) as crop:
                            with crop.resize((data["image_size"], data["image_size"]), Image.Resampling.BILINEAR) as resized:
                                resized.save(destination / row["crop"])
    write_json(destination / "data.pending.json", manifest)
    (destination / "data.pending.json").replace(manifest_path)
    return manifest_path


class CBCLDataset:
    """PNG -> RGB/CHW 张量；构建 Dataset 本身也不导入 PyTorch。"""
    def __init__(self, root, rows, image_size):
        self.root, self.rows, self.image_size = Path(root), rows, image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[int(index)]
        with Image.open(self.root / row["crop"]) as image:
            with image.convert("RGB") as rgb:
                if rgb.size != (self.image_size, self.image_size):
                    with rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR) as resized:
                        array = np.array(resized, dtype=np.float32) / 255.0
                else:
                    array = np.array(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean, std = tensor.new_tensor(MEAN)[:, None, None], tensor.new_tensor(STD)[:, None, None]
        return (tensor - mean) / std, torch.tensor(row["label"], dtype=torch.long)


def load_data(manifest_path, image_size=None):
    """保持原五元组：train、val、test、每机训练局部索引、manifest。"""
    manifest_path, manifest = _read_manifest(manifest_path)
    image_size = manifest["settings"]["image_size"] if image_size is None else image_size
    _positive_integer(image_size, "image_size", 32)
    _validate_crops(manifest_path.parent, manifest)
    datasets = [CBCLDataset(manifest_path.parent, [r for r in manifest["samples"] if r["split"] == split], image_size)
                for split in ("train", "val", "test")]
    return (*datasets, [client["indices"][:] for client in manifest["clients"]], manifest)


class AcquisitionSnapshots:
    """无副作用地查询累计样本前缀；epoch 的冻结与时钟推进由训练调用方负责。"""
    def __init__(self, manifest_or_path, collection):
        if isinstance(manifest_or_path, dict):
            manifest = manifest_or_path
            validate_manifest(manifest)
        else:
            _, manifest = _read_manifest(manifest_or_path)
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
        count = available_samples_at(time_s, self.collection)
        if uav_id is None:
            return [list(indices[:count]) for indices in self._indices]
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or not 0 <= uav_id < len(self._indices):
            raise ValueError("uav_id 必须是有效的零起始 UAV 编号")
        return list(self._indices[uav_id][:count])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CBCL场景隔离、IID等量分配及采集顺序准备")
    parser.add_argument("--config", help="统一JSON配置覆盖")
    parser.add_argument("--data-root", help="Images.zip和Annotations.zip所在目录")
    parser.add_argument("--output", help="数据清单和crops的输出目录")
    parser.add_argument("--num-uavs", "--uavs", dest="num_uavs", type=int)
    parser.add_argument("--samples-per-uav", type=int, help="所有UAV共同容量")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.num_uavs is not None:
            config["mission"]["num_uavs"] = args.num_uavs
        for key, value in (("root", args.data_root), ("output", args.output)):
            if value is not None:
                config["data"][key] = str(Path(value).expanduser().resolve())
        if args.samples_per_uav is not None:
            config["data"]["samples_per_uav"] = args.samples_per_uav
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
