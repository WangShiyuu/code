"""用途：处理 CBCL StreetScenes 九类区域分类数据，并给不同无人机分配训练样本。
运行：python data.py --uavs 3 --partition iid --samples-per-uav 100 200 300
运行：python data.py --uavs 3 --partition noniid --alpha 0.3 --train-samples 600
完整参数：python data.py --help

原始目录需要 Images.zip 和 Annotations.zip；程序不下载或修改原数据。
一个标注 polygon 的外接矩形就是一个分类样本，不是整张街景的单标签分类。
先按原始街景划分训练/验证/测试，避免同一原图的区域进入不同评价集合。
这是本项目的可复现划分，不声称为 CBCL 官方划分。训练区域可分给不同无人机，
每个区域只分配一次；这样才能在不复制样本的前提下精确设置各无人机样本数。
默认输出：本文件旁 output/data/data.json 和 crops/，所有训练方式共用这一清单。
依赖：numpy、Pillow；只有训练读取张量时才需要 torch。适用于 Python 3.10+。
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image


CLASSES = ("bicycle", "building", "car", "pedestrian", "road", "sidewalk", "sky", "store", "tree")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "data"


def default_data_root():
    """从代码目录向上寻找现有 CBCL，移动整个项目后仍可自动定位。"""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "DataSet" / "CBCL StreetScenes"
        if (candidate / "Images.zip").is_file() and (candidate / "Annotations.zip").is_file():
            return str(candidate)
    return str(Path(__file__).resolve().parent / "datasets" / "CBCL StreetScenes")


def write_json(path, value):
    """写可读 UTF-8 JSON；拒绝 NaN/Infinity，避免结果无法被其他程序读取。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _basename(name):
    # ZIP/XML 内的名字可使用不同操作系统的分隔符，统一后只取文件名。
    return PurePosixPath(name.replace("\\", "/")).name


def read_cbcl(data_root):
    """读取 ZIP/XML，保留类别、原图、标注序号和裁剪框，不解压整个 2 GB 图像包。"""
    root = Path(data_root).expanduser().resolve()
    for name in ("Images.zip", "Annotations.zip"):
        if not (root / name).is_file():
            raise ValueError(f"CBCL 缺少 {root / name}，请设置 --data-root")
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
                # CBCL MATLAB 标注从 1 开始；PIL 从 0 开始且右/下边界不包含在裁剪中。
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


def split_scenes(records, seed, val_fraction, test_fraction):
    """只按原图划分 holdout；改变 UAV 数量/容量/非 IID 参数不会改变评价场景。"""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("验证和测试比例必须大于 0，且总和小于 1")
    scenes = sorted({row["scene_id"] for row in records})
    if len(scenes) < 3:
        raise ValueError("至少需要 3 个有效场景才能隔离训练、验证和测试")
    scenes = np.random.default_rng(seed).permutation(scenes).tolist()
    n_val, n_test = max(1, round(len(scenes) * val_fraction)), max(1, round(len(scenes) * test_fraction))
    if n_val + n_test >= len(scenes):
        raise ValueError("当前比例使训练场景为空；请减小验证/测试比例")
    mapping = {scene: "val" for scene in scenes[:n_val]}
    mapping.update({scene: "test" for scene in scenes[n_val:n_val + n_test]})
    return {subset: [row for row in records if mapping.get(row["scene_id"], "train") == subset]
            for subset in ("train", "val", "test")}


def resolve_sizes(available, uavs, train_samples=None, samples_per_uav=None):
    """解析精确容量：一个值广播给所有 UAV，多个值按 UAV 编号依次对应。"""
    if uavs < 1:
        raise ValueError("--uavs 必须至少为 1")
    if samples_per_uav is not None:
        sizes = list(samples_per_uav)
        if len(sizes) == 1:
            sizes *= uavs
        if len(sizes) != uavs or any(size < 1 for size in sizes):
            raise ValueError("--samples-per-uav 需给 1 个正整数，或与 UAV 数量一致的正整数")
        if train_samples is not None and train_samples != sum(sizes):
            raise ValueError("--train-samples 必须等于 --samples-per-uav 的总和，或省略")
    else:
        total = available if train_samples is None else train_samples
        if total < uavs:
            raise ValueError("训练样本总数必须至少等于 UAV 数量，保证每架 UAV 非空")
        # 除法余数给编号较小的 UAV，因此默认容量最多相差 1 个样本。
        sizes = [total // uavs + int(i < total % uavs) for i in range(uavs)]
    if sum(sizes) > available:
        raise ValueError(f"请求 {sum(sizes)} 个训练样本，但训练场景只有 {available} 个；不允许重复采样")
    return sizes


def partition_uavs(labels, sizes, partition, alpha, seed):
    """返回各 UAV 的训练集局部索引，容量精确、互不重复、完整覆盖选中样本。"""
    if partition not in ("iid", "noniid"):
        raise ValueError("--partition 只能是 iid 或 noniid")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("--alpha 必须为有限正数")
    if sum(sizes) != len(labels) or min(sizes) < 1:
        raise ValueError("UAV 容量必须非空且与选中的训练样本数一致")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    clients = [[] for _ in sizes]
    if partition == "iid":
        # IID：随机打散样本，再按容量切片；是随机 IID 抽样，有限数据不保证类别比例完全相同。
        start = 0
        for i, size in enumerate(sizes):
            clients[i] = order[start:start + size].tolist()
            start += size
    else:
        # 每一类独立生成 Dirichlet(alpha) UAV 偏好；alpha 越小，类别分布通常越不均衡。
        # 有容量约束时，已满 UAV 的概率置零；乘剩余容量以兼容不等大小的 UAV。
        # 因此这是容量受限的 Dirichlet 划分，实际分布以 data.json 中计数为准。
        preferences = rng.dirichlet(np.full(len(sizes), alpha), size=len(CLASSES))
        remaining = np.array(sizes, dtype=np.int64)
        for index in order:
            probabilities = preferences[int(labels[index])] * remaining
            if probabilities.sum() <= 0:  # 极小 alpha 可数值下溢，仍按剩余容量可靠完成分配。
                probabilities = remaining.astype(np.float64)
            client = int(rng.choice(len(sizes), p=probabilities / probabilities.sum()))
            clients[client].append(int(index))
            remaining[client] -= 1
        for indices in clients:
            rng.shuffle(indices)
    return clients


def _fingerprint(manifest):
    """绑定预处理参数、样本及 UAV 映射，训练日志可用这个值核对同一份数据。"""
    value = {key: manifest[key] for key in ("schema", "settings", "classes", "samples", "clients")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_manifest(manifest):
    """检查跨集合原图泄漏、样本重复和 UAV 索引；错误清单不能进入训练。"""
    if manifest.get("schema") != 1 or manifest.get("classes") != list(CLASSES):
        raise ValueError("仅支持本代码生成的 CBCL 九分类 data.json")
    if manifest.get("fingerprint") != _fingerprint(manifest):
        raise ValueError("data.json 指纹不一致，文件可能被修改；请重新准备到新的输出目录")
    scenes, sample_ids, counts = {}, set(), Counter()
    for row in manifest["samples"]:
        if row["sample_id"] in sample_ids or row["split"] not in ("train", "val", "test"):
            raise ValueError("清单存在重复样本或非法集合")
        if not 0 <= row["label"] < len(CLASSES) or CLASSES[row["label"]] != row["class_name"]:
            raise ValueError("类别和标签不一致")
        sample_ids.add(row["sample_id"])
        if scenes.setdefault(row["scene_id"], row["split"]) != row["split"]:
            raise ValueError("同一原始街景出现在不同集合，存在评价泄漏")
        crop = Path(row["crop"])
        if crop.is_absolute() or crop.drive or ".." in crop.parts or not crop.parts or crop.parts[0] != "crops":
            raise ValueError("缓存图片路径必须是 crops 内的相对路径")
        counts[row["split"]] += 1
    clients = manifest["clients"]
    indices = [i for client in clients for i in client["indices"]]
    if sorted(indices) != list(range(counts["train"])):
        raise ValueError("UAV 索引必须恰好覆盖训练集一次，不能重复或遗漏")
    if len(clients) != manifest["settings"]["uavs"] or any(not c["indices"] for c in clients):
        raise ValueError("UAV 数量不一致或存在空 UAV")
    if any(counts[name] == 0 for name in ("train", "val", "test")):
        raise ValueError("训练、验证和测试集合均需要非空")


def prepare_data(args):
    """根据 argparse 参数生成 PNG 缓存和清单；参数相同的完成缓存直接复用。"""
    root = Path(args.data_root).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    if args.image_size < 32:
        raise ValueError("--image-size 至少为 32")
    sizes_option = getattr(args, "samples_per_uav", None)
    settings = dict(source_root=str(root), image_size=args.image_size, uavs=args.uavs,
                    partition=args.partition, alpha=args.alpha, seed=args.seed,
                    train_samples=args.train_samples, samples_per_uav=sizes_option,
                    val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    for name in ("Images.zip", "Annotations.zip"):
        source = root / name
        if not source.is_file():
            raise ValueError(f"缺少 {source}，请设置 --data-root")
    # 原始包发生变化时不悄悄复用旧缓存；这里只读文件元信息，不反复计算 2 GB 哈希。
    settings["source_files"] = {name: dict(size=(root / name).stat().st_size,
                                            mtime_ns=(root / name).stat().st_mtime_ns)
                                for name in ("Images.zip", "Annotations.zip")}
    manifest_path = destination / "data.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest["settings"] != settings:
            raise ValueError("此输出目录已有其他参数的数据，请用 --output 指定新目录")
        if any(not (destination / row["crop"]).is_file() for row in manifest["samples"]):
            raise ValueError("缓存图片缺失，请使用新的输出目录重新准备")
        return manifest_path
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("输出目录非空且没有完成清单，请选择新的 --output")
    records, report = read_cbcl(root)
    subsets = split_scenes(records, args.seed, args.val_fraction, args.test_fraction)
    sizes = resolve_sizes(len(subsets["train"]), args.uavs, args.train_samples, sizes_option)
    # 容量不足会先报错。抽样不放回，未选中的训练区域不混入验证/测试。
    chosen = np.random.default_rng(args.seed + 1).permutation(len(subsets["train"]))[:sum(sizes)]
    subsets["train"] = [subsets["train"][int(i)] for i in chosen]
    client_indices = partition_uavs([row["label"] for row in subsets["train"]], sizes,
                                    args.partition, args.alpha, args.seed + 2)
    samples = [{**row, "split": split} for split in ("train", "val", "test") for row in subsets[split]]
    grouped = defaultdict(list)
    (destination / "crops").mkdir(parents=True)
    for index, row in enumerate(samples):
        row["crop"] = f"crops/{index:07d}.png"
        grouped[row["member"]].append(row)
    # 一张街景可能有很多标注区域，统一解码一次再裁剪，减少重复读取高分辨率 JPEG。
    with zipfile.ZipFile(root / "Images.zip") as archive:
        for index, (member, rows) in enumerate(sorted(grouped.items())):
            with archive.open(member) as stream, Image.open(stream) as source:
                rgb = source.convert("RGB")
                for row in rows:
                    rgb.crop(tuple(row["bbox"])).resize((args.image_size, args.image_size),
                                                       Image.Resampling.BILINEAR).save(destination / row["crop"])
                rgb.close()
            if (index + 1) % 100 == 0:
                print(f"已处理 {index + 1}/{len(grouped)} 个街景", flush=True)
    clients = [dict(uav_id=i, indices=indices, samples=len(indices),
                    class_counts={name: sum(subsets["train"][j]["label"] == label for j in indices)
                                  for label, name in enumerate(CLASSES)})
               for i, indices in enumerate(client_indices)]
    manifest = dict(schema=1, settings=settings, classes=list(CLASSES), mean=list(MEAN), std=list(STD),
                    samples=samples, clients=clients, dataset_report=report,
                    split_protocol="custom scene-group holdout; training crops may belong to different UAVs",
                    subset_counts={split: len(rows) for split, rows in subsets.items()},
                    scene_counts={split: len({r["scene_id"] for r in rows}) for split, rows in subsets.items()},
                    class_counts={split: dict(Counter(row["class_name"] for row in rows)) for split, rows in subsets.items()})
    manifest["fingerprint"] = _fingerprint(manifest)
    validate_manifest(manifest)
    # 最后才发布清单，防止训练读取尚未完成的图片缓存。
    write_json(destination / "data.pending.json", manifest)
    (destination / "data.pending.json").replace(manifest_path)
    return manifest_path


class CBCLDataset:
    """PyTorch 可读取的数据集：PNG -> RGB/CHW 张量 -> 固定均值/标准差归一化。"""
    def __init__(self, root, rows, image_size):
        self.root, self.rows, self.image_size = Path(root), rows, image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch  # 数据划分本身不需要 GPU，也不要求安装 PyTorch。
        row = self.rows[int(index)]
        with Image.open(self.root / row["crop"]) as image:
            rgb = image.convert("RGB")
            if rgb.size != (self.image_size, self.image_size):
                rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.array(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean, std = tensor.new_tensor(MEAN)[:, None, None], tensor.new_tensor(STD)[:, None, None]
        return (tensor - mean) / std, torch.tensor(row["label"], dtype=torch.long)


def load_data(manifest_path, image_size=None):
    """训练统一入口：返回 train、val、test 数据集、UAV 局部索引列表、完整元数据。"""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path /= "data.json"
    if not manifest_path.is_file():
        # GitHub 只保存代码，预处理缓存需在训练主机本地生成。
        # 不自动选择容量和 non-IID 参数，避免把默认划分误用为用户的实验配置。
        raise ValueError(
            f"未找到 CBCL 数据划分清单：{manifest_path}\n"
            "首次训练请先在 code 目录准备数据，例如：\n"
            f'python data.py --data-root "{default_data_root()}" '
            f'--uavs 3 --partition iid --output "{manifest_path.parent}"\n'
            "请按实验需要设置 UAV 数量、样本量和 IID/non-IID 参数。\n"
            "原始数据目录应包含 Images.zip 和 Annotations.zip；位置不同时修改 --data-root。\n"
            f'准备完成后，训练参数 --data 指向 "{manifest_path.parent / "data.json"}"。\n'
            "若已准备数据，直接用 --data 指向现有 data.json。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    image_size = manifest["settings"]["image_size"] if image_size is None else image_size
    if image_size < 32:
        raise ValueError("image_size 至少为 32")
    missing = next((row["crop"] for row in manifest["samples"] if not (manifest_path.parent / row["crop"]).is_file()), None)
    if missing:
        raise ValueError(f"缓存图像缺失：{manifest_path.parent / missing}")
    datasets = [CBCLDataset(manifest_path.parent, [r for r in manifest["samples"] if r["split"] == name], image_size)
                for name in ("train", "val", "test")]
    return (*datasets, [client["indices"] for client in manifest["clients"]], manifest)


def build_parser():
    parser = argparse.ArgumentParser(description="CBCL 处理与 UAV 数据划分；先按原图隔离评价集合，再精确分配训练区域")
    parser.add_argument("--data-root", default=default_data_root(), help="包含 Images.zip、Annotations.zip 的 CBCL 目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="PNG 缓存与 data.json 的输出目录")
    parser.add_argument("--uavs", type=int, default=3, help="无人机数量，默认 3")
    parser.add_argument("--partition", choices=("iid", "noniid"), default="iid", help="训练样本分配方式")
    parser.add_argument("--alpha", type=float, default=0.5, help="non-iid Dirichlet 浓度，越小通常越不均衡")
    parser.add_argument("--train-samples", type=int, default=None, help="选中训练样本总数；省略则用全部训练区域")
    parser.add_argument("--samples-per-uav", type=int, nargs="+", default=None, help="各 UAV 精确样本数，例如 100 200 300；单值广播")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="验证街景比例，默认 0.15")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="测试街景比例，默认 0.15")
    parser.add_argument("--image-size", type=int, default=64, help="缓存区域边长，默认 64；至少 32")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        path = prepare_data(args)
        info = json.loads(path.read_text(encoding="utf-8"))
        print(f"数据清单：{path}\n集合样本数：{info['subset_counts']}\nUAV 样本数：{[c['samples'] for c in info['clients']]}")
    except (ValueError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        parser.exit(2, f"数据准备失败：{exc}\n")