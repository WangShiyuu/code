"""作用：画像、独立校准成本预演和完整长环境准备，与训练梯度更新分开。
运行指令：python -X utf8 -B train_sfl.py --prepare-only
准备只调用模型副本取得FLOP与载荷画像，不执行SGD；缓存和原600秒场景保持原样。
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from config import resolve_collection
from data import validate_manifest
from dynamic_controller import calibrate
from long_environment import ReferenceChannel, generate_long_environment
from sfl_model import make_common_models, profile_splits

def _write(path, value):
    """完整覆盖单份 JSON；不允许 NaN/Infinity 混入正式结果。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")

def _hash(path):
    """按块计算文件指纹，避免将长环境读进内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _portable_config(config):
    """比较实验参数时剔除机器相关目录；数据本身另由 SHA256 绑定。"""
    result = copy.deepcopy(config)
    for group, key in (("data", "root"), ("data", "output"), ("output", "scenario_dir"),
                       ("training", "split_profile_path")):
        result[group].pop(key, None)
    # 初始化文件由内容校验，不以两台电脑上的绝对路径是否相同判断。
    result["training"].pop("initialization_checkpoint", None)
    result["training"].pop("device", None)
    result["training"].pop("workers", None)
    return result

def prepare_dynamic(args, config):
    """独立信道预演、固定逐轮预算、生成完整正式长环境及准备清单。"""
    from run_output import create_run_directory, source_revision
    start = time.perf_counter()
    epochs = args.epochs or config["training"]["total_epochs"]
    period = config["training"]["aggregation_period"]
    if epochs % period:
        raise ValueError("epoch 数必须为聚合间隔的整数倍")
    manifest_path = (args.manifest or Path(config["data"]["output"]) / "data.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    collection = resolve_collection(config, manifest)
    output = create_run_directory(args.output)
    print(f"Output: {output}", flush=True)
    torch.set_num_threads(args.threads)
    model = make_common_models(1, len(manifest["classes"]),
                              seed=config["training"]["initialization_seed"],
                              checkpoint=config["training"]["initialization_checkpoint"])[0]
    profiles = profile_splits(model, config["selection"]["candidates"], config["data"]["image_size"],
                              config["training"]["batch_size"])
    seed = config["optimization"]["calibration_seed"]
    seed = config["channel"]["seed"] + 1 if seed is None else seed
    print(f"Reference preflight: {epochs} epochs, calibration seed={seed}, no SGD", flush=True)
    calibration = calibrate(config, profiles, collection, ReferenceChannel(config, seed=seed), epochs)
    calibration["calibration_seed"] = seed
    calibration["formal_seed"] = config["channel"]["seed"]
    calibration["network_updates"] = 0
    _write(output / "calibration.json", calibration)
    _write(output / "split_profiles.json", profiles)
    # duration 只延长环境覆盖；collection 中 lambda 始终来自原采集计划。
    deadline = calibration["deadline_s"]
    dt = config["mission"]["trace_interval_s"]
    horizon = math.ceil(deadline / dt) * dt
    print(f"Preparing formal environment to {horizon:.3f}s (deadline={deadline:.3f}s)...", flush=True)
    scenario = generate_long_environment(config, manifest_path, output / "environment", horizon,
                                          collection=collection,
                                          max_part_bytes=config["optimization"]["max_part_bytes"])
    init = config["training"]["initialization_checkpoint"]
    prepared = dict(schema_version=2, status="prepared", config=config, epochs=epochs, source_git=source_revision(),
                    manifest_sha256=_hash(manifest_path), data_fingerprint=manifest["fingerprint"],
                    manifest_path=str(manifest_path), environment="environment", calibration="calibration.json",
                    profiles="split_profiles.json", scenario_sha256=_hash(scenario),
                    calibration_sha256=_hash(output / "calibration.json"),
                    profiles_sha256=_hash(output / "split_profiles.json"),
                    initialization_sha256=_hash(init) if init else None,
                    preparation_wall_s=time.perf_counter() - start,
                    created_at_utc=datetime.now(timezone.utc).isoformat())
    _write(output / "preparation.json", prepared)
    print(f"Prepared: {output}; collection rate={collection['rate_samples_per_s']:g}/s", flush=True)
    return output
