"""四种基线共用的 CUDA 训练、资源审计和结果记录。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np

from code_v3.config import SPLIT_POINTS, load_config, resolve_training_device, uav_parameter
from code_v3.optimization import (
    audit_resources, channel_at, evaluate_resources, grouped_profile, reference_resources,
)
from code_v3.training import (
    aggregate_models, checkpoint_payload, epoch_snapshot, evaluate_model, initialize_model,
    prepare_data, profile_splits, train_epoch_branches,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
BEIJING = timezone(timedelta(hours=8))
RESOURCE_KEYS = ("frequency_hz", "power_w", "up_hz", "down_hz")
UAV_FIELDS = ("split", "next_split", "channel_gain", *RESOURCE_KEYS, "energy_j", "remaining_j")
EPOCH_FIELDS = (
    "epoch", "status", "decision", "attempts", "aggregate", "samples_per_uav", "start_s", "end_s",
    "duration_s", "system_energy_j", "server_energy_j", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy", "val_macro_f1",
)


@dataclass(frozen=True)
class EpochState:
    epoch: int
    count: int
    current: np.ndarray
    gain: np.ndarray
    gain_history: tuple[np.ndarray, ...]
    remaining: np.ndarray
    reference: dict
    profile: dict
    config: dict
    rng: np.random.Generator

    @property
    def choosing_next(self) -> bool:
        return self.epoch % 2 == 0 and self.epoch < self.config["training"]["epochs"]


@dataclass(frozen=True)
class EpochDecision:
    problem: dict | None
    resources: dict | None
    status: str
    attempts: int = 1


def source_hash(baseline: str) -> str:
    """对入口、实验循环和实际依赖的 v3 模块计算源码哈希。"""
    paths = [ROOT / f"{baseline}.py", ROOT / "experiment.py"]
    paths.extend(ROOT.parent / "code_v3" / name for name in ("config.py", "optimization.py", "training.py"))
    digest = hashlib.sha256()
    for path in paths:
        name = path.relative_to(ROOT.parent).as_posix()
        digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def create_output_directory(baseline: str, digest: str) -> Path:
    while True:
        stamp = datetime.now(BEIJING).strftime("%Y%m%d_%H%M%S_%f")
        directory = OUTPUT_ROOT / f"output_{baseline}_{stamp}_{digest[:12]}"
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            continue


def report_items(mapping: dict, prefix: str = ""):
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from report_items(value, f"{name}.")
        else:
            yield name, value


def load_cli_config(args: argparse.Namespace) -> dict:
    updates = {"training": {"device": args.device}}
    if args.checkpoint is not None:
        updates["training"]["checkpoint"] = str(args.checkpoint.expanduser().resolve())
    if args.epochs is not None:
        updates["training"]["epochs"] = args.epochs
        updates["budget"] = {"mode": "empirical"}
    if args.data_root is not None:
        updates["data"] = {"root": str(args.data_root.expanduser().resolve())}
    if args.seed is not None:
        updates["seed"] = args.seed
    if getattr(args, "split", None) is not None:
        updates["optimization"] = {"fixed_split": args.split}
    config = load_config(args.config, updates)
    config["output_root"] = str(OUTPUT_ROOT)
    if not config["training"]["checkpoint"]:
        config["training"]["checkpoint"] = str(ROOT.parent / "code_v2" / "resnet18-f37072fd.pth")

    if args.device != "cuda" and not args.device.startswith("cuda:"):
        raise ValueError("基线训练要求使用 CUDA，请指定 cuda 或 cuda:N。")
    try:
        config["training"]["device"] = resolve_training_device(args.device)
    except RuntimeError as exc:
        raise RuntimeError(f"无法启用 {args.device}，请检查 NVIDIA 驱动及 PyTorch CUDA 版本。") from exc
    return config


def save_models(config, data, output, report, last_model, best_model, completed):
    import torch

    for name, model in (("last_aggregated", last_model), ("best_validation", best_model)):
        if completed:
            metrics = evaluate_model(model, data, data["test"], config)
            report[f"{name}_test"] = {key: metrics[key] for key in ("loss", "accuracy", "macro_f1", "samples")}
        if config["training"]["save_checkpoint"]:
            torch.save(checkpoint_payload(model), output / f"{name}_model.pt")


def execute_training(config, choose_epoch, output, report):
    import torch

    n, epochs = config["num_uavs"], config["training"]["epochs"]
    torch.set_num_threads(config["training"]["torch_threads"])
    data = prepare_data(config)
    base, provenance = initialize_model(config, require_pretrained=True)
    profile = grouped_profile(profile_splits(base, config))
    report.update(dataset_kind=data["kind"], data_fingerprint=data["fingerprint"], initialization=provenance)
    current = np.full(n, config["optimization"]["fixed_split"], dtype=int)
    remaining = uav_parameter(config, "energy_cap_j").copy()
    reference = reference_resources(config)
    # 策略随机数独立于数据划分、模型初始化和训练批次。
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 4]))
    physical_s = data["start_s"]
    gains = []
    branches = [base] * n
    last_model, best_model = None, None
    report.update(start_s=physical_s, finish_s=physical_s, duration_s=0.0, epochs_completed=0,
                  aggregations_completed=0, last_aggregation_epoch=0, processed_samples=0,
                  distinct_training_samples=0, system_energy_j=config["task"]["fixed_energy_j"],
                  server_energy_j=0.0, uav_energy_j=np.zeros(n), remaining_j=remaining.copy(), best_val_accuracy=None)

    fields = [*EPOCH_FIELDS, *(f"uav{uid}_{key}" for uid in range(1, n + 1) for key in UAV_FIELDS)]
    with (output / "history.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        history = csv.DictWriter(stream, fieldnames=fields)
        history.writeheader()
        for epoch in range(1, epochs + 1):
            snapshot = epoch_snapshot(data, physical_s, epoch, config)
            count = snapshot["count"]
            gain = channel_at(physical_s, config)["gain"]
            state = EpochState(epoch, count, current, gain, tuple(gains), remaining, reference, profile, config, rng)
            decision = choose_epoch(state)
            problem, resources = decision.problem, decision.resources
            row = dict(epoch=epoch, decision=decision.status, attempts=decision.attempts,
                       samples_per_uav=count, start_s=physical_s)

            if problem is not None:
                row["aggregate"] = int(problem["aggregate"])
            per_uav = {"channel_gain": gain}
            if problem is not None:
                per_uav.update(split=problem["split"], next_split=problem["next_split"])
            if resources is not None:
                per_uav.update(resources)
            for key, values in per_uav.items():
                if values is not None:
                    row.update({f"uav{uid}_{key}": value for uid, value in enumerate(values, 1)})

            audit = audit_resources(problem, resources, profile, config) if problem is not None and resources is not None else None
            if audit is None or not audit["ok"]:
                report.update(status="incomplete", failure_epoch=epoch,
                              failure=decision.status if audit is None else "resource_audit_failed", failure_audit=audit)
                history.writerow({**row, "status": "infeasible"})
                break

            metrics = evaluate_resources(problem, resources, profile, config)
            branches, stats = train_epoch_branches(branches, data, snapshot["indices"], problem["split"],
                                                   epoch, config, metrics["loads"])
            if problem["aggregate"]:
                last_model = aggregate_models(branches, [count] * n)
                branches = [last_model] * n
                validation = evaluate_model(last_model, data, data["val"], config)
                row.update(val_loss=validation["loss"], val_accuracy=validation["accuracy"],
                           val_macro_f1=validation["macro_f1"])
                if best_model is None or validation["accuracy"] > report["best_val_accuracy"]:
                    best_model = last_model
                    report.update(best_val_accuracy=validation["accuracy"], best_val_epoch=epoch)
                report["aggregations_completed"] += 1
                report["last_aggregation_epoch"] = epoch

            current = problem["next_split"] if problem["next_split"] is not None else problem["split"]
            reference = resources
            gains.append(gain)
            physical_s += metrics["duration_s"]
            remaining = remaining - metrics["uav_energy_j"]
            row.update(status="completed", end_s=physical_s, duration_s=metrics["duration_s"],
                       system_energy_j=metrics["total_energy_j"], server_energy_j=metrics["server_energy_j"],
                       train_loss=np.mean([stat["loss"] for stat in stats]),
                       train_accuracy=np.mean([stat["accuracy"] for stat in stats]))
            for uid, (energy, battery) in enumerate(zip(metrics["uav_energy_j"], remaining), 1):
                row.update({f"uav{uid}_energy_j": energy, f"uav{uid}_remaining_j": battery})
            history.writerow(row)
            stream.flush()

            report.update(epochs_completed=epoch, finish_s=physical_s, remaining_j=remaining,
                          duration_s=physical_s - report["start_s"], distinct_training_samples=n * count)
            report["processed_samples"] += n * count
            report["system_energy_j"] += metrics["total_energy_j"]
            report["server_energy_j"] += metrics["server_energy_j"]
            report["uav_energy_j"] += metrics["uav_energy_j"]
            print(f"epoch {epoch}/{epochs} D={count} split={problem['split'].tolist()} "
                  f"T={metrics['duration_s']:.3f}s E={metrics['total_energy_j']:.3f}J", flush=True)

    completed = report["epochs_completed"] == epochs
    if last_model is not None:
        save_models(config, data, output, report, last_model, best_model, completed)
    if completed:
        report["status"] = "completed"


def run_baseline(baseline: str, choose_epoch: Callable[[EpochState], EpochDecision], description: str,
                 *, fixed_split: bool = False) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="可选的 v3 配置覆盖文件")
    parser.add_argument("--checkpoint", type=Path, help="ResNet18 预训练权重，默认使用仓库中的权重")
    parser.add_argument("--data-root", type=Path, help="存放 Images.zip 与 Annotations.zip 的目录")
    parser.add_argument("--device", default="cuda", help="CUDA 设备，如 cuda:0")
    parser.add_argument("--epochs", type=int, help="覆盖经验训练轮数，必须为正偶数")
    parser.add_argument("--seed", type=int, help="覆盖数据划分、初始化与随机策略种子")
    if fixed_split:
        parser.add_argument("--split", type=int, choices=SPLIT_POINTS, help="固定切分点，默认 4")
    args = parser.parse_args()

    try:
        config = load_cli_config(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{baseline}: {exc}\n")

    import torch

    device = config["training"]["device"]
    gpu = torch.cuda.get_device_name(device)
    digest = source_hash(baseline)
    output = create_output_directory(baseline, digest)
    started = perf_counter()
    report = dict(baseline=baseline, status="running", started_at=datetime.now(BEIJING).isoformat(),
                  source_sha256=digest, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu=gpu, epochs_planned=config["training"]["epochs"], epochs_completed=0, config=config)
    print(f"{baseline} | {device} ({gpu}) | {output}", flush=True)
    try:
        execute_training(config, choose_epoch, output, report)
    except KeyboardInterrupt:
        report.update(status="interrupted", failure="KeyboardInterrupt")
        return 130
    except Exception as exc:
        report.update(status="error", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(finished_at=datetime.now(BEIJING).isoformat(), wall_s=perf_counter() - started)
        with (output / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("field", "value"))
            writer.writerows(report_items(report))

    if report["status"] == "incomplete":
        print(f"epoch {report['failure_epoch']}: {report['failure']}", file=sys.stderr)
        return 2
    return 0
