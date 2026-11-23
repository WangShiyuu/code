"""四种基线的命令行入口，复用 code_v2 的实验流程和归档协议。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable
import zipfile

import numpy as np

from code_v2 import uav_sfl


ROOT = Path(__file__).resolve().parent
REFERENCE_CONFIG = ROOT / "results1122_config.json"


@dataclass(frozen=True)
class EpochState:
    epoch: int
    count: int
    current: np.ndarray
    gain: np.ndarray
    gain_history: list[np.ndarray]
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
    solve: dict | None = None
    selection: dict | None = None


def load_cli_config(args: argparse.Namespace, baseline: str) -> dict:
    config = uav_sfl.load_config(REFERENCE_CONFIG)
    if args.config:
        path = args.config.expanduser().resolve()
        overrides = uav_sfl.migrate_config_updates(json.loads(path.read_text(encoding="utf-8-sig")))
        # 归档可作为配置重用，实际策略由本次入口决定。
        overrides.pop("baseline", None)
        for group, key in (("data", "root"), ("training", "checkpoint")):
            if overrides.get(group, {}).get(key):
                overrides[group][key] = str((path.parent / Path(overrides[group][key]).expanduser()).resolve())
        config = uav_sfl.merge_config(config, overrides)
    # 归档中的输出路径属于原机器，使用 --output 才改变本次写入位置。
    config["output_root"] = str(ROOT / "output")

    for arg, group, key in (
        (args.data_root, "data", "root"),
        (args.checkpoint, "training", "checkpoint"),
    ):
        if arg is not None:
            config[group][key] = str(arg.expanduser().resolve())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["budget"]["mode"] = "empirical"
    if getattr(args, "split", None) is not None:
        config["optimization"]["fixed_split"] = args.split
    config["optimization"]["strategy"] = {
        "fixed_split": "fixed_opt",
        "fixed_split_resources": "fixed_fixed",
    }.get(baseline, "dynamic_opt")
    config["baseline"] = baseline
    if args.device is not None:
        config["training"]["device"] = args.device
    elif args.mode == "simulate":
        config["training"]["device"] = "cpu"
    config = uav_sfl.validate_config(config)
    if args.mode == "train":
        device = config["training"]["device"]
        if device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("基线训练要求使用 CUDA，请指定 cuda 或 cuda:N。")
    return config


def make_epoch_policy(choose_epoch: Callable[[EpochState], EpochDecision], seed: int):
    # 策略采样与训练采样使用独立的随机数流。
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))

    def policy(epoch, count, current, gain, gain_history, remaining, reference, profile, config):
        state = EpochState(epoch, count, current, gain, gain_history, remaining, reference, profile, config, rng)
        decision = choose_epoch(state)
        problem, resources = decision.problem, decision.resources
        if decision.solve is not None:
            solved = decision.solve
        else:
            audit = None
            metrics = None
            if problem is not None and resources is not None:
                audit = uav_sfl.audit_resources(problem, resources, profile, config)
                if audit["ok"]:
                    metrics = uav_sfl.evaluate_resources(problem, resources, profile, config)
            ok = audit is not None and audit["ok"]
            status = decision.status if ok or audit is None else f"{decision.status}_infeasible"
            solved = {"ok": ok, "status": status, "resources": resources,
                      "metrics": metrics, "audit": audit, "trace": [], "wall_s": 0.0}
        attempts = []
        if problem is not None:
            attempts.append({"split": problem["split"], "next_split": problem["next_split"], "solve": solved})
        return problem, solved, decision.selection, attempts

    return policy


def run_baseline(baseline: str, choose_epoch: Callable[[EpochState], EpochDecision], description: str,
                 *, fixed_split: bool = False) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("mode", nargs="?", choices=("train", "simulate"), default="train",
                        help="train 执行 CUDA 训练，simulate 计算模型时延和能耗")
    parser.add_argument("--config", type=Path, help="覆盖 results1122 默认配置的 JSON")
    parser.add_argument("--checkpoint", type=Path, help="ResNet18 预训练权重，默认使用仓库中的权重")
    parser.add_argument("--data-root", type=Path, help="存放 Images.zip 与 Annotations.zip 的目录")
    parser.add_argument("--device", help="模型设备，训练默认 cuda:0，成本仿真默认 cpu")
    parser.add_argument("--epochs", type=int, help="覆盖经验训练轮数，必须为正偶数")
    parser.add_argument("--seed", type=int, help="覆盖数据划分、初始化与随机策略种子")
    parser.add_argument("--output", type=Path, help="为本次运行指定独立输出目录")
    if fixed_split:
        parser.add_argument("--split", type=int, choices=uav_sfl.SPLIT_POINTS, help="固定切分点，默认 4")
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()

    try:
        config = load_cli_config(args, baseline)
        requested_device = config["training"]["device"]
        config["training"]["device"] = uav_sfl.resolve_training_device(requested_device)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{baseline}: {exc}\n")

    label = f"{args.mode}_{baseline}"
    if fixed_split:
        label += f"_p{config['optimization']['fixed_split']}"
    train = args.mode == "train"
    try:
        if args.output is not None and args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"实验目录非空: {args.output}")
        source_files = {
            path.relative_to(ROOT.parent).as_posix(): path.read_bytes()
            for path in (ROOT / f"{baseline}.py", ROOT / "experiment.py", REFERENCE_CONFIG)
        }
        with uav_sfl.experiment_output(
            config, label, args.output, training_device_requested=requested_device, source_files=source_files,
        ) as (output, metadata):
            metadata.update(mode=args.mode, baseline=baseline)
            if train:
                import torch

                metadata.update(gpu=torch.cuda.get_device_name(config["training"]["device"]),
                                cuda_version=torch.version.cuda)
            with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
                snapshot.writestr("code_v2/uav_sfl.py", (output / "uav_sfl_snapshot.py").read_bytes())
                for name, content in source_files.items():
                    snapshot.writestr(name, content)

            print(f"{baseline} | {config['training']['device']} | {output}", flush=True)
            (output / "decisions.jsonl").touch(exist_ok=False)
            policy = make_epoch_policy(choose_epoch, config["seed"])
            result = uav_sfl.execute_experiment(
                config, train, None, None, output, metadata["code_version"], epoch_policy=policy,
            )
            summary = result["summary"]
            metadata.update(status=summary["status"], epochs_completed=summary["epochs_completed"],
                            epochs_planned=summary["epochs_planned"], failure=summary["failure"])
    except KeyboardInterrupt:
        return 130
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        print(f"{baseline}: {exc}", file=sys.stderr)
        return 1

    report = {key: summary[key] for key in ("status", "strategy", "epochs_completed", "epochs_planned",
                                          "system_energy_j", "processed_samples")}
    if summary["failure"] is not None:
        report.update(failure_epoch=summary["failure"]["epoch"], failure_reason=summary["failure"]["reason"])
    print(json.dumps({"output_dir": str(output.resolve()), "summary": report},
                     default=uav_sfl.json_value, ensure_ascii=False, indent=2))
    return 0 if summary["completed"] else 2
