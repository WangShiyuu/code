"""UAV-SFL 实验入口。config 查看参数，prepare 准备数据，profile 生成模型画像，
simulate 运行资源仿真，train 进行联合训练，check 校验研究协议，smoke 运行四轮合成训练。
"""
from __future__ import annotations

import argparse
import copy
import csv
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np

# 直接运行 main.py 时，将 code_v3 的父目录加入路径，与包方式使用相同的模块名。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_v3.config import (
    ROOT, json_value, load_config, save_json, smoke_config, training_budget,
    uav_parameter, validate_config,
)
from code_v3.optimization import (
    channel_at, client_cycle_table, grouped_profile, reference_resources,
    select_epoch_splits, solve_epoch_candidates,
)
from code_v3.training import (
    aggregate_models, epoch_snapshot, evaluate_model, initialize_model, prepare_data,
    profile_splits, save_training_results, train_epoch_branches,
)
from code_v3.verification import verify_implementation


def read_source_files() -> dict[str, bytes]:
    """读取组成当前实验的模块，供源码摘要与归档共用。"""
    names = ("config.py", "main.py", "optimization.py", "training.py", "verification.py")
    return {name: (ROOT / name).read_bytes() for name in names}


def code_version(sources: dict[str, bytes] | None = None) -> dict:
    """记录当前提交与实际源码哈希，Git 不可用时仍可追溯源码。"""
    sources = read_source_files() if sources is None else sources
    file_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    manifest = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_sha = hashlib.sha256(manifest).hexdigest()
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
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": source_sha, "source_files_sha256": file_hashes}


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
def experiment_output(config: dict, label: str, output: str | Path | None = None):
    """为各入口保存参数、版本和运行状态，独占结果目录。"""
    sources = read_source_files()
    version = code_version(sources)
    directory = Path(output) if output else make_output_directory(config, label, version)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "config.json").exists():
        raise FileExistsError(f"实验目录已被使用: {directory}")
    now = lambda: datetime.now(timezone(timedelta(hours=8))).isoformat()
    metadata = {"mode": label, "status": "running", "started_at": now(), "finished_at": None,
                "output_dir": str(directory.resolve()), "code_version": version}
    # 排他创建运行记录，防止两个显式 --output 同时写入同一目录。
    with (directory / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    started = time.perf_counter()
    try:
        save_json(directory / "config.json", config)
        with zipfile.ZipFile(directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for name, content in sources.items():
                snapshot.writestr(f"code_v3/{name}", content)
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
    with experiment_output(config, label or ("train" if train else "simulate"), output) as (directory, metadata):
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


def build_argument_parser():
    """构造七个子命令共用的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["config", "prepare", "profile", "simulate", "train", "check", "smoke"])
    parser.add_argument("--config", help="JSON 局部参数覆盖文件")
    parser.add_argument("--output", help="本次运行的新目录；不得覆盖已有 config.json")
    parser.add_argument("--checkpoint", help="ResNet18 本地预训练参数路径")
    parser.add_argument("--allow-random-init", action="store_true", help="显式允许随机初始化，仅用于诊断/指定的消融")
    return parser


def load_cli_config(args: argparse.Namespace):
    """读取 JSON 和显式命令行覆盖，保持原参数优先级。"""
    updates = {}
    if args.checkpoint or args.allow_random_init:
        updates["training"] = {}
        if args.checkpoint:
            updates["training"]["checkpoint"] = str(Path(args.checkpoint).resolve())
        if args.allow_random_init:
            updates["training"]["allow_random_init"] = True
    return load_config(args.config, updates)


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
    with experiment_output(config, args.command, args.output) as (output, metadata):
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
    args = build_argument_parser().parse_args()
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
