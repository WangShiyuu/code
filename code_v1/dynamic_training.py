"""作用：连接动态切分、块资源求解与真实 SFL 梯度更新，保存可追溯的实验结果。

运行指令（通过统一入口，不单独执行本模块）：
    python -X utf8 -B train_sfl.py --prepare-only
    python -X utf8 -B train_sfl.py --prepared-run output/output_时间戳_git哈希 --save-checkpoint
    python -X utf8 -B train_sfl.py --epochs 4 --device cpu

准备阶段只画像和预演成本，不更新网络。执行阶段先求解并核验块，再计算各机
梯度，所有梯度有效后统一提交 SGD。物理时钟只累计正文的计算、通信和聚合时延；
求解器、本机读图及评价的墙钟耗时另记。失败不补造聚合，不保存未聚合的局部模型
冒充全局结果。每次执行保存的 checkpoint 是最近一次完整聚合状态。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from config import load_config
from scenario_preparation import prepare_dynamic, _write, _hash, _portable_config
from execution import DynamicExecution
from data import AcquisitionSnapshots, CBCLDataset, validate_manifest
from dynamic_controller import (DynamicController, SelectionInfeasible, allocate_epoch_budgets,
                                make_aggregation_problem, make_training_problem)
from long_environment import open_environment
from resource_solver import compact_decision, evaluate_block, solve_block
from sfl_model import equal_fedavg, make_common_models


def _cpu_state(model):
    """独立复制完整聚合权重，防止后续局部更新污染最后安全检查点。"""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class _IndexedSubset(Dataset):
    """返回图像、标签和训练池局部索引；采样顺序仍由原 DataLoader 决定。"""

    def __init__(self, dataset, indices):
        """冻结训练池局部索引副本，保留清单中不可变的采集到达顺序。"""
        self.dataset, self.indices = dataset, tuple(indices)

    def __len__(self):
        """返回轮初冻结快照的样本数，新增样本不会改变当前轮长度。"""
        return len(self.indices)

    def __getitem__(self, index):
        """读取一张已到达图像，返回图像、类别和原始训练池局部索引。"""
        actual = self.indices[index]
        image, label = self.dataset[actual]
        return image, label, actual


def run_dynamic(args):
    """运行正式主算法；求解、梯度或预算失败时返回可审计的部分结果。"""
    from train_sfl import evaluate, split_step, cost_snapshot, epoch_costs
    wall_start = time.perf_counter()
    config = load_config(args.config)
    optimization_source = getattr(args, "optimization_results", None)
    if optimization_source:
        from decision_replay import locate_optimization
        if args.prepare_only:
            raise ValueError("读取优化结果执行训练时不能同时 --prepare-only")
        inferred = locate_optimization(optimization_source).parent.parent
        if args.prepared_run and Path(args.prepared_run).resolve() != inferred:
            raise ValueError("--prepared-run 与 --optimization-results 不属于同一运行包")
        args.prepared_run = inferred
    if args.split:
        raise ValueError("动态模式不接受 --split；固定切点请显式使用 --mode diagnostic")
    if args.scenario:
        raise ValueError("动态模式生成独立长环境；已有准备包请使用 --prepared-run")
    output = Path(args.prepared_run).resolve() if args.prepared_run else prepare_dynamic(args, config)
    prepared = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if _portable_config(config) != _portable_config(prepared["config"]):
        raise ValueError("当前控制/物理/训练配置与准备包不一致，请用相同 --config 或重新准备")
    epochs = prepared["epochs"]
    if args.epochs is not None and args.epochs != epochs:
        raise ValueError("--epochs 与准备包不一致")
    for file, key in (("environment/scenario.json", "scenario_sha256"), ("calibration.json", "calibration_sha256"),
                       ("split_profiles.json", "profiles_sha256")):
        if _hash(output / file) != prepared[key]:
            raise ValueError(f"准备文件已改变：{file}")
    if args.prepare_only:
        return dict(prepared_only=True, outputs=dict(run_dir=str(output)), epochs=epochs)
    if (output / "execution_started.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("此准备包已经启动过训练；重新准备可创建另一份独立实验")
    manifest_path = (args.manifest or Path(config["data"]["output"]) / "data.json").resolve()
    if _hash(manifest_path) != prepared["manifest_sha256"]:
        raise ValueError("数据清单与准备包不一致")
    init = config["training"]["initialization_checkpoint"]
    if (_hash(init) if init else None) != prepared["initialization_sha256"]:
        raise ValueError("初始化权重内容与准备包不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    trace = open_environment(output / "environment")
    snapshots = AcquisitionSnapshots(manifest, trace.collection)
    datasets = [CBCLDataset(manifest_path.parent, [row for row in manifest["samples"] if row["split"] == split],
                            config["data"]["image_size"]) for split in ("train", "val")]
    train, validation = datasets
    profiles = json.loads((output / "split_profiles.json").read_text(encoding="utf-8"))
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    # 显式读取优化结果时不导入或调用优化求解流程；其他情况下保留在线连接。
    replay = None
    if optimization_source:
        from decision_replay import DecisionReplay
        replay = DecisionReplay(optimization_source, output, _hash)
    torch.set_num_threads(args.threads)
    device_name = args.device or config["training"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前 PyTorch 或设备不支持 CUDA")
    device = torch.device(device_name)
    seed, n = config["training"]["initialization_seed"], config["mission"]["num_uavs"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    models = make_common_models(n, len(manifest["classes"]), seed=seed, checkpoint=init)
    last_global = _cpu_state(models[0])
    initial = {name: p.detach().cpu().clone() for name, p in models[0].named_parameters()}
    cuts = [config["selection"]["initial_split"]] * n
    for model, cut in zip(models, cuts):
        model.move_split(cut, device, device)
    optimizers = [torch.optim.SGD(model.parameters(), lr=config["training"]["epoch_step_size"])
                  for model in models]
    rng = random.Random(config["data"]["seed"])
    eval_indices = rng.sample(range(len(validation)), min(args.eval_samples, len(validation)))
    evaluation_before = evaluate(models[0], validation, eval_indices, config["training"]["eval_batch_size"],
                                 config["training"]["workers"], device)
    physical, controller = DynamicExecution(config, trace), DynamicController(config, profiles, trace)
    period, batch_size = config["training"]["aggregation_period"], config["training"]["batch_size"]
    epoch_records, aggregations, period_records = [], [], []
    resource_ranges, solver_stats = {}, dict(blocks=0, wall_s=0.0, max_original_feasibility=0.0,
                                            max_kkt_residual=0.0, max_direction_residual=0.0)
    wall_stages = {key: 0.0 for key in ("wall_client_forward_s", "wall_server_forward_backward_s",
                                       "wall_client_backward_and_update_s", "wall_sgd_commit_s")}
    visits, seen = [0] * n, [set() for _ in range(n)]
    warm, failure, completed_epochs, last_global_epoch = None, None, 0, 0
    pending_entry = None
    block_log = (output / "block_decisions.jsonl").open("w", encoding="utf-8")
    _write(output / "execution_started.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat()))

    def select_next(counts, current, start_epoch, *, incoming=None):
        """已知样本量下筛选下一周期，所有预算来自准备阶段的固定列表。"""
        budgets = dict(epoch_budgets_s=calibration["epoch_budgets_s"][start_epoch - 1:start_epoch - 1 + period],
                       incoming_aggregation=incoming, final_period=start_epoch + period - 1 == epochs)
        if replay:
            chosen, record = replay.select(start_epoch, physical.time_s, counts, current)
        else:
            chosen, record = controller.select(physical.time_s, counts, current,
                                                first_period=start_epoch == 1, budgets=budgets)
        record["start_epoch"] = start_epoch
        period_records.append(record)
        _write(output / "period_splits.json", period_records)
        return chosen

    def solve(problem, epoch, block, extra=None):
        """先求解与核验再进入模型计算；日志同时保留失败块，不将其算为完成。"""
        nonlocal warm, pending_entry
        result = (replay.solve(problem, epoch, block, physical.time_s, list(cuts)) if replay
                  else solve_block(problem, config, warm_start=warm))
        diagnostic = result.get("diagnostics", {})
        entry = dict(epoch=epoch, block=block, kind=problem.kind, start_s=physical.time_s,
                     budget_s=problem.budget_s, channel_gains=problem.channel_gains,
                     up_bits=problem.up_bits, down_bits=problem.down_bits,
                     client_cycles=problem.client_cycles, server_cycles=problem.server_cycles,
                     cuts=list(cuts), committed=False, decision_source="replay" if replay else "online",
                     decision=compact_decision(result))
        if extra:
            entry.update(extra)
        if not replay:
            solver_stats["wall_s"] += diagnostic.get("wall_s", 0.0)
        if result.get("status") != "success":
            block_log.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"); block_log.flush()
            raise RuntimeError(f"资源求解失败：{result.get('reason', result.get('status'))}")
        # 审计失败也必须保留该次已求出的资源、预算及失败原因。
        pending_entry = entry
        # 再次重算原问题，不能仅依赖求解器 status 或凸子问题 gap。
        audit = evaluate_block(problem, config, result["resources"])
        duration = audit["duration_s"]
        if (abs(duration - result["duration_s"]) > 1e-6 * max(1, duration)
                or abs(audit["energy_j"] - result["energy_j"]) > 1e-6 * max(1, audit["energy_j"])):
            raise RuntimeError("求解决策与原物理成本重算不一致")
        if duration > problem.budget_s + config["optimization"]["feasibility_tolerance"] * max(1, problem.budget_s):
            raise RuntimeError("原问题块时延超过预算")
        if physical.time_s + duration > min(calibration["deadline_s"], trace.duration_s) + 1e-6:
            raise RuntimeError("当前块将超出总期限或预生成环境")
        for key in ("original_feasibility", "kkt_residual"):
            value = diagnostic.get(key)
            limit = config["optimization"]["feasibility_tolerance" if key == "original_feasibility" else "kkt_tolerance"]
            if value is None or value > limit:
                raise RuntimeError(f"原问题审计未通过：{key}={value}")
        direction = diagnostic.get("direction_residual", diagnostic.get("direction_norm"))
        if direction is None or direction > config["optimization"]["direction_tolerance"]:
            raise RuntimeError(f"方向残差审计未通过：{direction}")
        warm = result["resources"]
        return result, entry

    def committed(problem, result, entry):
        """仅对已提交块更新物理台账、实际资源区间及求解统计。"""
        physical.commit(problem, result)
        entry.update(committed=True, end_s=physical.time_s)
        block_log.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"); block_log.flush()
        solver_stats["blocks"] += 1
        diag = result["diagnostics"]
        for field in ("original_feasibility", "kkt_residual"):
            solver_stats["max_" + field] = max(solver_stats["max_" + field], diag[field])
        solver_stats["max_direction_residual"] = max(solver_stats["max_direction_residual"],
                                                    diag.get("direction_residual", diag.get("direction_norm", 0)))
        for key, values in result["resources"].items():
            if not isinstance(values, list):
                continue
            ranges = resource_ranges.setdefault(key, [dict(min=None, max=None) for _ in range(n)])
            for i, value in enumerate(values):
                if value is None or value <= 0:
                    continue
                ranges[i]["min"] = value if ranges[i]["min"] is None else min(value, ranges[i]["min"])
                ranges[i]["max"] = value if ranges[i]["max"] is None else max(value, ranges[i]["max"])

    try:
        # 首周期直接读取初始CSI；只有实际块开始才向20次观测窗口加入一条。
        cuts = select_next([trace.sample_count_at(0.0)] * n, cuts, 1)
        for model, cut in zip(models, cuts):
            model.move_split(cut, device, device)
        for epoch in range(1, epochs + 1):
            start_s, before = physical.time_s, cost_snapshot(physical)
            current = snapshots.snapshot_indices(start_s)
            counts = [len(indices) for indices in current]
            gains = trace.channel_at(start_s)
            allocation = allocate_epoch_budgets(config, profiles, cuts, counts, gains,
                        calibration["epoch_budgets_s"][epoch - 1], epoch % period == 0, final=epoch == epochs)
            loaders = []
            for uav, indices in enumerate(current):
                # 显式生成排列，既保持原采集顺序，又能准确统计中途失败前已提交的样本。
                generator = torch.Generator().manual_seed(seed + epoch * n + uav)
                # 与旧 DataLoader shuffle=True 保持同一调用及更新规则；下列索引
                # 由采样器本身提供，不另用不同随机排列改变训练数据次序。
                loader = DataLoader(_IndexedSubset(train, indices), batch_size=batch_size, shuffle=True,
                                    generator=generator, num_workers=config["training"]["workers"], drop_last=False)
                # 将索引随样本一起返回，统计以实际读取结果为准。
                loaders.append(iter(loader))
            losses, correct, used = [0.0] * n, [0] * n, [0] * n
            record = dict(epoch=epoch, start_s=start_s, samples_per_uav=counts[0], cuts=list(cuts),
                          budget_s=calibration["epoch_budgets_s"][epoch - 1], completed=False,
                          committed_batches=0, batches_per_uav=len(allocation["batch_budgets_s"]),
                          tail_batch_samples=counts[0] % batch_size or batch_size)
            epoch_records.append(record)
            training_wall = time.perf_counter()
            for batch, budget in enumerate(allocation["batch_budgets_s"]):
                block_counts = [min(batch_size, count - batch * batch_size) for count in counts]
                gains = controller.observe(physical.time_s)
                problem = make_training_problem(config, profiles, cuts, block_counts, gains, budget)
                result, entry = solve(problem, epoch, batch + 1)
                pending_entry = entry
                outputs, index_batches = [], []
                for uav, loader in enumerate(loaders):
                    images, labels, actual_indices = next(loader)
                    step = config["training"]["epoch_step_size"] * labels.numel() / counts[uav]
                    out = split_step(models[uav], optimizers[uav], images, labels, cuts[uav], step, device, commit=False)
                    for key in out.keys() & wall_stages.keys():
                        wall_stages[key] += out[key]
                    profile = profiles[cuts[uav]]
                    for field, profile_field in (("activation_bytes", "activation_bytes_per_sample"),
                                                  ("gradient_bytes", "gradient_bytes_per_sample"),
                                                  ("label_bytes", "label_bytes_per_sample")):
                        if out[field] != profile[profile_field] * out["samples"]:
                            raise RuntimeError("实际张量载荷与本块画像不一致")
                    outputs.append(out); index_batches.append(actual_indices.tolist())
                # 各机均已获得有效梯度；至此之前所有模型参数保持本批更新前状态。
                update_start = time.perf_counter()
                for optimizer in optimizers:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                wall_stages["wall_sgd_commit_s"] += time.perf_counter() - update_start
                committed(problem, result, entry)
                pending_entry = None
                for uav, out in enumerate(outputs):
                    visits[uav] += out["samples"]; used[uav] += out["samples"]
                    seen[uav].update(index_batches[uav])
                    losses[uav] += out["mean_loss"] * out["samples"]
                    correct[uav] += out["correct"]
                record["committed_batches"] += 1
                if batch == 0 or (batch + 1) % 10 == 0 or batch + 1 == len(allocation["batch_budgets_s"]):
                    print(f"epoch {epoch}/{epochs} batch {batch+1}/{len(allocation['batch_budgets_s'])}; "
                          f"cuts={cuts}; D={counts[0]}; sim={physical.time_s:.3f}s", flush=True)
            completed_epochs = epoch
            record.update(training_end_s=physical.time_s, mean_losses=[loss / d for loss, d in zip(losses, counts)],
                          accuracies=[v / d for v, d in zip(correct, counts)], completed=True,
                          wall_training_s=time.perf_counter() - training_wall)
            if epoch % period == 0:
                old_cuts = list(cuts)
                aggregation_start = physical.time_s
                gains = controller.observe(physical.time_s)
                new_cuts = list(cuts)
                if epoch < epochs:
                    incoming = dict(budget_s=allocation["aggregation_budget_s"], old_cuts=old_cuts, gains=gains)
                    new_cuts = select_next([trace.sample_count_at(physical.time_s)] * n, cuts, epoch + 1, incoming=incoming)
                problem = make_aggregation_problem(config, profiles, old_cuts, new_cuts, gains,
                                                    allocation["aggregation_budget_s"], final=epoch == epochs)
                result, entry = solve(problem, epoch, "aggregation", dict(old_cuts=old_cuts, new_cuts=new_cuts))
                pending_entry = entry
                # 单机中完整模型一直存在；旧前缀上传和新前缀下发的真实载荷
                # 由 problem 分别给出，改变切点仅移动原 Parameter，不重新初始化。
                equal_fedavg(models)
                for model, cut in zip(models, new_cuts):
                    model.move_split(cut, device, device)
                committed(problem, result, entry)
                pending_entry = None
                cuts = new_cuts
                last_global, last_global_epoch = _cpu_state(models[0]), epoch
                if args.save_checkpoint:
                    torch.save(dict(model_state_dict=last_global, epochs=last_global_epoch,
                                    cuts=cuts, data_fingerprint=manifest["fingerprint"]), output / "checkpoint.pt")
                metrics = evaluate(models[0], validation, eval_indices, config["training"]["eval_batch_size"],
                                   config["training"]["workers"], device)
                aggregations.append(dict(after_epoch=epoch, start_s=aggregation_start, end_s=physical.time_s,
                                         old_cuts=old_cuts, new_cuts=new_cuts, final=epoch == epochs, validation=metrics))
            record["end_with_aggregation_s"] = physical.time_s
            record["costs"] = epoch_costs(before, physical)
        if replay:
            replay.finish()
    except (RuntimeError, ValueError, FloatingPointError, OSError, SelectionInfeasible) as error:
        failure = dict(type=type(error).__name__, reason=str(error), time_s=physical.time_s,
                       completed_epochs=completed_epochs, last_aggregated_epoch=last_global_epoch)
        if pending_entry is not None:
            pending_entry["failure"] = failure
            block_log.write(json.dumps(pending_entry, ensure_ascii=False, allow_nan=False) + "\n")
        if hasattr(error, "record"):
            period_records.append(error.record)
        if epoch_records and "costs" not in epoch_records[-1]:
            epoch_records[-1]["costs"] = epoch_costs(before, physical)
            epoch_records[-1]["end_with_aggregation_s"] = physical.time_s
        print(f"Stopped: {failure['reason']}", flush=True)
    finally:
        block_log.close()
        if replay:
            replay.close()
    _write(output / "period_splits.json", period_records)
    # 失败时也保留最近聚合（首次聚合前则为统一初值），不会保存局部模型当最终结果。
    if args.save_checkpoint:
        torch.save(dict(model_state_dict=last_global, epochs=last_global_epoch,
                        data_fingerprint=manifest["fingerprint"], status="failed" if failure else "completed"),
                   output / "checkpoint.pt")
    delta = math.sqrt(math.fsum((last_global[name].double() - p.double()).square().sum().item()
                               for name, p in initial.items()))
    from run_output import source_revision
    return dict(schema_version=2, kind="dynamic_uav_sfl", source_git=source_revision(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        outputs=dict(run_dir=str(output), environment_dir=str(output / "environment"),
                     summary_json=str(output / "summary.json"), report_md=str(output / "report.md"),
                     key_metrics_csv=str(output / "key_metrics.csv"),
                     checkpoint=str(output / "checkpoint.pt") if args.save_checkpoint else None,
                     block_decisions=str(output / "block_decisions.jsonl"), period_splits=str(output / "period_splits.json")),
        config=config, run=dict(epochs=completed_epochs, requested_epochs=epochs, status="failed" if failure else "completed",
            aggregation_period=period, aggregations=len(aggregations), last_aggregated_epoch=last_global_epoch,
            device=str(device), torch_version=torch.__version__, threads=args.threads,
            task_epoch_budget_enforced=True, energy_constraints_enforced=False,
            resource_policy="周期切分+块级CVXPY/Clarabel近端子问题及原目标Armijo",
            synchronization="块起始CSI冻结、批末同步，飞行及采集持续推进",
            endpoint="最近完成聚合的完整模型保留服务器；末次不下发"),
        inputs=dict(manifest=str(manifest_path), manifest_sha256=prepared["manifest_sha256"],
                    data_fingerprint=manifest["fingerprint"], scenario=str(output / "environment"), collection=trace.collection),
        preparation=prepared, calibration=calibration, split_profiles=profiles, period_records=period_records,
        epoch_records=epoch_records, aggregation_records=aggregations, resource_ranges=resource_ranges,
        solver_statistics=solver_stats, failure=failure, validation_before=evaluation_before,
        optimization_input=dict(mode="replay" if replay else "online", path=str(replay.path) if replay else None,
                                offline_solver_wall_s=replay.summary["wall_s"] if replay else None),
        validation_after=aggregations[-1]["validation"] if aggregations else evaluation_before,
        validation_subset_sha256=hashlib.sha256(json.dumps(eval_indices).encode()).hexdigest(), parameter_change_l2=delta,
        data_participation=dict(unique_trained_samples_per_uav=[len(v) for v in seen],
            sample_visits_per_uav=visits, available_samples_per_uav_at_end=trace.sample_count_at(physical.time_s),
            minimum_passes_required=None), physical_execution=physical.report(),
        wall_timing=dict(solver_s=solver_stats["wall_s"], total_before_report_write_s=time.perf_counter() - wall_start,
                         preparation_s=prepared["preparation_wall_s"],
                         neural_compute_and_sgd_s=math.fsum(wall_stages.values()),
                         summed_branch_compute_stages_s=wall_stages),
        interpretation=["能耗按正文物理模型估计；求解和本机计算墙钟均不计入物理时延",
                        "100 epoch 是任务设置，不代表已用分类准确率证明理论梯度收敛",
                        "失败仅计已提交批次，检查点只保留最近完成聚合的模型"])
