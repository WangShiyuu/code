"""作用：独立执行周期切分和块资源优化，输出可供训练直接读取的决策结果。

运行指令（在 code_v1 目录）：
    python -X utf8 -B train_sfl.py --prepare-only
    python -X utf8 -B optimize_sfl.py --prepared-run output/output_时间戳_git哈希
    python -X utf8 -B optimize_sfl.py --prepared-run output/output_时间戳_git哈希 --retry-failed
    python -X utf8 -B train_sfl.py --optimization-results output/output_时间戳_git哈希

本入口只执行优化及成本时序预演，不加载训练图像、不执行网络梯度或 SGD。
模型画像已经在准备阶段取得，后续样本量由同一物理时钟和固定到达序列确定。
因此可以预先求解整段决策，再让训练按相同时序重放。每次决策仍仅使用该时刻
可见的样本量与当前/历史 CSI，不在选择或求解时读取正式环境未来值。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from config import load_config
from dynamic_controller import (DynamicController, SelectionInfeasible, allocate_epoch_budgets,
                                make_aggregation_problem, make_training_problem)
from scenario_preparation import _hash, _portable_config, _write
from execution import DynamicExecution
from long_environment import open_environment
from resource_solver import compact_decision, solve_block


def _optimization_directory(output, retry_failed):
    """创建优化目录，或在明确要求重试时完整归档原失败结果。

    重试只重新计算优化决策，不重建长环境，也不改 preparation.json，因而原准备包
    的文件指纹继续有效。成功结果及已经启动训练的准备包禁止重试。归档使用同一
    运行目录中的原子重命名，不删除文件；拒绝符号链接、目录联接和越界路径。
    """
    destination = output / "optimization"
    if destination.is_symlink() or getattr(destination, "is_junction", lambda: False)():
        raise ValueError("优化目录不能是符号链接或目录联接")
    if destination.resolve().parent != output:
        raise ValueError("优化目录必须位于当前运行目录内")
    if not retry_failed:
        if destination.exists():
            raise FileExistsError("优化目录已经存在；若上次优化失败且未启动训练，请增加 --retry-failed")
        destination.mkdir()
        return destination
    if (output / "execution_started.json").exists() or (output / "summary.json").exists():
        raise ValueError("此准备包已启动训练，禁止重试优化；请另建准备包")
    summary_path = destination / "optimization.json"
    if not summary_path.is_file():
        raise ValueError("--retry-failed 仅用于已有 optimization.json 的失败优化目录")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    if previous.get("status") != "failed" or previous.get("kind") != "dynamic_uav_sfl_optimization":
        raise ValueError("--retry-failed 仅允许重试 status=failed 的独立优化，不能覆盖成功结果")
    from run_output import source_revision
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    name = f"optimization_failed_{stamp}_{source_revision()['short_hash']}"
    for suffix in range(10000):
        archive = output / (name + (f"_{suffix:02d}" if suffix else ""))
        # 归档前检查最终绝对路径；不把未经检查的计算路径交给移动操作。
        if archive.resolve().parent != output:
            raise ValueError("失败优化归档路径超出当前运行目录")
        if archive.exists():
            continue
        destination.rename(archive)
        print(f"Previous failed optimization archived: {archive}", flush=True)
        destination.mkdir()
        return destination
    raise FileExistsError("无法创建唯一的失败优化归档目录")


def _print_failure(failure):
    """打印直接定位失败所需的信息；详细逐次诊断仍保存在原结果文件中。"""
    print(f"Reason: {failure.get('reason', '未记录')}", flush=True)
    print(f"Failure location: epoch={failure.get('epoch', '未记录')}; "
          f"block={failure.get('block', '未记录')}; "
          f"elapsed={failure.get('time_s', 0.0):.6f}s; "
          f"completed_epochs={failure.get('completed_epochs', 0)}; "
          f"committed_blocks={failure.get('committed_blocks', 0)}", flush=True)
    diagnostics = failure.get("diagnostics") or {}
    fields = ("iterations", "original_feasibility", "kkt_residual", "direction_residual",
              "direction_norm", "recovery_slack", "proven_infeasible",
              "minimum_time_lower_bound_s", "block_budget_s", "last_solver_error")
    values = [f"{key}={diagnostics[key]}" for key in fields if diagnostics.get(key) is not None]
    print("Solver diagnostics: " + ("; ".join(values) if values else "本失败点尚无求解残差，见失败原因"), flush=True)


def optimize_prepared(directory, config, *, retry_failed=False):
    """求解完整准备任务；可显式归档失败优化后重试，不改变物理或预算参数。

    保留 optimize_prepared(directory, config) 的既有调用接口。仅完整成功的优化
    可以交给训练；失败块不推进模拟时钟，日志指出失败轮次、块和求解残差。
    """
    output = Path(directory).expanduser().resolve()
    prepared = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if _portable_config(config) != _portable_config(prepared["config"]):
        raise ValueError("优化配置与准备包不一致")
    for file, key in (("environment/scenario.json", "scenario_sha256"), ("calibration.json", "calibration_sha256"),
                       ("split_profiles.json", "profiles_sha256")):
        if _hash(output / file) != prepared[key]:
            raise ValueError(f"准备文件已改变：{file}")
    destination = _optimization_directory(output, retry_failed)
    trace = open_environment(output / "environment")
    profiles = json.loads((output / "split_profiles.json").read_text(encoding="utf-8"))
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    controller = DynamicController(config, profiles, trace)
    physical = DynamicExecution(config, trace)
    epochs, period = prepared["epochs"], config["training"]["aggregation_period"]
    n = config["mission"]["num_uavs"]
    cuts, warm, periods, completed, block_count = [config["selection"]["initial_split"]] * n, None, [], 0, 0
    started, failure = time.perf_counter(), None
    # context 始终指向当前正在处理的操作；失败时可以区分候选筛选和实际块求解。
    context = dict(epoch=1, block="period_selection", diagnostics={})
    decisions = destination / "block_decisions.jsonl"

    def select(start_epoch, incoming=None):
        """周期选择只使用当前已到达样本，并保留独立的候选扫描记录。"""
        context.update(epoch=start_epoch, block="period_selection", diagnostics={})
        chosen, record = controller.select(physical.time_s, [trace.sample_count_at(physical.time_s)] * n, cuts,
             first_period=start_epoch == 1,
             budgets=dict(epoch_budgets_s=calibration["epoch_budgets_s"][start_epoch - 1:start_epoch - 1 + period],
                          incoming_aggregation=incoming, final_period=start_epoch + period - 1 == epochs))
        record["start_epoch"] = start_epoch
        periods.append(record)
        return chosen

    with decisions.open("w", encoding="utf-8") as stream:
        def solve(problem, epoch, block, new_cuts=None):
            """输出单块优化结果，任何失败不推进优化模拟时钟。"""
            nonlocal warm, block_count
            context.update(epoch=epoch, block=block, diagnostics={})
            result = solve_block(problem, config, warm_start=warm)
            context["diagnostics"] = result.get("diagnostics", {})
            entry = dict(epoch=epoch, block=block, kind=problem.kind, start_s=physical.time_s,
                         budget_s=problem.budget_s, channel_gains=problem.channel_gains,
                         client_cycles=problem.client_cycles, server_cycles=problem.server_cycles,
                         up_bits=problem.up_bits, down_bits=problem.down_bits, cuts=list(cuts),
                         new_cuts=new_cuts, decision=compact_decision(result), committed=False)
            if result["status"] == "success":
                if physical.time_s + result["duration_s"] > calibration["deadline_s"] + 1e-6:
                    raise RuntimeError("优化块超出总期限")
                physical.commit(problem, result)
                entry.update(committed=True, end_s=physical.time_s)
                warm, block_count = result["resources"], block_count + 1
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
            if result["status"] != "success":
                raise RuntimeError(result["reason"])

        try:
            cuts = select(1)
            for epoch in range(1, epochs + 1):
                context.update(epoch=epoch, block="budget_allocation", diagnostics={})
                counts = [trace.sample_count_at(physical.time_s)] * n
                allocation = allocate_epoch_budgets(config, profiles, cuts, counts, trace.channel_at(physical.time_s),
                    calibration["epoch_budgets_s"][epoch - 1], epoch % period == 0, final=epoch == epochs)
                for block, (samples, budget) in enumerate(zip(allocation["batch_sample_counts"],
                                                               allocation["batch_budgets_s"]), 1):
                    problem = make_training_problem(config, profiles, cuts, samples,
                                                    controller.observe(physical.time_s), budget)
                    solve(problem, epoch, block)
                if epoch % period == 0:
                    gains = controller.observe(physical.time_s)
                    new_cuts = list(cuts) if epoch == epochs else select(epoch + 1, dict(
                        old_cuts=list(cuts), gains=gains, budget_s=allocation["aggregation_budget_s"]))
                    problem = make_aggregation_problem(config, profiles, cuts, new_cuts, gains,
                                                        allocation["aggregation_budget_s"], final=epoch == epochs)
                    solve(problem, epoch, "aggregation", new_cuts)
                    cuts = new_cuts
                completed = epoch
                print(f"optimization epoch {epoch}/{epochs}; blocks={block_count}; sim={physical.time_s:.3f}s", flush=True)
        except (RuntimeError, ValueError, SelectionInfeasible) as error:
            failure = dict(type=type(error).__name__, reason=str(error), time_s=physical.time_s,
                           epoch=context["epoch"], block=context["block"], completed_epochs=completed,
                           committed_blocks=block_count, diagnostics=context["diagnostics"])
            if hasattr(error, "record"):
                periods.append(error.record)
    _write(destination / "period_splits.json", periods)
    from run_output import source_revision
    result = dict(schema_version=1, kind="dynamic_uav_sfl_optimization", status="failed" if failure else "completed",
                  epochs=epochs, completed_epochs=completed, blocks=block_count, failure=failure,
                  source_git=source_revision(), created_at_utc=datetime.now(timezone.utc).isoformat(),
                  preparation_sha256=_hash(output / "preparation.json"),
                  decisions_sha256=_hash(decisions), periods_sha256=_hash(destination / "period_splits.json"),
                  decisions="block_decisions.jsonl", periods="period_splits.json", network_updates=0,
                  physical_execution=physical.report(), wall_s=time.perf_counter() - started)
    _write(destination / "optimization.json", result)
    print(f"Optimization {result['status']}: {destination / 'optimization.json'}", flush=True)
    if failure:
        _print_failure(failure)
    return result


def main(argv=None):
    """只接受已有准备包；训练与独立优化使用完全相同的物理及控制配置。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="与准备阶段相同的局部JSON配置")
    parser.add_argument("--retry-failed", action="store_true",
                        help="仅在优化失败且未启动训练时，保留旧失败目录并复用准备包重新优化")
    args = parser.parse_args(argv)
    try:
        result = optimize_prepared(args.prepared_run, load_config(args.config), retry_failed=args.retry_failed)
    except (ValueError, OSError) as error:
        # 输入、指纹及重试条件错误是可解释的用户级失败，不输出难读的 Python traceback。
        print(f"Optimization could not start: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
