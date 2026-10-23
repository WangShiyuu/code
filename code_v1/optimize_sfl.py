"""作用：独立执行周期切分和块资源优化，输出可供训练直接读取的决策结果。

运行指令（在 code_v1 目录）：
    python -X utf8 -B train_sfl.py --prepare-only
    python -X utf8 -B optimize_sfl.py --prepared-run output/output_时间戳_git哈希
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
import time

from config import load_config
from dynamic_controller import (DynamicController, SelectionInfeasible, allocate_epoch_budgets,
                                make_aggregation_problem, make_training_problem)
from scenario_preparation import _hash, _portable_config, _write
from execution import DynamicExecution
from long_environment import open_environment
from resource_solver import compact_decision, solve_block


def optimize_prepared(directory, config):
    """求解完整已准备任务并流式写入结果；失败立即停止，不保存伪可行后续块。"""
    output = Path(directory).expanduser().resolve()
    prepared = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if _portable_config(config) != _portable_config(prepared["config"]):
        raise ValueError("优化配置与准备包不一致")
    for file, key in (("environment/scenario.json", "scenario_sha256"), ("calibration.json", "calibration_sha256"),
                       ("split_profiles.json", "profiles_sha256")):
        if _hash(output / file) != prepared[key]:
            raise ValueError(f"准备文件已改变：{file}")
    destination = output / "optimization"
    destination.mkdir(exist_ok=False)
    trace = open_environment(output / "environment")
    profiles = json.loads((output / "split_profiles.json").read_text(encoding="utf-8"))
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    controller = DynamicController(config, profiles, trace)
    physical = DynamicExecution(config, trace)
    epochs, period = prepared["epochs"], config["training"]["aggregation_period"]
    n = config["mission"]["num_uavs"]
    cuts, warm, periods, completed, block_count = [config["selection"]["initial_split"]] * n, None, [], 0, 0
    started, failure = time.perf_counter(), None
    decisions = destination / "block_decisions.jsonl"

    def select(start_epoch, incoming=None):
        """周期选择只使用当前已到达样本，并保留独立的候选扫描记录。"""
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
            result = solve_block(problem, config, warm_start=warm)
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
            failure = dict(type=type(error).__name__, reason=str(error), time_s=physical.time_s)
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
    return result


def main(argv=None):
    """只接受已有准备包；训练与独立优化使用完全相同的物理及控制配置。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="与准备阶段相同的局部JSON配置")
    args = parser.parse_args(argv)
    result = optimize_prepared(args.prepared_run, load_config(args.config))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
