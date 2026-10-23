"""作用：流式读取独立优化结果，绑定准备包，向训练提供已求解的资源。

运行指令：python -X utf8 -B train_sfl.py --optimization-results output/output_时间戳_git哈希_git哈希
本模块不调用 CVXPY，不重新选择切分；它核对任务、时刻、CSI、负载和预算，
确保训练调用的是当前物理块对应的结果，不能跨场景或跳过失败块继续执行。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path


def locate_optimization(value):
    """接受运行目录、optimization目录或具体JSON文件。"""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    return path / "optimization.json" if (path / "optimization.json").exists() else path / "optimization" / "optimization.json"


class DecisionReplay:
    """只保留当前块，长任务不一次加载所有资源与迭代诊断。"""

    def __init__(self, source, run_directory, digest):
        """绑定完整成功的优化包，校验准备包和两份决策文件的SHA256后打开流；不读取图片。"""
        path = locate_optimization(source)
        self.summary = json.loads(path.read_text(encoding="utf-8"))
        self.path = path
        if (self.summary.get("status") != "completed"
                or self.summary.get("kind") != "dynamic_uav_sfl_optimization"):
            raise ValueError("独立优化没有完整成功，不能作为正式训练输入")
        if self.summary["preparation_sha256"] != digest(Path(run_directory) / "preparation.json"):
            raise ValueError("优化结果与当前准备包不一致")
        blocks, periods = path.parent / self.summary["decisions"], path.parent / self.summary["periods"]
        if digest(blocks) != self.summary["decisions_sha256"] or digest(periods) != self.summary["periods_sha256"]:
            raise ValueError("独立优化结果指纹不一致")
        self.periods = {record["start_epoch"]: record for record in json.loads(periods.read_text(encoding="utf-8"))}
        self.stream, self.read_blocks = blocks.open(encoding="utf-8"), 0

    def select(self, epoch, time_s, counts, cuts):
        """核对周期起点后返回原选择结果；训练不会再次执行候选扫描。"""
        record = copy.deepcopy(self.periods[epoch])
        if (abs(record["time_s"] - time_s) > 1e-6 or record["sample_counts"] != counts
                or record["current_cuts"] != cuts):
            raise ValueError("训练的周期起点与独立优化不一致")
        return list(record["selected_cuts"]), record

    def solve(self, problem, epoch, block, time_s, cuts):
        """匹配当前块并返回资源，物理成本仍由训练端独立重算。"""
        line = self.stream.readline()
        if not line:
            raise ValueError("独立优化结果在训练完成前已读尽")
        entry = json.loads(line)
        if (entry["epoch"] != epoch or entry["block"] != block or entry["kind"] != problem.kind
                or entry["cuts"] != cuts or not entry["committed"] or abs(entry["start_s"] - time_s) > 1e-6):
            raise ValueError("独立优化结果顺序或切点与训练不一致")
        for key in ("channel_gains", "client_cycles", "server_cycles", "up_bits", "down_bits"):
            expected, actual = getattr(problem, key), entry[key]
            if len(expected) != len(actual) or any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-20)
                                                 for a, b in zip(expected, actual)):
                raise ValueError(f"独立优化块的 {key} 与训练不一致")
        if not math.isclose(entry["budget_s"], problem.budget_s, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("独立优化块预算与训练不一致")
        self.read_blocks += 1
        return copy.deepcopy(entry["decision"])

    def finish(self):
        """完整训练结束时检查没有遗漏或多余的决策。"""
        if self.stream.readline().strip() or self.read_blocks != self.summary["blocks"]:
            raise ValueError("训练完成时仍有未使用的优化块")

    def close(self):
        """关闭逐块输入流；成功和失败路径均由训练入口调用。"""
        self.stream.close()
