"""作用：演示后续训练如何按模拟时间读取位置、信道和累计采集样本。

运行指令（在 code_v1 目录下，先用 prepare_scenario.py 生成场景）：
    python -X utf8 -B example_read.py
    python -X utf8 -B example_read.py --with-data
    python -X utf8 -B example_read.py --uav 2 --times 0 12.345 600
    python -X utf8 -B example_read.py --scenario output/scenario/scenario.json
    python -X utf8 -B example_read.py --help

输入：scenario.json 及其对应的 environment.csv；--with-data 额外读取场景绑定的
data.json。默认 UAV 编号为0，所有查询时间均为任务起点之后的秒数。
输出：在控制台逐行打印位置、CSI观测时间、下一变化时刻、窗口均值、样本数量和
参考上行速率；--with-data 还显示索引快照大小及前几个训练子集局部索引。

这是只读示例，不推进真实时间、不生成训练结果、不修改CSV。默认只需要标准库；
--with-data 需要NumPy和Pillow，但不会读取图像张量或启动PyTorch训练。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from config import ROOT
from environment import EnvironmentTrace


def rate_bps(bandwidth_hz, power_w, channel_gain, noise_psd_w_hz):
    """按当前资源计算 Shannon 参考速率，返回 bit/s。

    bandwidth_hz为带宽Hz，power_w为发射功率W，channel_gain为线性功率增益，
    noise_psd_w_hz为噪声功率谱密度W/Hz。先以N0*B得到带宽内噪声功率，
    再计算 B*log2(1+p*h/(N0*B))。这里不选择资源，不代表资源优化结果。
    """
    for name, value in (("bandwidth_hz", bandwidth_hz), ("power_w", power_w),
                        ("channel_gain", channel_gain), ("noise_psd_w_hz", noise_psd_w_hz)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}必须为有限正数")
    # log1p(x)精确计算ln(1+x)，低信噪比时比先计算1+x更稳定；除以ln(2)换底。
    return bandwidth_hz * math.log1p(power_w * channel_gain / (noise_psd_w_hz * bandwidth_hz)) / math.log(2)


def main(argv=None):
    """读取场景并打印若干时刻的示例状态；所有查询均使用同一份既有物理过程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=str(ROOT / "output" / "scenario" / "scenario.json"))
    parser.add_argument("--uav", type=int, default=0)
    parser.add_argument("--times", type=float, nargs="+", help="查询模拟时间（秒）")
    parser.add_argument("--with-data", action="store_true", help="同时读取真实样本索引快照；需NumPy和Pillow")
    args = parser.parse_args(argv)
    try:
        scenario_path = Path(args.scenario).expanduser().resolve()
        meta = json.loads(scenario_path.read_text(encoding="utf-8-sig"))
        csv_path = scenario_path.parent / meta["trace"]["csv"]
        env = EnvironmentTrace(csv_path, scenario_path)
        # 使用生成此CSV时保存的配置，避免当前config.py后来被修改而混用两组参数。
        config = env.scenario["config"]
        snapshots = None
        if args.with_data:
            # 必须匹配生成环境时的数据清单版本，否则索引序列和采集容量可能不一致。
            manifest_path = Path(meta["manifest_path"])
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != meta["manifest_sha256"]:
                raise ValueError("数据清单与生成环境时的版本不同")
            from data import AcquisitionSnapshots
            snapshots = AcquisitionSnapshots(manifest_path, env.collection)
        # 默认包含非采样节点，展示坐标插值及CSI保持；短任务只保留范围内的示例时间。
        if args.times is None:
            times = [t for t in (0.0, 0.035, 1.23, 12.345) if t <= env.duration_s]
            if env.duration_s not in times:
                times.append(env.duration_s)
        else:
            times = args.times
        # sim_time_s是模拟时钟，由调用方给定；实际训练按计算/传输/同步时延推进。
        # 不使用电脑墙钟，也不sleep；对未来查询的示例不表示算法可提前看到未来CSI。
        for sim_time_s in times:
            state = env.state_at(sim_time_s, args.uav)
            # 均分带宽与参考功率仅用于演示速率公式，实际训练可传入资源分配器的值。
            bandwidth = config["radio"]["total_up_hz"] / len(env.uav_ids)
            current_rate = rate_bps(bandwidth, config["uav"]["p_up_ref_w"],
                                    state["channel_gain"], config["channel"]["noise_psd_w_hz"])
            row = dict(time_s=sim_time_s, uav_id=args.uav, position_m=state["position_m"],
                       channel_time_s=state["channel_time_s"], next_change_time_s=state["next_change_time_s"],
                       channel_gain=state["channel_gain"],
                       mean_channel_gain=env.window_mean(sim_time_s, config["selection"]["channel_window_size"])[args.uav],
                       available_samples=env.sample_count_at(sim_time_s),
                       reference_bandwidth_hz=bandwidth, reference_up_rate_bps=current_rate)
            # 只展示前几个索引，避免把数千个样本编号刷满控制台；完整快照仍可由接口返回。
            if snapshots is not None:
                indices = snapshots.snapshot_indices(sim_time_s, args.uav)
                row.update(snapshot_size=len(indices), first_train_subset_indices=indices[:5])
            print(json.dumps(row, ensure_ascii=False, allow_nan=False))
        print("查询不会推进环境。若一次传输跨过next_change_time_s，后续需分段读取CSI；"
              "参考速率不是资源优化结果。")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"读取失败：{error}\n")


if __name__ == "__main__":
    # 作为模块导入rate_bps等函数时，不会自动执行读取演示。
    main()
