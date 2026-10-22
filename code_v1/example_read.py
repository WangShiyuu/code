"""读取既有轨迹/CSI；不训练、不等待、不改变预生成的环境。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from config import ROOT
from environment import EnvironmentTrace


def rate_bps(bandwidth_hz, power_w, channel_gain, noise_psd_w_hz):
    """给定当前资源的Shannon速率；带宽Hz、功率W、增益为线性值。"""
    for name, value in (("bandwidth_hz", bandwidth_hz), ("power_w", power_w),
                        ("channel_gain", channel_gain), ("noise_psd_w_hz", noise_psd_w_hz)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}必须为有限正数")
    return bandwidth_hz * math.log1p(power_w * channel_gain / (noise_psd_w_hz * bandwidth_hz)) / math.log(2)


def main(argv=None):
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
        config = env.scenario["config"]
        snapshots = None
        if args.with_data:
            manifest_path = Path(meta["manifest_path"])
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != meta["manifest_sha256"]:
                raise ValueError("数据清单与生成环境时的版本不同")
            from data import AcquisitionSnapshots
            snapshots = AcquisitionSnapshots(manifest_path, env.collection)
        if args.times is None:
            times = [t for t in (0.0, 0.035, 1.23, 12.345) if t <= env.duration_s]
            if env.duration_s not in times:
                times.append(env.duration_s)
        else:
            times = args.times
        # 示例时钟由调用方给定；实际训练应按模型计算/传输/同步时间推进。
        for sim_time_s in times:
            state = env.state_at(sim_time_s, args.uav)
            bandwidth = config["radio"]["total_up_hz"] / len(env.uav_ids)
            current_rate = rate_bps(bandwidth, config["uav"]["p_up_ref_w"],
                                    state["channel_gain"], config["channel"]["noise_psd_w_hz"])
            row = dict(time_s=sim_time_s, uav_id=args.uav, position_m=state["position_m"],
                       channel_time_s=state["channel_time_s"], next_change_time_s=state["next_change_time_s"],
                       channel_gain=state["channel_gain"],
                       mean_channel_gain=env.window_mean(sim_time_s, config["selection"]["channel_window_size"])[args.uav],
                       available_samples=env.sample_count_at(sim_time_s),
                       reference_bandwidth_hz=bandwidth, reference_up_rate_bps=current_rate)
            if snapshots is not None:
                indices = snapshots.snapshot_indices(sim_time_s, args.uav)
                row.update(snapshot_size=len(indices), first_train_subset_indices=indices[:5])
            print(json.dumps(row, ensure_ascii=False, allow_nan=False))
        print("查询不会推进环境。若一次传输跨过next_change_time_s，后续需分段读取CSI；"
              "参考速率不是资源优化结果。")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"读取失败：{error}\n")


if __name__ == "__main__":
    main()
