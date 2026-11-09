"""复核一次切分决策：在同一预测窗口为每个组合分别分配资源，输出能耗明细。"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

import uav_sfl as s


def decision_snapshot(directory, epoch):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    config = s.load_config(str(directory / "config.json"))
    profile = s.grouped_profile(read("split_profiles.json"))
    reference = s.reference_resources(config)
    with (directory / "decisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["epoch"] == epoch:
                selection = row["selection"]
                if not selection:
                    raise ValueError("请选择首轮或非最终聚合轮")
                recovery = selection.get("reference_recovery")
                if recovery and recovery["ok"]:
                    reference = recovery["resources"]
                return dict(config=config, profile=profile, epoch=epoch, count=row["count_per_uav"],
                            current=row["split"], gain=selection["window_gain"],
                            remaining=row["remaining_before_j"],
                            reference={k: np.asarray(v) for k, v in reference.items()})
            reference = row["resources"]
    raise ValueError("日志中找不到该决策轮")


def compare_candidate(cuts, snapshot):
    return s.forecast_split_energy(cuts, **snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="已完成的动态策略仿真目录")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    snapshot = decision_snapshot(args.run_dir, args.epoch)
    candidates = product(s.SPLIT_POINTS, repeat=snapshot["config"]["num_uavs"])
    evaluate = partial(compare_candidate, snapshot=snapshot)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, candidates, chunksize=4))
    rows.sort(key=lambda row: (row[1] is None, row[1], row[0]))
    output = args.run_dir / f"optimized_split_comparison_epoch_{args.epoch}.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "forecast_energy_j", "status"])
        writer.writerows((json.dumps(cuts), energy, status) for cuts, energy, status in rows)
    print(f"{output}: {sum(r[1] is not None for r in rows)}/{len(rows)} feasible; best row = {rows[0]}")


if __name__ == "__main__":
    main()
