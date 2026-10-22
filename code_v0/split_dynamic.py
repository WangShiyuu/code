# 用途：根据 UAV/服务器算力与当前带宽，为 ResNet18 选择动态切点并使用 CBCL 训练。
# 统一默认配置：普通 BN，SGD 学习率 0.01、动量 0.9、权重衰减 0.0005，batch size 32。
# 默认输入 64x64、seed 42；默认 --scheduler cosine，100 轮，每个全局 epoch 后递减一次。
# 最佳模型按验证准确率最高选择；准确率相同时选验证损失较低者，测试集仅用于最终评价。
# 首次运行先准备数据：python data.py --uavs 5 --partition iid --samples-per-uav 1000 1250 1500 1750 2000 --output output/data_load_7500
# 训练指令：python split_dynamic.py --rounds 100 --client-gflops 2 --server-gflops 100 --bandwidth-mbps 20 --lr 0.01
# 设置候选切点：python split_dynamic.py --rounds 100 --candidates P2 P4 P6 --aggregation-period 1
# 随轮次改变带宽：python split_dynamic.py --rounds 100 --bandwidth-trace bandwidth.json --output output/split_dynamic_run
# 手动切点日程：python split_dynamic.py --rounds 100 --split-schedule cuts.json
# 每个 SL 轮次就是一个本地完整 epoch；每 K=aggregation-period 轮开始时重新选切点。
# 时延选择算法位于 split_selection.py；新的物理能耗 trade-off 暂只在该文件独立规划。
"""Split dynamic：按当前资源条件选择切点，也支持手动日程。参数说明：python split_dynamic.py --help。"""
from __future__ import annotations

from models import split_index
from split_selection import choose_latency_cuts
from training_common import (
    TrainingSession, common_parser, dynamic_inputs, profile_splits,
)


def run(args):
    """动态切点 SplitFed 的完整循环：资源读取、切点选择、训练和聚合。"""
    candidates = sorted({f'P{split_index(cut)}' for cut in args.candidates}, key=split_index)
    session = TrainingSession(args)
    compute, trace, manual = dynamic_inputs(args, len(session.sizes))
    profiles = None
    if manual is None:
        # 在模型副本上统计各切点的计算量和通信张量大小，不更新实际训练模型。
        # 结合用户提供的 GFLOPS/Mbps，得到预计耗时；真实训练耗时单独记录。
        profiles = profile_splits(session.models[0], args, candidates)
        session.save_profiles(profiles)
    cuts = [None] * len(session.sizes)
    for epoch in range(args.rounds):
        costs = None
        if epoch % args.aggregation_period == 0:
            # 一个聚合周期内固定切点，聚合完成后再选择，避免周期内迁移切点。
            remaining = min(args.aggregation_period, args.rounds - epoch)
            if manual is not None:
                cuts = manual[epoch]
            else:
                # 只使用本轮带宽，不读取未来带宽。各 UAV 独立比较候选切点。
                cuts, costs = choose_latency_cuts(profiles, session.sizes, compute, trace[epoch],
                                         args.server_gflops, remaining)
        session.schedule.append(dict(round=epoch + 1, cuts=cuts[:], bandwidth_mbps=trace[epoch],
                                     selection_cost_seconds=costs,
                                     selection='manual' if manual is not None else 'minimum_modeled_latency'))
        start = session.begin_epoch()
        stats = []
        for uav in range(len(session.clients)):
            # 当前切点决定前缀与后缀的位置；激活送往服务器，边界梯度返回 UAV。
            # 每轮仍完整遍历每架 UAV 持有的全部样本，切点不改变样本使用次数。
            stats.append(session.train_one(epoch, uav=uav, cut=cuts[uav]))
        aggregated = session.should_aggregate(epoch)
        if aggregated:
            # 不同切点对应同一个 ResNet18 结构；完整参数与 BN 运行统计按相同规则聚合。
            # 广播后清空本地 SGD 动量，保留当前学习率；最后不足 K 轮仍执行聚合。
            session.aggregate()
        session.record(epoch, stats, cuts, aggregated, start, [0] if aggregated else [])
    return session.finish()


def main(argv=None):
    parser = common_parser(__doc__, 'split_dynamic')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    # Windows 下使用多进程 DataLoader 时，此保护可防止子进程重复执行训练。
    main()