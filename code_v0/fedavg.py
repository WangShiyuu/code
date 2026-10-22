# 用途：各 UAV 用自己的 CBCL 样本训练完整 ResNet18，然后按样本量进行 FedAvg 聚合。
# 统一默认配置：普通 BN，SGD 学习率 0.01、动量 0.9、权重衰减 0.0005，batch size 32。
# 默认输入 64x64、seed 42；默认 --scheduler cosine，100 轮，每个全局 epoch 后递减一次。
# 最佳模型按验证准确率最高选择；准确率相同时选验证损失较低者，测试集仅用于最终评价。
# 首次运行先准备数据：python data.py --uavs 5 --partition iid --samples-per-uav 1000 1250 1500 1750 2000 --output output/data_load_7500
# 训练指令：python fedavg.py --rounds 100 --aggregation-period 1 --batch-size 32 --lr 0.01
# 指定数据与输出：python fedavg.py --data output/data_load_7500/data.json --rounds 100 --output output/fedavg_run
# 每轮每架 UAV 完整遍历自己的样本一次；--aggregation-period K 表示每 K 轮聚合。
"""FedAvg：完整模型本地训练后进行样本量加权平均。参数说明：python fedavg.py --help。"""
from __future__ import annotations

from training_common import TrainingSession, common_parser


def run(args):
    """FedAvg 的完整轮次循环：本地训练、定期聚合、评价和保存。"""
    session = TrainingSession(args)
    cuts = [None] * len(session.sizes)
    for epoch in range(args.rounds):
        start = session.begin_epoch()
        stats = []
        for uav in range(len(session.clients)):
            # 当前 UAV 训练自己的完整模型；一轮包含自己的全部样本，不限制 batch 数。
            stats.append(session.train_one(epoch, uav=uav))
        aggregated = session.should_aggregate(epoch)
        if aggregated:
            # 权重 = 该 UAV 样本数 / 总样本数；模型参数与 BN 运行统计按同一规则广播。
            # 广播后清空本地 SGD 动量，保留当前学习率；三种聚合方法均使用这个规则。
            # 最后不足 K 轮时仍聚合，以得到可评价和保存的最终全局模型。
            session.aggregate()
        # 仅在聚合后评价共享模型；其余轮次照常记录本地训练指标。
        session.record(epoch, stats, cuts, aggregated, start, [0] if aggregated else [])
    return session.finish()


def main(argv=None):
    parser = common_parser(__doc__, 'fedavg')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    # Windows 下使用多进程 DataLoader 时，此保护可防止子进程重复执行训练。
    main()