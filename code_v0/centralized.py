# 用途：使用所有 UAV 被分配的 CBCL 样本，集中训练一个完整 ResNet18。
# 统一默认配置：普通 BN，SGD 学习率 0.01、动量 0.9、权重衰减 0.0005，batch size 32。
# 默认输入 64x64、seed 42；默认 --scheduler cosine，100 轮，每个全局 epoch 后递减一次。
# 最佳模型按验证准确率最高选择；准确率相同时选验证损失较低者，测试集仅用于最终评价。
# 首次运行先准备数据：python data.py --uavs 5 --partition iid --samples-per-uav 1000 1250 1500 1750 2000 --output output/data_load_7500
# 训练指令：python centralized.py --rounds 100 --batch-size 32 --lr 0.01
# 指定数据与输出：python centralized.py --data output/data_load_7500/data.json --rounds 100 --output output/centralized_run
# 每轮把汇总训练集完整遍历一次；尾批保留，因此每个样本每轮只使用一次。
"""Centralized：所有 UAV 的训练样本汇总到同一个模型。参数说明：python centralized.py --help。"""
from __future__ import annotations

from training_common import TrainingSession, common_parser


def run(args):
    """集中训练的完整轮次循环；公共模块负责数据读取、记录和模型保存。"""
    session = TrainingSession(args)
    # centralized 只有一个模型。训练数据是全部 UAV 样本的并集，不额外加入验证或测试样本。
    cuts = [None] * len(session.sizes)
    for epoch in range(args.rounds):
        start = session.begin_epoch()
        stats = [session.train_one(epoch, centralized=True)]
        # 没有模型间聚合。每轮评价唯一模型，按验证准确率最高选 best.pt；准确率相同时选损失更低者。
        session.record(epoch, stats, cuts, False, start, [0])
    return session.finish()


def main(argv=None):
    parser = common_parser(__doc__, 'centralized')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    # Windows 下使用多进程 DataLoader 时，此保护可防止子进程重复执行训练。
    main()