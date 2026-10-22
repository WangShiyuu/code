# 用途：每架 UAV 仅使用自己的 CBCL 样本，独立训练一个完整 ResNet18。
# 统一默认配置：普通 BN，SGD 学习率 0.01、动量 0.9、权重衰减 0.0005，batch size 32。
# 默认输入 64x64、seed 42；默认 --scheduler cosine，100 轮，每个全局 epoch 后递减一次。
# 最佳模型按验证准确率最高选择；准确率相同时选验证损失较低者，测试集仅用于最终评价。
# 首次运行先准备数据：python data.py --uavs 5 --partition iid --samples-per-uav 1000 1250 1500 1750 2000 --output output/data_load_7500
# 训练指令：python local.py --rounds 100 --batch-size 32 --lr 0.01
# 指定数据与输出：python local.py --data output/data_load_7500/data.json --rounds 100 --output output/local_run
# 每轮每架 UAV 完整遍历自己的全部样本一次；所有模型之间不交换或聚合参数。
"""Local：每架 UAV 独立训练与保存自己的模型。参数说明：python local.py --help。"""
from __future__ import annotations

from training_common import TrainingSession, common_parser


def run(args):
    """本地独立训练的完整轮次循环。"""
    session = TrainingSession(args)
    cuts = [None] * len(session.sizes)
    model_indices = list(range(len(session.models)))
    for epoch in range(args.rounds):
        start = session.begin_epoch()
        stats = []
        for uav in range(len(session.clients)):
            # UAV 之间使用不同模型；train_one 只访问当前 UAV 的全部本地样本。
            stats.append(session.train_one(epoch, uav=uav))
        # 各模型在共同验证集上分别按准确率最高选 best；相同时选损失更低者，存入各自 uav_XX 子目录。
        session.record(epoch, stats, cuts, False, start, model_indices)
    return session.finish()


def main(argv=None):
    parser = common_parser(__doc__, 'local')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    # Windows 下使用多进程 DataLoader 时，此保护可防止子进程重复执行训练。
    main()