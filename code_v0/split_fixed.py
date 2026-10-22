# 用途：在手动指定的固定位置切分 ResNet18，使用 CBCL 进行 SplitFed 训练。
# 统一默认配置：普通 BN，SGD 学习率 0.01、动量 0.9、权重衰减 0.0005，batch size 32。
# 默认输入 64x64、seed 42；默认 --scheduler cosine，100 轮，每个全局 epoch 后递减一次。
# 最佳模型按验证准确率最高选择；准确率相同时选验证损失较低者，测试集仅用于最终评价。
# 首次运行先准备数据：python data.py --uavs 5 --partition iid --samples-per-uav 1000 1250 1500 1750 2000 --output output/data_load_7500
# 训练指令：python split_fixed.py --rounds 100 --split P4 --aggregation-period 1 --lr 0.01
# 各 UAV 可使用不同固定切点，例如 5 架 UAV：python split_fixed.py --rounds 100 --split P0 P2 P4 P6 P8
# 指定输出：python split_fixed.py --data output/data_load_7500/data.json --rounds 100 --split P4 --output output/split_fixed_run
# 每个 SL 轮次就是一个本地完整 epoch；切点 P0..P8 在全部训练轮次中保持固定。
"""Split fixed：手动指定各 UAV 的固定切点。参数说明：python split_fixed.py --help。"""
from __future__ import annotations

from models import split_index
from training_common import TrainingSession, common_parser


def run(args):
    """固定切点 SplitFed 的完整轮次循环。"""
    # 先检查切点是否合法；split_index 将 P0..P8 转成对应的残差块边界编号。
    fixed = [f'P{split_index(cut)}' for cut in args.split]
    session = TrainingSession(args)
    if len(fixed) == 1:
        fixed *= len(session.sizes)
    if len(fixed) != len(session.sizes):
        raise ValueError('--split 需要一个切点，或与 UAV 数量相同的切点列表')
    for epoch in range(args.rounds):
        start = session.begin_epoch()
        stats = []
        for uav in range(len(session.clients)):
            # 前缀在 UAV 上计算，激活和标签送到服务器后缀；服务器返回边界梯度。
            # 每架 UAV 拥有独立的前缀与服务器后缀，当前全部本地样本遍历一次。
            stats.append(session.train_one(epoch, uav=uav, cut=fixed[uav]))
        aggregated = session.should_aggregate(epoch)
        if aggregated:
            # 将各 UAV 完整模型和 BN 运行统计按相同规则聚合；服务器后缀在服务器内部聚合。
            # 广播后清空本地 SGD 动量，保留当前学习率；最后不足 K 轮仍执行聚合。
            session.aggregate()
        session.record(epoch, stats, fixed, aggregated, start, [0] if aggregated else [])
    return session.finish()


def main(argv=None):
    parser = common_parser(__doc__, 'split_fixed')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    # Windows 下使用多进程 DataLoader 时，此保护可防止子进程重复执行训练。
    main()