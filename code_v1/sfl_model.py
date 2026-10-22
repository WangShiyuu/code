"""用途：连续飞行 UAV-SFL 的 ResNet18、统一初始化、等权聚合和计算画像。

运行指令（在 code_v1 目录执行，需先安装 PyTorch）：
    python -X utf8 -B sfl_model.py --split P4 --image-size 64 --batch-size 4
该命令只在内存中的模型副本上统计指定切点的 FLOPs 与字节数，并打印 JSON；
不读取训练图片，不更新模型，不生成测试文件。训练由 train_sfl.py 调用本模块。

输入为 RGB 图像；默认 64×64、CBCL 九分类。P0 位于 stem 之后，P1—P8
位于相应的完整残差块之后。模型保持完整参数状态，改变切点只移动现有层。
BN 始终使用固定的 running_mean/running_var，仿射参数 gamma/beta 仍可训练。
画像中的 FLOPs 仅覆盖 PyTorch FlopCounterMode 支持的算子，不是设备实测时延
或能量；激活与梯度按当前 float32 张量统计，标签按 int64 统计。
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterator, Sequence
import copy
import json
from pathlib import Path

import torch
from torch import nn


# 仅在完整残差块的边界切分，保证 shortcut 不会横跨无人机和服务器。
SPLIT_POINTS = tuple(f"P{index}" for index in range(9))
PROFILE_SCOPE = (
    "PyTorch FlopCounterMode 支持的算子；矩阵乘法/卷积按乘法与加法分别计数。"
    "未计优化器更新、数据读取、内存搬运、通信与 Python 开销；"
    "BN、激活、池化及损失等未被当前计数器覆盖的算子也可能漏计。"
    "计算时延和能耗须结合配置中的硬件参数估计，不能视为实测设备指标。"
)


def split_index(split: str | int) -> int:
    """将 P0—P8 转为前缀包含的残差块数，拒绝布尔值和范围外的切点。"""
    if isinstance(split, str):
        name = split.strip().upper()
        if name in SPLIT_POINTS:
            return int(name[1:])
    elif isinstance(split, int) and not isinstance(split, bool) and 0 <= split <= 8:
        return split
    raise ValueError(f"切分位置必须为 P0—P8 或整数 0—8，收到 {split!r}")


class BasicBlock(nn.Module):
    """标准 ResNet18 残差块；尺寸变化时同步变换 shortcut 后再相加。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """按输入/输出通道数建立残差块，stride=2 时将空间尺寸减半。"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """执行两次卷积和残差相加，不在残差块内部暴露切分边界。"""
        identity = value if self.downsample is None else self.downsample(value)
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class SplitResNet18(nn.Module):
    """同一完整模型提供可分离的前向接口，切点变化不重建网络。

    本项目使用 frozen BN：train() 不更新 BN 统计量，尾批只有一个样本时
    仍可训练。该设定与普通 SGD、每 epoch 累计步长的训练约定配合使用。
    """

    def __init__(self, num_classes: int = 9, bn_mode: str = "frozen"):
        """建立完整 ResNet18 并初始化卷积和 BN，不自动加载外部预训练权重。"""
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes 必须为不小于 2 的整数")
        if bn_mode != "frozen":
            raise ValueError("code_v1 的系统设定要求 bn_mode='frozen'")
        self.num_classes = num_classes
        self.bn_mode = bn_mode
        self.current_split = 0
        # 使用标准 ResNet18 stem，与归档模型的参数键及形状保持兼容。
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        specifications = (
            (64, 1), (64, 1), (128, 2), (128, 1),
            (256, 2), (256, 1), (512, 2), (512, 1),
        )
        blocks, in_channels = [], 64
        for number, (out_channels, stride) in enumerate(specifications, start=1):
            blocks.append((f"block{number}", BasicBlock(in_channels, out_channels, stride)))
            in_channels = out_channels
        self.residual_blocks = nn.Sequential(OrderedDict(blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.train()

    def train(self, mode: bool = True):
        """保持 BN 统计量冻结；gamma/beta 的 requires_grad 不受 eval() 影响。"""
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward_prefix(self, images: torch.Tensor, split: str | int) -> torch.Tensor:
        """计算无人机端前缀，返回仍保留前缀求导图的边界激活。"""
        count = split_index(split)
        value = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        for index, block in enumerate(self.residual_blocks):
            if index >= count:
                break
            value = block(value)
        return value

    def forward_suffix(self, activation: torch.Tensor, split: str | int) -> torch.Tensor:
        """计算服务器端后缀；P8 后缀仍包含平均池化与分类层。"""
        count = split_index(split)
        value = activation
        for index, block in enumerate(self.residual_blocks):
            if index >= count:
                value = block(value)
        return self.fc(torch.flatten(self.avgpool(value), 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """完整推理；允许前缀与后缀分处不同的计算设备。"""
        activation = self.forward_prefix(images.to(self.conv1.weight.device), self.current_split)
        return self.forward_suffix(activation.to(self.fc.weight.device), self.current_split)

    def move_split(self, split: str | int, client_device="cpu", server_device="cpu"):
        """移动现有层并保留参数对象及权重；本项目 SGD 没有待迁移动量状态。

        调用方先移动模型，再创建优化器最为直接。若后续改变切点，普通 SGD
        无动量、无权重衰减，不需要搬移额外的优化器张量。
        """
        count = split_index(split)
        client_device, server_device = torch.device(client_device), torch.device(server_device)
        if torch.__future__.get_overwrite_module_params_on_conversion():
            raise RuntimeError("请关闭 PyTorch 参数转换时替换 Parameter 对象的选项")
        # 在默认 PyTorch 转换规则下，Module.to 保留 Parameter 的身份。
        for module in (self.conv1, self.bn1, self.relu, self.maxpool):
            module.to(client_device)
        for index, block in enumerate(self.residual_blocks):
            block.to(client_device if index < count else server_device)
        self.avgpool.to(server_device)
        self.fc.to(server_device)
        self.current_split = count
        return self

    def prefix_state_names(self, split: str | int) -> set[str]:
        """返回前缀参数与 buffers 的键；模型同步须包含 BN 状态。"""
        count = split_index(split)
        prefixes = ("conv1.", "bn1.") + tuple(
            f"residual_blocks.block{number}." for number in range(1, count + 1)
        )
        return {name for name in self.state_dict() if name.startswith(prefixes)}

    def prefix_parameter_names(self, split: str | int) -> set[str]:
        """返回前缀中可学习参数的键，不包含 BN 的 buffers。"""
        names = self.prefix_state_names(split)
        return {name for name, _ in self.named_parameters() if name in names}

    def prefix_parameters(self, split: str | int) -> Iterator[nn.Parameter]:
        """返回现有前缀参数的迭代器，不克隆或替换参数。"""
        names = self.prefix_parameter_names(split)
        return (parameter for name, parameter in self.named_parameters() if name in names)


def make_common_models(
    num_uavs: int, num_classes: int = 9, seed: int = 42, checkpoint: str | Path | None = None,
) -> list[SplitResNet18]:
    """只初始化一次完整模型，再复制给各机，保证每架 UAV 的初始状态相同。

    可选 checkpoint 应为本模型 state_dict，或含 model_state_dict/state_dict
    字段的字典。严格检查键与形状，不自动下载或替换分类头。
    """
    if isinstance(num_uavs, bool) or not isinstance(num_uavs, int) or num_uavs <= 0:
        raise ValueError("num_uavs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    # 仅使用并恢复 CPU RNG；建模与复制不会改变调用方后续的数据打乱序列。
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        base = SplitResNet18(num_classes=num_classes)
    if checkpoint is not None:
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(saved, dict):
            raise ValueError("初始 checkpoint 必须为状态字典或包含状态字典的字典")
        state = saved.get("model_state_dict", saved.get("state_dict", saved))
        base.load_state_dict(state, strict=True)
    return [base] + [copy.deepcopy(base) for _ in range(num_uavs - 1)]


@torch.no_grad()
def equal_fedavg(models: Sequence[SplitResNet18]) -> None:
    """以 1/N 聚合完整模型并原地广播，保留所有 Parameter 对象身份。

    每架 UAV 样本容量及采集数量相同，因此本项目采用等权聚合。整数 BN
    计数在冻结 BN 时应保持完全相同；发现差异时拒绝聚合，避免掩盖状态错误。
    """
    if not models:
        raise ValueError("至少需要一个参与聚合的模型")
    states = [model.state_dict(keep_vars=True) for model in models]
    reference = states[0]
    # 全部校验先于写回，防止不兼容模型造成部分参数已经被改写。
    for model, state in zip(models, states):
        if model.bn_mode != "frozen" or state.keys() != reference.keys():
            raise ValueError("聚合模型必须具有一致结构，并使用冻结 BN")
        for name, tensor in reference.items():
            other = state[name]
            if other.shape != tensor.shape or other.dtype != tensor.dtype:
                raise ValueError(f"模型状态 {name} 的形状或类型不一致")
            if not tensor.is_floating_point() and not tensor.is_complex():
                if not torch.equal(tensor, other.to(tensor.device)):
                    raise ValueError(f"冻结 BN 的整数状态 {name} 不一致")
    for name, tensor in reference.items():
        if tensor.is_floating_point() or tensor.is_complex():
            # 逐张量累加，避免同时复制 N 个完整状态字典；低精度用 float32 累加。
            dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
            average = torch.zeros_like(tensor, dtype=dtype)
            for state in states:
                average.add_(state[name].to(device=tensor.device, dtype=dtype), alpha=1 / len(states))
            for state in states:
                state[name].copy_(average.to(device=state[name].device, dtype=state[name].dtype))
        # 相同的整数 buffers 无需改写。


def profile_splits(
    base: SplitResNet18, candidates: Sequence[str | int] = ("P4",),
    image_size: int = 64, batch_size: int = 4,
) -> dict[str, dict]:
    """在 CPU 模型副本上分别统计四段前后向 FLOPs 和精确张量载荷字节。

    不执行 optimizer.step，不改变输入模型或调用方 RNG。每个画像 batch
    最多使用 4 个样本，按样本归一化；卷积/线性层计数随 batch 数线性增长。
    float32 激活与 int64 标签的尺寸由真实张量取得，包含尾批时训练按实际
    样本数乘回。聚合时的 prefix_state_bytes 含参数及 BN buffers。
    """
    from torch.utils.flop_counter import FlopCounterMode

    for value, name in ((image_size, "image_size"), (batch_size, "batch_size")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    names = tuple(dict.fromkeys(f"P{split_index(cut)}" for cut in candidates))
    if not names:
        raise ValueError("至少需要一个切分画像")
    profiles = {}
    with torch.random.fork_rng(devices=[]):
        model = copy.deepcopy(base).cpu().float().train()
        batch = min(batch_size, 4)
        images = torch.randn(batch, 3, image_size, image_size)
        labels = torch.arange(batch, dtype=torch.long) % model.num_classes
        for cut in names:
            model.zero_grad(set_to_none=True)
            with FlopCounterMode(display=False) as counter:
                boundary = model.forward_prefix(images, cut)
            client_forward = counter.get_total_flops()
            # detach 明确切断前缀图，服务器反向只产生边界梯度。
            received = boundary.detach().clone().requires_grad_(True)
            with FlopCounterMode(display=False) as counter:
                logits = model.forward_suffix(received, cut)
                loss = nn.functional.cross_entropy(logits, labels)
            server_forward = counter.get_total_flops()
            with FlopCounterMode(display=False) as counter:
                loss.backward()
            server_backward = counter.get_total_flops()
            if received.grad is None:
                raise RuntimeError(f"{cut} 没有生成返回无人机的边界梯度")
            with FlopCounterMode(display=False) as counter:
                boundary.backward(received.grad.detach())
            client_backward = counter.get_total_flops()
            state = model.state_dict()
            prefix_bytes = sum(
                state[name].numel() * state[name].element_size()
                for name in model.prefix_state_names(cut)
            )
            profiles[cut] = {
                "client_forward_flops_per_sample": client_forward / batch,
                "client_backward_flops_per_sample": client_backward / batch,
                "server_forward_flops_per_sample": server_forward / batch,
                "server_backward_flops_per_sample": server_backward / batch,
                "client_flops_per_sample": (client_forward + client_backward) / batch,
                "server_flops_per_sample": (server_forward + server_backward) / batch,
                "activation_bytes_per_sample": boundary.numel() * boundary.element_size() // batch,
                "gradient_bytes_per_sample": received.grad.numel() * received.grad.element_size() // batch,
                "label_bytes_per_sample": labels.element_size(),
                "prefix_state_bytes": prefix_bytes,
                "profile_batch_size": batch,
                "image_size": image_size,
                "dtype": "float32",
                "flop_count_scope": PROFILE_SCOPE,
            }
    return profiles


def main() -> None:
    """独立打印模型画像，便于读取数量级；不写入文件或启动训练。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLIT_POINTS, nargs="+", default=["P4"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=9)
    args = parser.parse_args()
    model = make_common_models(1, num_classes=args.num_classes)[0]
    print(json.dumps(profile_splits(model, args.split, args.image_size, args.batch_size),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
