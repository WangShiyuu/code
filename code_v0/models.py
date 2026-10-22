"""用途：五种训练方式共用的标准 ResNet18、P0—P8 切分与 FedAvg 聚合。
训练指令（本文件由 training_common.py 和训练脚本导入，不单独启动训练）：
    python split_fixed.py --data output/data/data.json --rounds 100 --split P4 --output output/split_fixed

只使用 CBCL 九分类，输入为 64×64 的 RGB 图像。P0 位于 stem 之后；
P1—P8 分别位于第 1—8 个完整残差块之后，不切开残差分支。
修改切分位置仅移动现有层，不重新初始化模型，不丢失已训练的参数。
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
import math

import torch
from torch import nn


SPLIT_POINTS = tuple(f"P{i}" for i in range(9))


def split_index(split: str | int) -> int:
    """将 P0—P8 转为前缀包含的残差块数量，例如 P4 对应 4。"""
    if isinstance(split, str):
        text = split.strip().upper()
        if text in SPLIT_POINTS:
            return int(text[1:])
    elif isinstance(split, int) and not isinstance(split, bool) and 0 <= split <= 8:
        return split
    raise ValueError(f"切分位置必须为 P0—P8 或整数 0—8，收到 {split!r}")


class BasicBlock(nn.Module):
    """ResNet18 的完整残差块：两次卷积的结果加上 shortcut，再经过 ReLU。"""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 当空间尺寸或通道数变化时，shortcut 也要变换，才能逐元素相加。
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value if self.downsample is None else self.downsample(value)
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class SplitResNet18(nn.Module):
    """同一个完整模型既可普通训练，也可执行前缀／后缀分离训练。

    batch 默认使用普通 BN：训练更新统计量，评估使用统计量。
    可选 frozen 固定 BN 的 running_mean/running_var，但 gamma/beta 仍可学习。
    这让各无人机的小批次不会分别积累不同的 BN 统计量，也能处理最后一批
    只有一个样本的情况；batch 的单样本尾批由训练循环临时用累计统计量。
    所有训练方法必须使用相同的 BN 设置，才便于公平比较。
    """

    def __init__(self, num_classes: int = 9, bn_mode: str = "batch"):
        super().__init__()
        if bn_mode not in ("frozen", "batch"):
            raise ValueError("bn_mode 必须为 frozen 或 batch")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
            raise ValueError("num_classes 必须为不小于 2 的整数")
        self.num_classes = num_classes
        self.bn_mode = bn_mode
        self.current_split = 0
        # 使用标准 ResNet18 的 7×7 stem，不能替换成 CIFAR 风格的 3×3 stem。
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 四个 stage，各有两个残差块；每次通道数翻倍时，首块 stride=2。
        blocks = []
        inplanes = 64
        for number, (planes, stride) in enumerate(
            ((64, 1), (64, 1), (128, 2), (128, 1),
             (256, 2), (256, 1), (512, 2), (512, 1)), start=1
        ):
            blocks.append((f"block{number}", BasicBlock(inplanes, planes, stride)))
            inplanes = planes
        self.residual_blocks = nn.Sequential(OrderedDict(blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        # 卷积使用 Kaiming 初始化；BN 初始为缩放 1、偏置 0。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.train()

    def train(self, mode: bool = True):
        # 外部 model.train() 不能意外打开 frozen BN 的统计量更新。
        super().train(mode)
        if self.bn_mode == "frozen":
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward_prefix(self, images: torch.Tensor, split: str | int) -> torch.Tensor:
        """无人机前向计算；返回要发送给服务器的边界激活。"""
        count = split_index(split)
        value = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        for index, block in enumerate(self.residual_blocks):
            if index >= count:
                break
            value = block(value)
        return value

    def forward_suffix(self, activation: torch.Tensor, split: str | int) -> torch.Tensor:
        """服务器前向计算；P8 的后缀仍有全局平均池化与分类层。"""
        count = split_index(split)
        value = activation
        for index, block in enumerate(self.residual_blocks):
            if index >= count:
                value = block(value)
        return self.fc(torch.flatten(self.avgpool(value), 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """完整推理；兼容模型全部在一台设备或前后缀分处两台设备。"""
        images = images.to(self.conv1.weight.device)
        activation = self.forward_prefix(images, self.current_split)
        activation = activation.to(self.fc.weight.device)
        return self.forward_suffix(activation, self.current_split)

    def move_split(self, split: str | int, client_device, server_device):
        """调整现有层的设备和计算分界，保留参数对象与已学权重。

        移动模型不会自动移动 SGD 动量；training_common.training_epoch 在移动
        切分层后调用 move_optimizer_state，按每个参数的新设备同步迁移状态。
        """
        count = split_index(split)
        client_device, server_device = torch.device(client_device), torch.device(server_device)
        # PyTorch 默认 Module.to 会保留 Parameter 身份。若调用者开启了全局
        # 参数替换选项，提前报错，防止已有优化器继续引用旧参数。
        if torch.__future__.get_overwrite_module_params_on_conversion():
            raise RuntimeError("请关闭 torch 的 overwrite_module_params_on_conversion 后再移动切分层")
        for module in (self.conv1, self.bn1, self.relu, self.maxpool):
            module.to(client_device)
        for index, block in enumerate(self.residual_blocks):
            block.to(client_device if index < count else server_device)
        self.avgpool.to(server_device)
        self.fc.to(server_device)
        self.current_split = count
        return self

    def prefix_parameter_names(self, split: str | int) -> set[str]:
        """取得前缀参数名，用于统计无人机模型通信量（不含 BN buffers）。"""
        count = split_index(split)
        prefixes = ("conv1.", "bn1.") + tuple(
            f"residual_blocks.block{i}." for i in range(1, count + 1)
        )
        return {name for name, _ in self.named_parameters() if name.startswith(prefixes)}

    def prefix_parameters(self, split: str | int) -> Iterator[nn.Parameter]:
        """返回现有前缀参数；可据此统计参数字节数，无需复制参数。"""
        names = self.prefix_parameter_names(split)
        return (parameter for name, parameter in self.named_parameters() if name in names)


@torch.no_grad()
def weighted_fedavg(models: Sequence[SplitResNet18], data_sizes: Sequence[int]) -> None:
    """按各无人机实际样本数加权聚合，再原地广播给全部模型。

    浮点参数与 BN 浮点 buffers 使用加权平均；BN 的整数计数取参与者最大值。
    batch 模式下，BN 统计量的平均是模型状态聚合，不等于重算全体数据统计量。
    每次只建立一个张量的累加器，不复制 N 份完整模型，不替换 Parameter 对象。
    """
    if not models or len(models) != len(data_sizes):
        raise ValueError("models 与 data_sizes 必须非空且长度相同")
    counts = [float(count) for count in data_sizes]
    if any(not math.isfinite(count) or count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("样本量必须为非负有限数，且至少一个无人机有样本")
    if any(model.bn_mode != models[0].bn_mode for model in models):
        raise ValueError("聚合模型必须使用相同的 BN 模式")
    # state_dict(keep_vars=True) 是现有张量的引用，而不是完整张量的副本。
    states = [model.state_dict(keep_vars=True) for model in models]
    reference = states[0]
    for state in states[1:]:
        if state.keys() != reference.keys():
            raise ValueError("聚合模型的参数／buffer 名称不一致")
        if any(state[name].shape != tensor.shape or state[name].dtype != tensor.dtype
               for name, tensor in reference.items()):
            raise ValueError("聚合模型的参数／buffer 形状或类型不一致")
    total = sum(counts)
    for name, tensor in reference.items():
        if tensor.is_floating_point() or tensor.is_complex():
            # 半精度参数在 float32 中累加，降低多次相加造成的舍入误差。
            dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
            average = torch.zeros_like(tensor, dtype=dtype)
            for state, count in zip(states, counts):
                if count:
                    average.add_(state[name].to(device=tensor.device, dtype=dtype), alpha=count / total)
        else:
            contributors = [state[name] for state, count in zip(states, counts) if count]
            average = contributors[0].to(tensor.device).clone()
            for value in contributors[1:]:
                value = value.to(tensor.device)
                average = torch.logical_or(average, value) if tensor.dtype == torch.bool else torch.maximum(average, value)
        # copy_ 保留参数身份，因此既有优化器仍绑定同一组参数。
        for state in states:
            state[name].copy_(average.to(device=state[name].device, dtype=state[name].dtype))
