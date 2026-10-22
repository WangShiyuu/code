# 用途：共用的数据读取、参数解析、单 epoch 更新、评价、日志与模型保存。
# 本模块由五个独立训练文件导入，训练流程位于对应文件中。
# 训练指令：python centralized.py --rounds 100
# python local.py --rounds 100
# python fedavg.py --rounds 100 --aggregation-period 1
# python split_fixed.py --rounds 100 --split P4
# python split_dynamic.py --rounds 100 --client-gflops 2 --server-gflops 100 --bandwidth-mbps 20
"""五种方法共用的小功能，完整参数见对应训练文件的 --help。

每个训练文件自己组织轮次、UAV 更新、切点选择和聚合；这里统一数据协议与输出。
一个 SL round 是一个完整本地 epoch。默认使用普通 BN、SGD 动量与权重衰减、余弦学习率。
SplitFed 在单进程中模拟通信，支持前后缀分处 CPU/CUDA，不建立真实无线网络。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import load_data
from experiment_config import configuration_snapshot, training_defaults
from models import SplitResNet18, split_index, weighted_fedavg

ROOT = Path(__file__).resolve().parent
METHODS = ('centralized', 'local', 'fedavg', 'split_fixed', 'split_dynamic')


def write_json(path, value):
    """中文直接写入 UTF-8；指标出现 NaN/Infinity 时立即报错。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('本代码使用 cpu 或 cuda/cuda:0 等设备')
    if device.type == 'cuda' and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'没有可用的 {device}，请使用 --device cpu')
    return device


def synchronize(*devices):
    # GPU 默认异步执行；计时时先等待计算完成，避免得到虚假的过短耗时。
    for device in set(map(str, devices)):
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(torch.device(device))


def expand_positive(values, count, name):
    values = list(values)
    if len(values) == 1:
        values *= count
    if len(values) != count or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f'{name} 需要一个正数（所有 UAV 共用），或 {count} 个正数')
    return values


def make_loader(dataset, batch_size, workers, seed=None):
    # shuffle=True 是一个完整随机排列，不会重复抽样；drop_last=False 保留尾批。
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None,
                      drop_last=False, num_workers=workers, generator=generator)


def make_scheduler(optimizer, name, epochs):
    """各方法用同一学习率计划；每个完整全局 epoch 结束后推进一次。"""
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
    raise ValueError('--scheduler 只能是 cosine 或 constant')


@torch.no_grad()
def move_optimizer_state(optimizer):
    """SGD 动量不随 model.to 自动移动，按每个参数的设备同步迁移其张量状态。"""
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value) and value.device != parameter.device:
                state[name] = value.to(parameter.device)


def training_epoch(model, dataset, optimizer, args, epoch, uav, client_device, server_device, cut=None):
    """只遍历一次 loader，因此每个样本在本轮恰好被处理一次。"""
    if cut is None:
        model.to(client_device)
    else:
        model.move_split(cut, client_device, server_device)
    # 切点改变或 CPU/CUDA 迁移后，历史动量必须与对应参数位于同一设备。
    move_optimizer_state(optimizer)
    model.train()
    loader = make_loader(dataset, args.batch_size, args.workers, args.seed + epoch * 100003 + uav * 997)
    loss_sum, correct, samples, steps, payload = 0.0, 0, 0, 0, 0
    for images, labels in loader:
        images = images.to(client_device)
        target_device = client_device if cut is None else server_device
        labels = labels.to(target_device)
        # 标准 BatchNorm 在 B=1 且空间 1x1 时不能估计方差。
        # 保留这一样本，临时使用已累计统计量；可选 frozen 模式始终使用固定统计量。
        if args.bn == 'batch' and len(labels) == 1:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        optimizer.zero_grad(set_to_none=True)
        if cut is None:
            logits = model(images)
        else:
            boundary = model.forward_prefix(images, cut)
            # detach 切断自动求导图，clone 模拟传输得到的独立张量。
            # 后缀只能看见激活与标签；稍后显式返回边界梯度。
            received = boundary.detach().to(server_device).clone().requires_grad_(True)
            logits = model.forward_suffix(received, cut)
            payload += boundary.numel() * boundary.element_size() + labels.numel() * labels.element_size()
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('损失非有限，请检查学习率、数据或初始权重')
        loss.backward()
        if cut is not None:
            returned = received.grad.detach().to(client_device)
            boundary.backward(returned)
            payload += returned.numel() * returned.element_size()
        # 两侧反向全部完成后才更新，保证 split 和完整网络 SGD 数学一致。
        optimizer.step()
        batch_count = len(labels)
        loss_sum += float(loss.detach()) * batch_count
        correct += int((logits.detach().argmax(1) == labels).sum())
        samples += batch_count
        steps += 1
        model.train()  # 恢复下一批 BN 的模式（frozen 模式由模型重载保持冻结）。
    if samples != len(dataset) or steps != math.ceil(len(dataset) / args.batch_size):
        raise AssertionError('本轮样本数或 batch 数与完整 epoch 定义不符')
    return dict(loss_sum=loss_sum, correct=correct, samples=samples, steps=steps, split_payload_bytes=payload)


@torch.no_grad()
def evaluate(model, dataset, args, device, num_classes):
    """统一评价：loss、accuracy、Macro-F1、混淆矩阵；不更新任何模型参数。"""
    model.to(device)
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum = 0.0
    for images, labels in make_loader(dataset, args.eval_batch_size, args.workers):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
        pred, target = logits.argmax(1).cpu().numpy(), labels.cpu().numpy()
        np.add.at(confusion, (target, pred), 1)
    count = int(confusion.sum())
    if count == 0:
        raise ValueError('评价集为空，请重新生成数据划分')
    tp = np.diag(confusion).astype(float)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    model.cpu()  # 独立分支依次使用计算设备，减少同时占用的 GPU 显存。
    return dict(loss=loss_sum / count, accuracy=float(tp.sum() / count), macro_f1=float(f1.mean()),
                samples=count, class_f1=f1.tolist(), class_support=confusion.sum(1).tolist(),
                confusion_matrix=confusion.tolist())


def checkpoint(path, model, classes, epoch, args, manifest_hash):
    # 保存 CPU 权重，其他机器没有 GPU 也能读取；这是模型权重文件，并非断点续训文件。
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    classes=classes, round=epoch, method=args.method, bn_mode=args.bn,
                    manifest_sha256=manifest_hash), path)


def load_initial(model, path, classes, backbone_only=False):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not backbone_only and saved.get('classes', classes) != classes:
        raise ValueError('权重文件的类别顺序与 CBCL 数据清单不一致')
    state = saved.get('model_state_dict', saved.get('state_dict', saved))
    # torchvision ResNet18 与当前模型结构相同，只把 layer1.0..layer4.1 名称换为 block1..8。
    mapped = {}
    for name, value in state.items():
        name = name.removeprefix('module.')
        pieces = name.split('.')
        if pieces[0] in ('layer1', 'layer2', 'layer3', 'layer4'):
            block = (int(pieces[0][-1]) - 1) * 2 + int(pieces[1]) + 1
            name = f'residual_blocks.block{block}.' + '.'.join(pieces[2:])
        mapped[name] = value
    if backbone_only:
        mapped = {k: v for k, v in mapped.items() if not k.startswith('fc.')}
        result = model.load_state_dict(mapped, strict=False)
        if set(result.missing_keys) != {'fc.weight', 'fc.bias'} or result.unexpected_keys:
            raise ValueError('初始权重不是兼容的完整 ResNet18 主干')
    else:
        model.load_state_dict(mapped, strict=True)


def profile_splits(base, args, candidates):
    """统计真实 ResNet18 算子的前后向 FLOPs，给动态切点成本模型使用。

    在副本上执行，没有 optimizer.step，不会改变训练权重和 BN 状态。
    只计数 FlopCounterMode 已覆盖的算子；不计 optimizer.step，BN 等未覆盖算子也可能漏计。
    FLOPs 是计算量；GFLOPS 与带宽是用户给定的硬件假设，预测耗时不是实测无线耗时或能耗。
    """
    from torch.utils.flop_counter import FlopCounterMode
    # fork_rng 恢复 CPU 随机状态，动态方式不会因 profiling 改变训练随机数。
    profiles = {}
    with torch.random.fork_rng(devices=[]):
        model = copy.deepcopy(base).cpu().train()
        if args.bn == 'batch' and args.batch_size == 1:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        batch = min(args.batch_size, 4)
        images = torch.randn(batch, 3, args.image_size, args.image_size)
        labels = torch.arange(batch) % base.num_classes
        for cut in candidates:
            model.zero_grad(set_to_none=True)
            with FlopCounterMode(display=False) as c:
                boundary = model.forward_prefix(images, cut)
            client_forward = c.get_total_flops()
            received = boundary.detach().clone().requires_grad_(True)
            with FlopCounterMode(display=False) as c:
                logits = model.forward_suffix(received, cut)
                loss = nn.functional.cross_entropy(logits, labels)
            server_forward = c.get_total_flops()
            with FlopCounterMode(display=False) as c:
                loss.backward()
            server_backward = c.get_total_flops()
            with FlopCounterMode(display=False) as c:
                boundary.backward(received.grad.detach())
            client_backward = c.get_total_flops()
            names = model.prefix_parameter_names(cut)
            prefix_bytes = sum(v.numel() * v.element_size() for k, v in model.state_dict().items()
                               if k in names or any(k.startswith(n.rsplit('.', 1)[0] + '.') for n in names))
            profiles[cut] = dict(client_flops_per_sample=(client_forward + client_backward) / batch,
                                 server_flops_per_sample=(server_forward + server_backward) / batch,
                                 activation_bytes_per_sample=boundary.numel() * boundary.element_size() / batch,
                                 gradient_bytes_per_sample=received.grad.numel() * received.grad.element_size() / batch,
                                 label_bytes_per_sample=labels.element_size(), prefix_state_bytes=prefix_bytes)
    return profiles


def dynamic_inputs(args, count):
    """每行 trace 表示一轮开始时的可用上/下行带宽；这里只用当前行，避免使用未来信息。"""
    compute = expand_positive(args.client_gflops, count, '--client-gflops')
    bandwidth = expand_positive(args.bandwidth_mbps, count, '--bandwidth-mbps')
    trace = [bandwidth[:] for _ in range(args.rounds)]
    if args.bandwidth_trace:
        raw = json.loads(Path(args.bandwidth_trace).read_text(encoding='utf-8-sig'))
        if len(raw) != args.rounds:
            raise ValueError('--bandwidth-trace 的 JSON 需要恰好 rounds 行，每行一个带宽或 N 个带宽')
        trace = [expand_positive([row] if isinstance(row, (int, float)) else row, count, 'trace 行') for row in raw]
    manual = None
    if args.split_schedule:
        raw = json.loads(Path(args.split_schedule).read_text(encoding='utf-8-sig'))
        if len(raw) != args.rounds:
            raise ValueError('--split-schedule 的 JSON 需要恰好 rounds 行')
        manual = []
        for row in raw:
            cuts = [row] * count if isinstance(row, str) else list(row)
            if len(cuts) != count:
                raise ValueError('切点日程每行必须为一个 P 值或 N 个 P 值')
            manual.append([f'P{split_index(cut)}' for cut in cuts])
        for r in range(args.rounds):
            if manual[r] != manual[(r // args.aggregation_period) * args.aggregation_period]:
                raise ValueError('同一 aggregation-period 内切点必须固定；请设置 period=1 或调整日程')
    return compute, trace, manual


def choose_cuts(profiles, sizes, compute, bandwidth, server_gflops, period):
    """保留旧调用接口；实际算法集中到 split_selection，避免维护两份选点逻辑。"""
    from split_selection import choose_latency_cuts
    return choose_latency_cuts(profiles, sizes, compute, bandwidth, server_gflops, period)


def write_history(path, rows):
    # 每轮立即落盘，训练意外中断也能查看已完成的轮数。
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



class TrainingSession:
    """管理一个实验的共用状态；五个训练文件明确控制各自的训练循环。

    统一记录和评价能避免五份代码在样本计数、输出格式或测试集使用上产生差异。
    train_one 只执行一个完整 epoch；聚合和切点策略由调用者安排。
    """

    def __init__(self, args):
        self.args = args
        if args.method not in METHODS:
            raise ValueError('未知训练方式')
        for name in ('rounds', 'aggregation_period', 'batch_size', 'eval_batch_size', 'cpu_threads'):
            if getattr(args, name) <= 0:
                raise ValueError(f'{name} 必须是正整数')
        if args.image_size < 32 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError('image-size >= 32、workers >= 0、lr 为正有限数')
        if args.method == 'split_dynamic' and (not math.isfinite(args.server_gflops) or args.server_gflops <= 0):
            raise ValueError('--server-gflops 必须为正有限数')
        if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
            raise ValueError('momentum 必须在 [0,1) 内')
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            raise ValueError('weight-decay 必须是非负有限数')
        if args.scheduler not in ('cosine', 'constant'):
            raise ValueError('--scheduler 只能是 cosine 或 constant')
        if args.backbone_only and not args.checkpoint:
            raise ValueError('--backbone-only 必须同时指定 --checkpoint')
        seed_all(args.seed)
        torch.set_num_threads(args.cpu_threads)
        self.client_device = resolve_device(args.client_device or args.device)
        self.server_device = resolve_device(args.server_device or args.device)
        manifest_path = Path(args.data).resolve()
        if manifest_path.is_dir():
            manifest_path /= 'data.json'
        self.train, self.validation, self.test, indices, self.meta = load_data(manifest_path, args.image_size)
        self.classes = self.meta['classes']
        self.sizes = [len(index) for index in indices]
        if not self.sizes or min(self.sizes) <= 0 or sum(self.sizes) != len(self.train):
            raise ValueError('每架 UAV 必须非空，且客户端样本总数必须等于训练集大小')
        self.clients = [Subset(self.train, index) for index in indices]
        self.manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output = (Path(args.output) if args.output else ROOT/'output'/args.method/f'{stamp}_seed{args.seed}').resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError(f'输出目录已有内容，请选一个新的 --output：{self.output}')
        base = SplitResNet18(num_classes=len(self.classes), bn_mode=args.bn)
        if args.checkpoint:
            load_initial(base, args.checkpoint, self.classes, args.backbone_only)
        # Centralized 只有一个模型；其他方式的每个 UAV 从相同权重开始。
        self.models = [base] if args.method == 'centralized' else [base]+[copy.deepcopy(base) for _ in self.sizes[1:]]
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                           weight_decay=args.weight_decay, foreach=False) for model in self.models]
        self.schedulers = [make_scheduler(optimizer, args.scheduler, args.rounds) for optimizer in self.optimizers]
        self.momentum_policy = ('reset local momentum after each model aggregation/broadcast'
                                if args.method in ('fedavg', 'split_fixed', 'split_dynamic')
                                else 'retain momentum across epochs')
        self.bn_aggregation = 'sample-weighted floating BN buffers; integer counters use maximum'
        self.selection = 'maximum validation accuracy; minimum validation loss on ties'
        config = dict(vars(args), output=str(self.output), data=str(manifest_path), dataset='CBCL StreetScenes',
                      classes=self.classes, client_sizes=self.sizes, client_device_resolved=str(self.client_device),
                      server_device_resolved=str(self.server_device), manifest_sha256=self.manifest_hash,
                      experiment_config=configuration_snapshot(args, args.method),
                      torch=str(torch.__version__), optimizer='SGD with momentum and weight decay',
                      lr_scheduler=type(self.schedulers[0]).__name__, scheduler_step='once after every global epoch',
                      momentum_policy=self.momentum_policy, bn_aggregation=self.bn_aggregation,
                      shuffle_protocol='seed + epoch_index*100003 + uav_id*997; full permutation',
                      early_stopping=False, energy_measured=False,
                      round_definition='1 SL round = 1 complete shuffled local epoch, drop_last=False',
                      split_semantics='independent UAV prefix and per-UAV server suffix; full-model weighted aggregation',
                      test_selection=self.selection+'; test selected model(s) once after all training')
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output/'config.json', config)
        write_json(self.output/'data_manifest.json', self.meta)
        self.history, self.schedule = [], []
        self.best_loss = [math.inf] * (len(self.sizes) if args.method == 'local' else 1)
        self.best_accuracy = [-1.0] * len(self.best_loss)
        self.best_round = [0] * len(self.best_loss)
        self.total_samples, self.total_payload = 0, 0
        self.started = time.perf_counter()
        print(f'{args.method}: UAV 样本量={self.sizes}, client={self.client_device}, server={self.server_device}, output={self.output}', flush=True)

    def begin_epoch(self):
        """训练轮计时包含本轮更新与聚合，不包含验证集评价。"""
        synchronize(self.client_device, self.server_device)
        return time.perf_counter()

    def train_one(self, epoch, uav=0, cut=None, centralized=False):
        """在指定数据上完整遍历一次，再把模型移回 CPU，减少 GPU 显存占用。"""
        model_index = 0 if centralized else uav
        dataset = self.train if centralized else self.clients[uav]
        result = training_epoch(self.models[model_index], dataset, self.optimizers[model_index], self.args,
                                epoch, uav, self.client_device, self.server_device, cut)
        self.models[model_index].cpu()
        # 动量也暂存 CPU，避免多个 UAV 的优化器同时占用 GPU 显存。
        move_optimizer_state(self.optimizers[model_index])
        return result

    def aggregate(self):
        """三种聚合方法共用：按样本数平均模型/BN，再广播，并清空本地动量。"""
        weighted_fedavg(self.models, self.sizes)
        # 广播后客户端参数已经改变，下一聚合区间从新权重重新积累本地动量。
        # 仅清 state，不重建优化器；当前学习率与 scheduler 的轮次不会重置。
        for optimizer in self.optimizers:
            optimizer.state.clear()

    def should_aggregate(self, epoch):
        """每 K 轮聚合，最后不足 K 轮也完成聚合；epoch 从 0 开始计数。"""
        return (epoch+1) % self.args.aggregation_period == 0 or epoch+1 == self.args.rounds

    def save_profiles(self, profiles):
        write_json(self.output/'split_profile.json', dict(profiles=profiles,
                   cost_model='D*(Cc/Fu+Cs/Fs+8*(A+G+label)/bandwidth)+16*M/(bandwidth*K)',
                   objective='minimum_modeled_latency', energy_measured=False,
                   scope='covered forward/backward operator FLOPs and modeled serial per-UAV latency; excludes optimizer updates, uncovered operators (such as BN), data loading, device/cut migration, wireless protocol and queueing cost'))

    def record(self, epoch, stats, cuts, aggregated, start, val_models):
        """审计完整 epoch，按验证准确率选择 best，最后推进下一轮学习率。"""
        args = self.args
        learning_rate = self.optimizers[0].param_groups[0]['lr']
        if any(optimizer.param_groups[0]['lr'] != learning_rate for optimizer in self.optimizers):
            raise AssertionError('各 UAV 本轮学习率不一致')
        processed = sum(s['samples'] for s in stats)
        if processed != sum(self.sizes):
            raise AssertionError('所有 UAV 的样本没有恰好遍历一次')
        # 通信只统计张量载荷。动量保存在本地；CPU/GPU 状态迁移不算 UAV 无线通信。
        payload = sum(s['split_payload_bytes'] for s in stats)
        if aggregated:
            if args.method == 'fedavg':
                state_bytes = sum(v.numel()*v.element_size() for v in self.models[0].state_dict().values())
                payload += 2*len(self.sizes)*state_bytes
            else:
                for model, cut in zip(self.models, cuts):
                    names = model.prefix_parameter_names(cut)
                    payload += 2*sum(v.numel()*v.element_size() for k,v in model.state_dict().items()
                                     if k in names or any(k.startswith(n.rsplit('.',1)[0]+'.') for n in names))
        synchronize(self.client_device, self.server_device)
        train_seconds = time.perf_counter()-start
        self.total_samples += processed
        self.total_payload += payload
        val_metrics = []
        for model_index in val_models:
            metrics = evaluate(self.models[model_index], self.validation, args, self.server_device, len(self.classes))
            val_metrics.append(metrics)
            # Local 每个 UAV 分别保存最优轮次；聚合方法只评价全局模型 models[0]。
            best_index = model_index if args.method == 'local' else 0
            if (metrics['accuracy'] > self.best_accuracy[best_index]
                    or (metrics['accuracy'] == self.best_accuracy[best_index]
                        and metrics['loss'] < self.best_loss[best_index])):
                self.best_accuracy[best_index] = metrics['accuracy']
                self.best_loss[best_index], self.best_round[best_index] = metrics['loss'], epoch+1
                destination = self.output/f'uav_{model_index:02d}' if args.method == 'local' else self.output
                checkpoint(destination/'best.pt', self.models[model_index], self.classes,
                           epoch+1, args, self.manifest_hash)
        row = dict(round=epoch+1, epoch=epoch+1, learning_rate=learning_rate, train_loss=sum(s['loss_sum'] for s in stats)/processed,
                   train_accuracy=sum(s['correct'] for s in stats)/processed,
                   processed_samples=processed, total_processed_samples=self.total_samples,
                   uav_samples=json.dumps(self.sizes), optimizer_steps=json.dumps([s['steps'] for s in stats]),
                   splits=json.dumps(cuts) if args.method.startswith('split_') else '', aggregated=aggregated,
                   val_loss=float(np.mean([m['loss'] for m in val_metrics])) if val_metrics else '',
                   val_accuracy=float(np.mean([m['accuracy'] for m in val_metrics])) if val_metrics else '',
                   val_macro_f1=float(np.mean([m['macro_f1'] for m in val_metrics])) if val_metrics else '',
                   uav_val_loss=json.dumps([m['loss'] for m in val_metrics]) if args.method == 'local' else '',
                   uav_val_accuracy=json.dumps([m['accuracy'] for m in val_metrics]) if args.method == 'local' else '',
                   training_seconds=train_seconds, payload_bytes=payload, total_payload_bytes=self.total_payload)
        self.history.append(row)
        write_history(self.output/'history.csv', self.history)
        if self.schedule:
            write_json(self.output/'split_schedule.json', self.schedule)
        # 未到聚合轮时也要推进学习率：一轮仍是所有 UAV 完整遍历一次。
        for scheduler in self.schedulers:
            scheduler.step()
        val_text = f", val_acc={row['val_accuracy']:.4f}" if val_metrics else ''
        print(f"round {epoch+1}/{args.rounds}: lr={learning_rate:.6g}, samples={processed}, loss={row['train_loss']:.5f}{val_text}, cuts={cuts}", flush=True)

    def finish(self):
        """训练结束后，保存 last 并且只评价验证集选出的 best；不重新训练。"""
        args = self.args
        expected = args.rounds*sum(self.sizes)
        if self.total_samples != expected:
            raise AssertionError('实际样本处理总量不等于 rounds × 总训练样本量')
        results = []
        final_models = self.models if args.method == 'local' else self.models[:1]
        for i, model in enumerate(final_models):
            destination = self.output/f'uav_{i:02d}' if args.method == 'local' else self.output
            checkpoint(destination/'last.pt', model, self.classes, args.rounds, args, self.manifest_hash)
            saved = torch.load(destination/'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model_state_dict'])
            metrics = evaluate(model, self.test, args, self.server_device, len(self.classes))
            results.append(dict(uav=i if args.method == 'local' else None, selected_round=self.best_round[i], **metrics))
        summary = dict(method=args.method, rounds=args.rounds, client_sizes=self.sizes, processed_samples=self.total_samples,
                       expected_processed_samples=expected, best_rounds=self.best_round, early_stopping=False,
                       best_val_accuracies=self.best_accuracy, best_val_losses=self.best_loss,
                       bn=args.bn, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                       scheduler=args.scheduler, model_selection=self.selection,
                       momentum_policy=self.momentum_policy, bn_aggregation=self.bn_aggregation,
                       test_accuracy=float(np.mean([m['accuracy'] for m in results])),
                       test_accuracy_std=float(np.std([m['accuracy'] for m in results])),
                       test_macro_f1=float(np.mean([m['macro_f1'] for m in results])), test_details=results,
                       evaluation='arithmetic mean of independent local models on shared test set' if args.method == 'local' else 'selected global model',
                       training_payload_bytes=self.total_payload,
                       payload_scope='estimated activation/label/gradient and same-cut aggregation upload+download; excludes optimizer state, cut migration, initialization, dataset collection, protocol and evaluation',
                       training_seconds=sum(row['training_seconds'] for row in self.history),
                       timing_scope='single-process simulation on this host; training_seconds includes data loading, device transfers and aggregation; wall_seconds also includes profiling, validation, final testing and checkpoint/output I/O',
                       energy_measured=False,
                       wall_seconds=time.perf_counter()-self.started, output=str(self.output))
        write_json(self.output/'summary.json', summary)
        print(f"完成：test_acc={summary['test_accuracy']:.4f}, test_macro_f1={summary['test_macro_f1']:.4f}, {self.output}", flush=True)
        return summary


def common_parser(description, method):
    """从统一配置取得默认值；显式 CLI 参数优先，训练方法名仍由文件决定。"""
    if method not in METHODS:
        raise ValueError('未知训练方式')
    defaults = training_defaults(method)
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(method=method, aggregation_period=defaults['aggregation_period'])
    parser.add_argument('--data', default=defaults['data'], help='data.py 生成的清单或其目录；默认值来自 experiment_config.py')
    parser.add_argument('--rounds','--epochs',dest='rounds',type=int,default=defaults['rounds'],help='总 SL 轮数，每轮就是一个完整 epoch')
    if method in ('fedavg','split_fixed','split_dynamic'):
        parser.add_argument('--aggregation-period',type=int,default=defaults['aggregation_period'],help='每 K 个完整 epoch 聚合；最后不足 K 轮也聚合')
    parser.add_argument('--batch-size',type=int,default=defaults['batch_size'],help='训练 mini-batch 大小，保留最后不足一批的样本')
    parser.add_argument('--eval-batch-size',type=int,default=defaults['eval_batch_size'])
    parser.add_argument('--image-size',type=int,default=defaults['image_size'],help='输入边长；与 data.py 准备尺寸一致可减少插值')
    parser.add_argument('--lr',type=float,default=defaults['lr'],help='SGD 初始学习率')
    parser.add_argument('--momentum',type=float,default=defaults['momentum'],help='SGD 动量，0 表示关闭')
    parser.add_argument('--weight-decay',type=float,default=defaults['weight_decay'],help='SGD 权重衰减，0 表示关闭')
    parser.add_argument('--scheduler',choices=('cosine','constant'),default=defaults['scheduler'],help='余弦下降或固定学习率，每完整轮推进一次')
    parser.add_argument('--seed',type=int,default=defaults['seed'],help='模型初始化与每轮排列种子；数据划分由清单固定')
    parser.add_argument('--bn',choices=('batch','frozen'),default=defaults['bn'],help='默认普通 BN；可选固定统计量，均训练 gamma/beta')
    parser.add_argument('--device',default=defaults['device'],help='auto/cpu/cuda/cuda:0；同时用于 UAV 和服务器')
    parser.add_argument('--client-device',default=defaults['client_device'],help='单独设置 UAV 计算设备')
    parser.add_argument('--server-device',default=defaults['server_device'],help='单独设置服务器计算和评价设备')
    parser.add_argument('--cpu-threads',type=int,default=defaults['cpu_threads'],help='CPU 计算线程数')
    parser.add_argument('--workers',type=int,default=defaults['workers'],help='DataLoader 工作进程；Windows 初学者推荐 0')
    if method == 'split_fixed':
        parser.add_argument('--split',nargs='+',default=defaults['split'],help='固定切点 P0..P8；一个值共用，也可逐 UAV 指定')
    if method == 'split_dynamic':
        parser.add_argument('--candidates',nargs='+',default=defaults['candidates'],help='动态候选切点')
        parser.add_argument('--client-gflops',type=float,nargs='+',default=defaults['client_gflops'],help='UAV 算力；一个值共用或逐 UAV 配置')
        parser.add_argument('--server-gflops',type=float,default=defaults['server_gflops'],help='现有动态时延模型的服务器计算能力，不是物理频率')
        parser.add_argument('--bandwidth-mbps',type=float,nargs='+',default=defaults['bandwidth_mbps'],help='现有模型的 UAV 上/下行速率，不是带宽 Hz')
        parser.add_argument('--bandwidth-trace',help='动态带宽 JSON 文件：[rounds][UAV]')
        parser.add_argument('--split-schedule',help='手动动态切点 JSON：[rounds][UAV]，跳过自动选择')
    parser.add_argument('--checkpoint',help='可选本地 ResNet18 初始权重；不自动下载权重')
    parser.add_argument('--backbone-only',action='store_true',help='明确只加载主干，CBCL 九分类头保持随机初始化')
    parser.add_argument('--output',help='空实验目录；默认 code/output/方法/时间戳_seed')
    return parser
