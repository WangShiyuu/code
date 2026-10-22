# 用途：用完整 ResNet18 做简单的 CBCL 中心式准确率基线，不限制数据集样本数量。
# 数据指令：python data.py --uavs 1 --partition iid --seed 42 --image-size 64 --output output/data_full
# 不要添加 --train-samples 或 --samples-per-uav；所有有效区域进入 train/val/test。
# 训练指令：python resnet18_full.py --data output/data_full/data.json --mode holdout --epochs 100 --batch-size 32 --lr 0.01 --output output/resnet18_baseline
# 全量拟合：python resnet18_full.py --data output/data_full/data.json --mode all --epochs 100 --output output/resnet18_all
# 一个 epoch 遍历当前训练集合一次，保留尾批；每次实验请使用新的空 output 目录。
"""简单的 ResNet18 中心式基线。参数：python resnet18_full.py --help。

默认 holdout：训练全部 train，验证集选择最佳权重，最后在独立 test 上评价一次。
可选 all：合并 train/val/test 全部训练，评价拟合准确率，不能解释成独立测试准确率。
普通 BatchNorm + SGD 动量 + 余弦学习率；没有 UAV 更新、切点或聚合。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from data import load_data
from experiment_config import configuration_snapshot, training_defaults
from models import SplitResNet18
from training_common import load_initial, make_loader, make_scheduler, resolve_device, seed_all, write_history, write_json

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """评价时关闭梯度和 BN 更新，统计准确率、各类别 F1 与混淆矩阵。"""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
        np.add.at(confusion, (labels.cpu().numpy(), logits.argmax(1).cpu().numpy()), 1)
    count = int(confusion.sum())
    if not count:
        raise ValueError("评价集不能为空")
    tp = np.diag(confusion).astype(float)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return dict(loss=loss_sum/count, accuracy=float(tp.sum()/count), macro_f1=float(f1.mean()),
                samples=count, class_f1=f1.tolist(), class_support=confusion.sum(1).tolist(),
                confusion_matrix=confusion.tolist())


def save_weights(path, model, classes, epoch, args, manifest_hash, training_subsets):
    """保存可移到 CPU 的权重及数据来源；记录 all 模式，防止它被当作干净的测试初始权重。"""
    torch.save(dict(model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    classes=classes, round=epoch, method=args.method, bn_mode=args.bn,
                    mode=args.mode, training_subsets=training_subsets, manifest_sha256=manifest_hash), path)


def run(args):
    """一个模型、一个优化器、一个训练循环；按 mode 选择训练与评价集合。"""
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.cpu_threads) < 1:
        raise ValueError("epochs、batch-size、eval-batch-size 和 cpu-threads 必须是正整数")
    if args.image_size < 32 or args.workers < 0:
        raise ValueError("image-size 至少为 32，workers 不能为负数")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr 必须为正有限数")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError("momentum 必须在 [0,1) 内")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay 必须为非负有限数")
    if args.backbone_only and not args.checkpoint:
        raise ValueError("--backbone-only 需要同时指定 --checkpoint")
    seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    manifest_path = Path(args.data).resolve()
    if manifest_path.is_dir():
        manifest_path /= "data.json"
    train, validation, test, _, meta = load_data(manifest_path, args.image_size)
    all_data = ConcatDataset([train, validation, test])
    all_valid = meta["dataset_report"]["usable_samples"]
    # 检查清单覆盖全部有效区域，避免误用之前只抽取 1500 个训练样本的清单。
    if len(all_data) != all_valid:
        raise ValueError(
            f"该清单只包含 {len(all_data)}/{all_valid} 个有效区域，请重新生成完整数据：\n"
            "python data.py --uavs 1 --partition iid --seed 42 --output output/data_full\n"
            "省略 --train-samples 和 --samples-per-uav，并用 --data 指向新的 data.json。"
        )
    holdout = args.mode == "holdout"
    training_data = train if holdout else all_data
    selection_data = validation if holdout else all_data
    final_data = test if holdout else all_data
    training_subsets = ["train"] if holdout else ["train", "val", "test"]
    selection_name, final_name = ("val", "test") if holdout else ("all_data", "all_data")
    interpretation = ("held-out test accuracy of the model selected by validation accuracy"
                      if holdout else "training-fit accuracy; no independent generalization test")
    classes = meta["classes"]
    source_counts = dict(train=len(train), val=len(validation), test=len(test))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (Path(args.output) if args.output else ROOT/"output/resnet18_full"/f"{stamp}_{args.mode}_seed{args.seed}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录已有内容，请使用新的 --output：{output}")
    # 两个残差块 × 四个 stage 的标准 ResNet18，始终执行完整网络；默认随机初始化。
    model = SplitResNet18(num_classes=len(classes), bn_mode=args.bn)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if holdout:
            if saved.get("mode") == "all" or set(saved.get("training_subsets", [])) & {"val", "test"}:
                raise ValueError("该初始权重训练过验证/测试数据，不能用于独立测试基线")
            if saved.get("classes") == classes and saved.get("manifest_sha256") != manifest_hash:
                raise ValueError("CBCL 初始权重来自不同或未知划分，无法保证测试集独立；请使用随机初始化或外部预训练主干")
        del saved
        load_initial(model, args.checkpoint, classes, args.backbone_only)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, foreach=False)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs)
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=False)
    selection_loader = DataLoader(selection_data, batch_size=args.eval_batch_size, shuffle=False, **loader_options)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", dict(vars(args), data=str(manifest_path), output=str(output),
               experiment_config=configuration_snapshot(args, args.method),
               device_resolved=str(device), classes=classes, manifest_sha256=manifest_hash,
               dataset="CBCL StreetScenes", dataset_samples=len(all_data), training_samples=len(training_data),
               source_subset_counts=source_counts, training_subsets=training_subsets, torch=str(torch.__version__),
               optimizer="SGD with momentum and weight decay", lr_scheduler=type(scheduler).__name__,
               scheduler_step="once after every global epoch", momentum_policy="retain momentum across epochs",
               shuffle_protocol="seed + epoch_index*100003 + uav_id*997; full permutation",
               selection=f"maximum {selection_name} accuracy; minimum loss on ties", evaluation=interpretation))
    write_json(output/"data_manifest.json", meta)
    history, total_samples = [], 0
    best_accuracy, best_loss, best_epoch = -1.0, math.inf, 0
    start = time.perf_counter()
    print(f"mode={args.mode}, 全部有效区域={len(all_data)}, 每轮训练={len(training_data)}, 集合={source_counts}, device={device}, output={output}", flush=True)
    for epoch in range(1, args.epochs+1):
        # 与 centralized.py 共用每轮的完整排列种子，方便用相同数据交叉核对。
        train_loader = make_loader(training_data, args.batch_size, args.workers, args.seed + (epoch-1)*100003)
        model.train()
        loss_sum, correct, processed, steps = 0.0, 0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 单样本尾批无法在 1x1 特征上估计 BN 方差，临时用累计统计量，保留这个样本。
            if args.bn == "batch" and len(labels) == 1:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("损失非有限，请检查数据或降低学习率")
            loss.backward()
            optimizer.step()
            processed += len(labels)
            steps += 1
            loss_sum += float(loss.detach())*len(labels)
            correct += int((logits.detach().argmax(1) == labels).sum())
            model.train()
        if processed != len(training_data) or steps != math.ceil(len(training_data)/args.batch_size):
            raise AssertionError("本轮没有完整遍历训练集合")
        total_samples += processed
        # holdout 只读取验证集来选模型；训练循环内不评价测试集。
        metrics = evaluate(model, selection_loader, device, len(classes))
        row = dict(epoch=epoch, learning_rate=learning_rate, train_loss=loss_sum/processed,
                   train_accuracy=correct/processed, processed_samples=processed, steps=steps)
        row.update({f"{selection_name}_loss": metrics["loss"], f"{selection_name}_accuracy": metrics["accuracy"],
                    f"{selection_name}_macro_f1": metrics["macro_f1"]})
        history.append(row)
        write_history(output/"history.csv", history)
        if metrics["accuracy"] > best_accuracy or (metrics["accuracy"] == best_accuracy and metrics["loss"] < best_loss):
            best_accuracy, best_loss, best_epoch = metrics["accuracy"], metrics["loss"], epoch
            save_weights(output/"best.pt", model, classes, epoch, args, manifest_hash, training_subsets)
        scheduler.step()
        print(f"epoch {epoch}/{args.epochs}: lr={learning_rate:.6g}, train_acc={correct/processed:.4f}, {selection_name}_acc={metrics['accuracy']:.4f}", flush=True)
    save_weights(output/"last.pt", model, classes, args.epochs, args, manifest_hash, training_subsets)
    # 恢复最佳权重。holdout 直到此处才读取 test；all 则在同一全量训练集上检查拟合。
    model.load_state_dict(torch.load(output/"best.pt", map_location=device, weights_only=True)["model_state_dict"])
    final_loader = DataLoader(final_data, batch_size=args.eval_batch_size, shuffle=False, **loader_options)
    final_metrics = evaluate(model, final_loader, device, len(classes))
    if total_samples != args.epochs*len(training_data):
        raise AssertionError("样本处理总量与完整 epoch 预算不一致")
    summary = dict(method=args.method, mode=args.mode, epochs=args.epochs, dataset_samples=len(all_data),
                   training_samples=len(training_data), source_subset_counts=source_counts,
                   training_subsets=training_subsets, all_valid_regions=all_valid,
                   bn=args.bn, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                   scheduler=args.scheduler, momentum_policy="retain momentum across epochs",
                   processed_samples=total_samples, best_epoch=best_epoch,
                   wall_seconds=time.perf_counter()-start, output=str(output), evaluation=interpretation)
    summary.update({f"best_{selection_name}_accuracy": best_accuracy, f"best_{selection_name}_loss": best_loss,
                    f"{final_name}_accuracy": final_metrics["accuracy"], f"{final_name}_macro_f1": final_metrics["macro_f1"],
                    f"{final_name}_details": final_metrics})
    write_json(output/"summary.json", summary)
    print(f"完成：{final_name}_accuracy={final_metrics['accuracy']:.4f}, {final_name}_macro_f1={final_metrics['macro_f1']:.4f}, {output}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    defaults = training_defaults("resnet18_full")
    parser.set_defaults(method="resnet18_full")
    parser.add_argument("--data", default=defaults["data"], help="不限制样本容量的完整数据清单")
    parser.add_argument("--mode", choices=("holdout", "all"), default=defaults["mode"], help="独立测试基线，或合并所有数据检查拟合")
    parser.add_argument("--epochs", "--rounds", dest="epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--momentum", type=float, default=defaults["momentum"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default=defaults["scheduler"], help="与五种方法共用学习率计划")
    parser.add_argument("--bn", choices=("batch", "frozen"), default=defaults["bn"], help="默认使用普通 BatchNorm")
    parser.add_argument("--image-size", type=int, default=defaults["image_size"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--cpu-threads", type=int, default=defaults["cpu_threads"])
    parser.add_argument("--workers", type=int, default=defaults["workers"], help="Windows 默认 0")
    parser.add_argument("--checkpoint", help="可选本地 ResNet18 初始权重；不得训练过当前验证/测试数据")
    parser.add_argument("--backbone-only", action="store_true", help="只加载主干，重新初始化九分类头")
    parser.add_argument("--output", help="空实验目录；默认 output/resnet18_full/时间戳_mode_seed")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
