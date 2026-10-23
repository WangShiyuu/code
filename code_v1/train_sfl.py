"""作用：连续采集 UAV-SFL 统一入口，默认运行周期切分与块资源优化主算法。

运行指令（在 code_v1 目录，先安装 requirements-training.txt）：
    python -X utf8 -B train_sfl.py
    python -X utf8 -B train_sfl.py --optimization-results output/output_时间戳_git哈希 --save-checkpoint
    python -X utf8 -B train_sfl.py --prepare-only
    python -X utf8 -B train_sfl.py --prepared-run output/output_时间戳_git哈希 --save-checkpoint
    python -X utf8 -B train_sfl.py --mode diagnostic --epochs 2 --split P4
    python -X utf8 -B train_sfl.py --help

先运行 prepare_scenario.py --stage all 准备真实数据。动态模式默认100 epoch，
每2 epoch聚合，独立信道预演确定1.2倍的逐轮预算，然后生成完整分片长环境。
--prepare-only 不更新网络；--prepared-run 绑定已有准备包执行。求解不收敛或
原约束未通过时停止，保存失败原因和已提交批次，不回退固定资源。

训练使用同一初始 ResNet18、冻结 BN、普通 SGD 和按快照大小归一化的步长。
每个 epoch 在共同物理时间冻结各机累计样本，完整遍历，保留尾批。每批执行
前缀前向、激活/标签上传、后缀前后向、梯度下传、前缀反向，然后批末同步。
本机顺序计算独立模型分支；模拟时间按服务器分支并行取最慢完成时间。

--mode diagnostic 保留旧固定切分、参考资源和逐CSV节点传输诊断，默认2 epoch；
动态模式的块内CSI冻结。后续比较算法放在其他代码中。统一模型的初始下发视为起飞前已完成，不计入此次
统计；最后聚合模型留在服务器，不再下发用于下一周期的前缀。

每次自动新建 output/output_本地时间戳_git哈希/，保存 summary.json、report.md、
key_metrics.csv 和环境输入副本。--output 指定父目录，仍会在其下新建本次
output_时间戳_git哈希目录；--save-checkpoint 可另存最终聚合模型。电脑墙钟耗时、模拟时延和模型
估计能耗分开报告；不能据此宣称 UAV 实测能耗或达到理论收敛条件。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from run_output import create_run_directory, source_revision
from config import ROOT, load_config, resolve_collection
from data import AcquisitionSnapshots, CBCLDataset, validate_manifest
from environment import EnvironmentTrace
from report_results import write_run_report
from sfl_model import SPLIT_POINTS, equal_fedavg, make_common_models, profile_splits
from transmission import TraceHorizonError, transmit


def _positive_int(value):
    """解析命令行正整数，拒绝零或负值，避免空训练或空评价。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def parse_args(argv=None):
    """解析诊断规模与路径；物理、采集和训练参数仍读取统一 config.py。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="可选 JSON 局部配置，规则与场景入口一致")
    parser.add_argument("--mode", choices=("dynamic", "diagnostic"), default="dynamic",
                        help="默认动态主算法；diagnostic 保留固定切分参考资源诊断")
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare-only", action="store_true", help="只预演成本及准备长环境，不更新神经网络")
    preparation.add_argument("--prepared-run", type=Path, help="使用已有 output_时间戳_git哈希 准备包执行训练")
    parser.add_argument("--optimization-results", type=Path,
                        help="读取独立优化结果目录或 optimization.json；训练不再运行求解器")
    parser.add_argument("--scenario", type=Path, help="含 environment.csv/scenario.json 的目录")
    parser.add_argument("--manifest", type=Path, help="真实 data.json；默认读取配置中的数据输出目录")
    parser.add_argument("--output", type=Path, default=ROOT / "output",
                        help="输出父目录；每次自动新建 output_时间戳_git哈希 子目录，保留历次运行")
    parser.add_argument("--epochs", type=_positive_int, help="动态默认100；诊断默认2；须为聚合间隔的倍数")
    parser.add_argument("--split", choices=SPLIT_POINTS, help="本次固定切点，默认配置 initial_split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="本机执行设备，默认读取配置")
    parser.add_argument("--threads", type=_positive_int, default=min(8, os.cpu_count() or 1),
                        help="本机 PyTorch CPU 线程数，不是模拟服务器分支数")
    parser.add_argument("--eval-samples", type=_positive_int, default=256,
                        help="固定验证子集大小，上限为完整验证集大小；不读取测试集")
    parser.add_argument("--save-checkpoint", action="store_true", help="另存最终聚合模型 checkpoint.pt")
    return parser.parse_args(argv)


def archive_environment(trace, scenario_dir, output):
    """归档本次实际使用的环境文件，不复制可复用的图片缓存。

    每份结果携带其 CSV、参数快照及轨迹图，Git 同步时可核对本次外部输入。
    数据清单仍由 summary 中的 SHA256 和指纹绑定。复制后再次核对 CSV，
    防止读取之后源文件被其他进程修改，导致归档与执行过程不一致。
    """
    folder = output / "environment"
    folder.mkdir()
    for name in ("environment.csv", "scenario.json", "patrol_map.svg"):
        source = scenario_dir / name
        if name == "patrol_map.svg" and not source.exists():
            continue
        shutil.copy2(source, folder / name)
    expected = trace.scenario["trace"]["csv_sha256"]
    if hashlib.sha256((folder / "environment.csv").read_bytes()).hexdigest() != expected:
        raise ValueError("归档 CSV 与本次加载的环境不一致，源文件可能在运行时被修改")
    if json.loads((folder / "scenario.json").read_text(encoding="utf-8-sig")) != trace.scenario:
        raise ValueError("归档场景参数与本次加载的环境不一致")
    return folder


def load_inputs(config, scenario_dir, manifest_path):
    """绑定 CSV 与数据清单，校验环境来源，并构造按需读图的数据集。

    不再次扫描所有未使用的 PNG；每张实际读取的图片由 Pillow 解码检查。
    清单结构、清单 SHA256、数据指纹和 CSV SHA256 均与场景元数据核对。
    硬件参数可为诊断而调整；轨迹、信道、服务器位置和采集规则必须与 CSV
    一致，避免把不同场景的参数与旧文件混用。
    """
    trace = EnvironmentTrace(scenario_dir / "environment.csv", scenario_dir / "scenario.json")
    scenario = trace.scenario
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    validate_manifest(manifest)
    if hashlib.sha256(raw).hexdigest() != scenario["manifest_sha256"]:
        raise ValueError("数据清单与场景 SHA256 不一致，请用该清单重新生成环境")
    if manifest["fingerprint"] != scenario["data_fingerprint"]:
        raise ValueError("数据指纹与场景不一致")
    for group in ("mission", "channel"):
        if config[group] != scenario["config"][group]:
            raise ValueError(f"{group} 已改变，请先生成对应环境，或使用对应配置")
    if config["server"]["position_m"] != scenario["config"]["server"]["position_m"]:
        raise ValueError("服务器位置与环境不一致，请重新生成环境")
    collection = resolve_collection(config, manifest)
    if collection != trace.collection:
        raise ValueError("采集规则与环境不一致，请重新生成环境")
    if config["data"]["image_size"] != manifest["settings"]["image_size"]:
        raise ValueError("图像尺寸与数据缓存不一致，请重新准备数据")
    datasets = [CBCLDataset(manifest_path.parent,
                           [row for row in manifest["samples"] if row["split"] == split],
                           config["data"]["image_size"]) for split in ("train", "val")]
    if not len(datasets[1]):
        raise ValueError("诊断需要非空验证集")
    return trace, manifest, datasets[0], datasets[1], AcquisitionSnapshots(manifest, collection)


def _sync(device):
    """等待 CUDA 操作完成，让本机阶段计时包含真实计算，而非仅计提交命令。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, dataset, indices, batch_size, workers, device):
    """评价同一个固定验证子集，返回平均交叉熵与准确率，不推进模拟时间。"""
    was_training = model.training
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("验证损失出现非有限值，请检查步长及模型状态")
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    model.train(was_training)
    return {"samples": count, "mean_loss": total_loss / count, "accuracy": correct / count}


def split_step(model, optimizer, images, labels, cut, step_size, device, *, commit=True):
    """执行一次精确 SFL 更新，返回损失、正确数、实际字节和本机阶段计时。

    detach/clone 模拟跨设备边界；后缀反向产生切分梯度，再传回前缀反向。
    两侧梯度均基于更新前权重求得，最后统一 SGD.step。这里的张量搬运是
    单机训练实现；无线传输时长由独立物理模拟器计算，不通过 sleep 实现。
    """
    model.train()
    images, labels = images.to(device), labels.to(device)
    for group in optimizer.param_groups:
        group["lr"] = step_size
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    start = time.perf_counter()
    boundary = model.forward_prefix(images, cut)
    _sync(device)
    after_forward = time.perf_counter()
    received = boundary.detach().clone().requires_grad_(True)
    logits = model.forward_suffix(received, cut)
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("训练损失出现非有限值，请检查 epoch_step_size；不会自动裁剪梯度或修改训练协议")
    loss.backward()
    _sync(device)
    after_server = time.perf_counter()
    if received.grad is None or not torch.isfinite(received.grad).all():
        raise FloatingPointError("切分层返回梯度缺失或非有限")
    boundary.backward(received.grad.detach())
    # 检查梯度而非静默跳过异常更新；诊断应暴露不稳定的原始参数。
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise FloatingPointError("模型梯度出现非有限值")
    # 动态模式先收齐所有 UAV 的有限梯度，再由调用方统一提交；诊断兼容旧行为。
    if commit:
        optimizer.step()
    _sync(device)
    end = time.perf_counter()
    return {
        "mean_loss": loss.item(), "correct": (logits.argmax(1) == labels).sum().item(),
        "samples": labels.numel(),
        "activation_bytes": boundary.numel() * boundary.element_size(),
        "gradient_bytes": received.grad.numel() * received.grad.element_size(),
        "label_bytes": labels.numel() * labels.element_size(),
        "wall_client_forward_s": after_forward - start,
        "wall_server_forward_backward_s": after_server - after_forward,
        "wall_client_backward_and_update_s": end - after_server,
    }


class PhysicalExecution:
    """记录参考资源下的物理时序和能耗；不求解优化，不施加任务预算。

    上下行各均分总带宽，所有分支独立并行，未使用的份额不重新分配。
    每次通信保持当前资源，随 CSV 节点累计比特；每批结束以最慢分支同步。
    等待期间仍按轨迹移动、采集，不增加悬停能耗。UAV 下行接收和本机
    数据处理的额外功耗尚未标定，不自动填零或并入发射能量。
    """

    def __init__(self, trace, config, profile):
        """绑定环境和画像，创建每机计算/发射及服务器计算/发射计量器。"""
        self.trace, self.config, self.profile = trace, config, profile
        self.n = config["mission"]["num_uavs"]
        self.time_s = 0.0
        self.up_hz = config["radio"]["total_up_hz"] / self.n
        self.down_hz = config["radio"]["total_down_hz"] / self.n
        self.uavs = [dict(compute_j=0.0, transmit_j=0.0, barrier_wait_s=0.0,
                          compute_s=0.0, up_s=0.0, down_s=0.0) for _ in range(self.n)]
        self.server = dict(compute_j=0.0, transmit_j=0.0, aggregation_j=0.0)
        self.communication = {name: {"bytes": 0, "transmissions": 0, "csi_segments": 0,
                                     "cross_boundary_transmissions": 0}
                              for name in ("activations_and_labels_up", "gradients_down",
                                           "prefix_state_up", "prefix_state_down")}
        # 只沿每个并行块的最慢分支累计，四项之和等于模拟总时长；各机
        # 累计工作时间另外保留，不能将 N 条并行链路时长相加当作完成时间。
        self.critical_path = dict(uav_compute_s=0.0, server_compute_s=0.0,
                                  communication_s=0.0, aggregation_s=0.0)
        self.blocks = 0

    def _require_coverage(self, time_s):
        """检查外部数据覆盖范围；此检查不是训练的时延可行性约束。"""
        if time_s > self.trace.duration_s:
            raise ValueError(f"CSV 仅覆盖 {self.trace.duration_s:g}s，诊断需至少 {time_s:g}s；"
                             "请延长环境，并显式保留原采集速率，不能将末行信道外推")

    def _transfer(self, uav, size_bytes, start_s, *, down, category):
        """执行固定参考资源传输，按发射方向把 p×时长归入正确设备。"""
        self._require_coverage(start_s)
        cfg = self.config
        result = transmit(self.trace, size_bytes * 8, start_s, uav,
                          self.down_hz if down else self.up_hz,
                          cfg["server"]["downlink_power_w"] if down else cfg["uav"]["p_up_ref_w"],
                          cfg["channel"]["noise_psd_w_hz"])
        ledger = self.communication[category]
        ledger["bytes"] += size_bytes
        ledger["transmissions"] += 1
        ledger["csi_segments"] += result.segments
        ledger["cross_boundary_transmissions"] += int(result.segments > 1)
        self.uavs[uav]["down_s" if down else "up_s"] += result.duration_s
        if down:
            self.server["transmit_j"] += result.energy_j
        else:
            self.uavs[uav]["transmit_j"] += result.energy_j
        return result.end_s

    def batch(self, results):
        """并行模拟各机同一个批次，再把共同时间推进到最晚完成时刻。"""
        start, ends, components = self.time_s, [], []
        uav_cfg, server_cfg, profile = self.config["uav"], self.config["server"], self.profile
        for uav, result in enumerate(results):
            count = result["samples"]
            client_forward_cycles = count * profile["client_forward_flops_per_sample"] * uav_cfg["cycles_per_flop"]
            client_backward_cycles = count * profile["client_backward_flops_per_sample"] * uav_cfg["cycles_per_flop"]
            server_cycles = count * profile["server_flops_per_sample"] * server_cfg["cycles_per_flop"]
            client_s = (client_forward_cycles + client_backward_cycles) / uav_cfg["f_ref_hz"]
            self.uavs[uav]["compute_s"] += client_s
            self.uavs[uav]["compute_j"] += (uav_cfg["kappa"] *
                (client_forward_cycles + client_backward_cycles) * uav_cfg["f_ref_hz"] ** 2)
            self.server["compute_j"] += server_cfg["kappa"] * server_cycles * server_cfg["frequency_hz"] ** 2
            t = start + client_forward_cycles / uav_cfg["f_ref_hz"]
            upload_start = t
            t = self._transfer(uav, result["activation_bytes"] + result["label_bytes"], t,
                               down=False, category="activations_and_labels_up")
            upload_s = t - upload_start
            t += server_cycles / server_cfg["frequency_hz"]
            download_start = t
            t = self._transfer(uav, result["gradient_bytes"], t, down=True, category="gradients_down")
            ends.append(t + client_backward_cycles / uav_cfg["f_ref_hz"])
            components.append((client_s, server_cycles / server_cfg["frequency_hz"],
                               upload_s + t - download_start))
        self.time_s = max(ends)
        self._require_coverage(self.time_s)
        slowest = max(range(self.n), key=ends.__getitem__)
        for key, value in zip(("uav_compute_s", "server_compute_s", "communication_s"), components[slowest]):
            self.critical_path[key] += value
        for uav, end in enumerate(ends):
            self.uavs[uav]["barrier_wait_s"] += self.time_s - end
        self.blocks += 1

    def aggregate(self, *, final):
        """先并行上传前缀，再聚合；仅非最终周期下发聚合后的新前缀。"""
        size = self.profile["prefix_state_bytes"]
        start = self.time_s
        ends = [self._transfer(uav, size, self.time_s, down=False, category="prefix_state_up")
                for uav in range(self.n)]
        upload_end = max(ends)
        self.critical_path["communication_s"] += upload_end - start
        for uav, end in enumerate(ends):
            self.uavs[uav]["barrier_wait_s"] += upload_end - end
        aggregation_s = self.config["server"]["aggregation_time_s"]
        self.critical_path["aggregation_s"] += aggregation_s
        self.time_s = upload_end + aggregation_s
        self.server["aggregation_j"] += aggregation_s * self.config["server"]["aggregation_power_w"]
        for ledger in self.uavs:
            ledger["barrier_wait_s"] += aggregation_s
        self._require_coverage(self.time_s)
        if not final:
            start = self.time_s
            ends = [self._transfer(uav, size, self.time_s, down=True, category="prefix_state_down")
                    for uav in range(self.n)]
            self.time_s = max(ends)
            self.critical_path["communication_s"] += self.time_s - start
            for uav, end in enumerate(ends):
                self.uavs[uav]["barrier_wait_s"] += self.time_s - end

    def report(self):
        """返回能耗组成，不将未知固定飞行/电池/接收参数解释为零。"""
        return {
            "simulated_elapsed_s": self.time_s, "training_blocks": self.blocks,
            "critical_path": dict(self.critical_path),
            "uavs": [dict(uav_id=uav, **ledger,
                          modeled_training_j=ledger["compute_j"] + ledger["transmit_j"])
                     for uav, ledger in enumerate(self.uavs)],
            "server": self.server,
            "modeled_system_training_j": math.fsum(
                ledger["compute_j"] + ledger["transmit_j"] for ledger in self.uavs) + math.fsum(self.server.values()),
            "communication": self.communication,
            "excluded": ["起飞前统一初始化下发", "固定飞行及识别/采集/基础运行能量",
                         "尚未标定的接收、空闲、内存及数据处理功耗", "无线协议开销与重传"],
            "battery_feasibility": None, "mission_deadline_enforced": False,
        }


def cost_snapshot(physical):
    """取得独立累计成本快照，供 epoch 末计算增量，避免字典引用随训练变化。"""
    return dict(critical_path=dict(physical.critical_path),
                uav_compute_j=math.fsum(ledger["compute_j"] for ledger in physical.uavs),
                uav_transmit_j=math.fsum(ledger["transmit_j"] for ledger in physical.uavs),
                server_compute_j=physical.server["compute_j"],
                server_transmit_j=physical.server["transmit_j"],
                aggregation_j=physical.server["aggregation_j"])


def epoch_costs(before, physical):
    """返回本 epoch 的时延及系统能耗增量，包含适用的聚合块。"""
    after = cost_snapshot(physical)
    result = {key: after[key] - before[key] for key in after if key != "critical_path"}
    result["modeled_system_training_j"] = math.fsum(result.values())
    result["critical_path"] = {key: after["critical_path"][key] - value
                               for key, value in before["critical_path"].items()}
    return result


def run_diagnostic(args):
    """准备输入并完成有限个完整聚合周期；报告与表格由 main 写入本次目录。"""
    wall_start = time.perf_counter()
    config = load_config(args.config)
    period = config["training"]["aggregation_period"]
    epochs = period if args.epochs is None else args.epochs
    if epochs % period:
        raise ValueError(f"--epochs 必须为 aggregation_period={period} 的整数倍，以完整聚合结束")
    scenario_dir = (args.scenario or Path(config["output"]["scenario_dir"])).expanduser().resolve()
    manifest_path = (args.manifest or Path(config["data"]["output"]) / "data.json").expanduser().resolve()
    trace, manifest, train, validation, snapshots = load_inputs(config, scenario_dir, manifest_path)
    torch.set_num_threads(args.threads)
    device_name = args.device or config["training"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前 PyTorch 或设备不支持 CUDA；可改用 --device cpu")
    device = torch.device(device_name)
    output = create_run_directory(args.output)
    environment_archive = archive_environment(trace, scenario_dir, output)
    print(f"Output: {output}", flush=True)
    seed = config["training"]["initialization_seed"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    n = config["mission"]["num_uavs"]
    models = make_common_models(n, num_classes=len(manifest["classes"]), seed=seed,
                                checkpoint=config["training"]["initialization_checkpoint"])
    # 保存初始 CPU 权重，仅用于最终参数变化量检查，不写临时 checkpoint。
    initial = {name: parameter.detach().cpu().clone() for name, parameter in models[0].named_parameters()}
    cut = args.split or config["selection"]["initial_split"]
    print(f"Profiling {len(config['selection']['candidates'])} split points on a model copy...", flush=True)
    profiles = profile_splits(models[0], tuple(dict.fromkeys([*config["selection"]["candidates"], cut])),
                              config["data"]["image_size"], config["training"]["batch_size"])
    physical = PhysicalExecution(trace, config, profiles[cut])
    for model in models:
        model.move_split(cut, device, device)
    optimizers = [torch.optim.SGD(model.parameters(), lr=config["training"]["epoch_step_size"],
                                  momentum=0.0, weight_decay=0.0) for model in models]
    rng = random.Random(config["data"]["seed"])
    eval_indices = rng.sample(range(len(validation)), min(args.eval_samples, len(validation)))
    print(f"Device={device}; UAVs={n}; split={cut}; epochs={epochs}; validation={len(eval_indices)}", flush=True)
    evaluation_before = evaluate(models[0], validation, eval_indices, config["training"]["eval_batch_size"],
                                 config["training"]["workers"], device)
    epoch_records, aggregations = [], []
    wall_stages = {key: 0.0 for key in ("wall_client_forward_s", "wall_server_forward_backward_s",
                                       "wall_client_backward_and_update_s")}
    train_wall_start = time.perf_counter()
    batch_limit, workers = config["training"]["batch_size"], config["training"]["workers"]
    last_snapshots = None
    for epoch in range(1, epochs + 1):
        start_s = physical.time_s
        costs_before = cost_snapshot(physical)
        current = snapshots.snapshot_indices(start_s)
        last_snapshots = current
        counts = [len(indices) for indices in current]
        if len(set(counts)) != 1:
            raise ValueError("IID 等量采集设定要求 epoch 开始时各机样本数一致")
        count = counts[0]
        loaders = []
        for uav, indices in enumerate(current):
            # 只打乱快照副本；不会修改清单中固定的采集顺序。独立 Generator
            # 使画像、评价及其他 UAV 的数据读取不影响本机的 batch 顺序。
            generator = torch.Generator().manual_seed(seed + epoch * n + uav)
            loader = DataLoader(Subset(train, indices), batch_size=batch_limit, shuffle=True,
                                generator=generator, num_workers=workers, drop_last=False)
            loaders.append(iter(loader))
        losses, correct = [0.0] * n, [0] * n
        batches = math.ceil(count / batch_limit)
        epoch_wall_start = time.perf_counter()
        for batch in range(batches):
            results = []
            for uav, loader in enumerate(loaders):
                images, labels = next(loader)
                step = config["training"]["epoch_step_size"] * labels.numel() / count
                result = split_step(models[uav], optimizers[uav], images, labels, cut, step, device)
                for key in wall_stages:
                    wall_stages[key] += result[key]
                losses[uav] += result["mean_loss"] * result["samples"]
                correct[uav] += result["correct"]
                if (result["activation_bytes"] != profiles[cut]["activation_bytes_per_sample"] * result["samples"]
                        or result["gradient_bytes"] != profiles[cut]["gradient_bytes_per_sample"] * result["samples"]):
                    raise RuntimeError("实际张量载荷与切分画像不一致")
                results.append(result)
            # 此处只移动物理模拟时钟；本机依次计算分支不意味着模拟中排队。
            physical.batch(results)
            if batch == 0 or (batch + 1) % 10 == 0 or batch + 1 == batches:
                print(f"epoch {epoch}/{epochs} batch {batch + 1}/{batches}; "
                      f"D={count}/UAV; sim={physical.time_s:.3f}s", flush=True)
        record = dict(epoch=epoch, start_s=start_s, training_end_s=physical.time_s,
                      samples_per_uav=count, batches_per_uav=batches,
                      tail_batch_samples=count - (batches - 1) * batch_limit,
                      mean_losses=[loss / count for loss in losses],
                      accuracies=[value / count for value in correct],
                      wall_training_s=time.perf_counter() - epoch_wall_start)
        if epoch % period == 0:
            # 物理上传后，服务器持有各机完整状态并等权聚合。单机内存中已有
            # 全部状态；equal_fedavg 的复制不再额外记为无线传输。
            aggregation_start = physical.time_s
            physical.aggregate(final=epoch == epochs)
            equal_fedavg(models)
            metrics = evaluate(models[0], validation, eval_indices,
                               config["training"]["eval_batch_size"], workers, device)
            aggregations.append(dict(after_epoch=epoch, start_s=aggregation_start,
                                     end_s=physical.time_s, validation=metrics))
        record["end_with_aggregation_s"] = physical.time_s
        record["costs"] = epoch_costs(costs_before, physical)
        epoch_records.append(record)
    _sync(device)
    train_wall_s = time.perf_counter() - train_wall_start
    parameter_change_squared = math.fsum((parameter.detach().cpu().double() - initial[name].double())
                                         .square().sum().item() for name, parameter in models[0].named_parameters())
    if not math.isfinite(parameter_change_squared) or parameter_change_squared == 0:
        raise RuntimeError("最终模型未产生有效参数更新")
    report = {
        "schema_version": 1, "kind": "fixed_split_reference_resource_training_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": dict(run_dir=str(output), environment_dir=str(environment_archive),
                        summary_json=str(output / "summary.json"), report_md=str(output / "report.md"),
                        key_metrics_csv=str(output / "key_metrics.csv"),
                        checkpoint=str(output / "checkpoint.pt") if args.save_checkpoint else None),
        "config": config,
        "run": dict(epochs=epochs, aggregation_period=period, aggregations=len(aggregations),
                    split=cut, device=str(device), torch_version=torch.__version__, threads=args.threads,
                    task_epoch_budget_enforced=False, energy_constraints_enforced=False,
                    initialization="统一完整模型，初始前缀在起飞前下发完成，本次不计下发开销",
                    endpoint="最终完整聚合模型保留于服务器，不再下发下一周期前缀",
                    resource_policy="固定参考频率/功率，上下行分别等分带宽，不执行优化",
                    synchronization="每批及聚合块末同步，飞行和采集持续进行"),
        "inputs": dict(manifest=str(manifest_path), scenario=str(scenario_dir),
                       manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       data_fingerprint=manifest["fingerprint"], csv_sha256=trace.scenario["trace"]["csv_sha256"],
                       collection=trace.collection),
        "split_profiles": profiles, "epoch_records": epoch_records, "aggregation_records": aggregations,
        "validation_before": evaluation_before, "validation_after": aggregations[-1]["validation"],
        "validation_subset_sha256": hashlib.sha256(json.dumps(eval_indices).encode()).hexdigest(),
        "parameter_change_l2": math.sqrt(parameter_change_squared),
        "data_participation": dict(unique_trained_samples_per_uav=len(last_snapshots[0]),
                                   sample_visits_per_uav=sum(record["samples_per_uav"] for record in epoch_records),
                                   available_samples_per_uav_at_end=trace.sample_count_at(physical.time_s),
                                   minimum_passes_required=None),
        "physical_execution": physical.report(),
        "wall_timing": dict(training_loop_including_aggregation_evaluation_s=train_wall_s,
                            summed_branch_compute_stages_s=wall_stages,
                            total_before_report_write_s=time.perf_counter() - wall_start),
        "interpretation": ["验证结果仅用于观察一次短诊断，不构成总体损失的收敛证明",
                           "能耗来自 FLOP/周期/频率/电容参数，不是本机或真实 UAV 电能测量",
                           "模拟物理时延不含本机加载图片、画像和评价耗时；CSV 覆盖不足为输入数据问题",
                           "晚到且尚未进入 epoch 快照的样本不补训，不要求逐图训练次数"],
    }
    if args.save_checkpoint:
        torch.save({"model_state_dict": {name: value.detach().cpu() for name, value in models[0].state_dict().items()},
                    "epochs": epochs, "split": cut, "data_fingerprint": manifest["fingerprint"]}, output / "checkpoint.pt")
    return report


def main(argv=None):
    """运行诊断，将 JSON、关键指标表和中文报告写入本次独立结果目录。"""
    args = parse_args(argv)
    try:
        if args.mode == "dynamic":
            from dynamic_training import run_dynamic
            report = run_dynamic(args)
            if report.get("prepared_only"):
                print(f"Preparation complete: {report['outputs']['run_dir']}", flush=True)
                return
        else:
            if args.prepare_only or args.prepared_run or args.optimization_results:
                raise ValueError("prepare 参数仅适用于动态模式")
            report = run_diagnostic(args)
    except TraceHorizonError as error:
        raise SystemExit(f"诊断未完成：{error} 这表示 CSV 覆盖不足，不是违反训练时延约束。") from error
    report.setdefault("source_git", source_revision())
    output = Path(report["outputs"]["run_dir"])
    path = output / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tables = write_run_report(report, output)
    physical = report["physical_execution"]
    print(f"{report['run'].get('status', 'completed')}. Simulated={physical['simulated_elapsed_s']:.3f}s; "
          f"modeled training energy={physical['modeled_system_training_j']:.3f}J; "
          f"parameter change L2={report['parameter_change_l2']:.6g}", flush=True)
    print(f"Report: {path}", flush=True)
    print(f"Key metrics: {tables['key_metrics_csv']}", flush=True)
    print(f"Readable report: {tables['report_md']}", flush=True)
    if report.get("failure"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
