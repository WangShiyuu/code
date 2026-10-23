"""将一次 UAV-SFL 训练的 summary.json 转为便于阅读和核对的中文结果表。

运行方式（在本模块所在目录）：
    python -X utf8 -B report_results.py
    python -X utf8 -B report_results.py --report output/output_时间戳/summary.json

默认按目录名字选择 output/output_*/summary.json 中最新的一份；若没有新版
目录，则回退到旧的 output/diagnostic/summary.json。两个结果文件直接写在
summary.json 所在目录，重复执行会更新表格，不运行训练、不新建训练目录。
训练入口也可以直接调用 write_run_report(report, output_dir)，传入内存报告。

本模块仅依赖 Python 标准库，不导入 torch、训练入口或当前 config.py，确保
旧结果始终使用 summary 内保存的运行参数。CSV 使用 UTF-8 BOM，方便 Excel
直接打开；Markdown 保留统计口径、单位和缺失说明。所有时延为秒、能量为 J、
频率/带宽为 Hz、功率为 W、载荷为字节 B。频率 f 与计算功率 kappa*f^3 分列。

系统模拟时延采用报告记录的并行关键路径，不能累加各机的分支工作时长。
旧报告若未保存 critical_path，显示“未记录”，不会从能量或并行总 FLOP 数
反推、拼凑时延分解。未标定的接收/空闲等能量也不会按零补入系统总能量。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
MISSING = "未记录"


@dataclass(frozen=True)
class _Metric:
    """保存同一条 CSV/Markdown 指标，单位和口径始终随数值一起输出。"""

    section: str
    key: str
    label: str
    entity: str
    epoch: Any
    value: Any
    unit: str
    scope: str


def _mapping(value: Any) -> Mapping[str, Any]:
    """只接受字典式对象；缺失分组返回空映射，使旧版报告可继续读取。"""
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> bool:
    """判断是否为有限实数；拒绝布尔值及 NaN/无穷大，防止无效派生指标。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _product(*values: Any) -> Any:
    """全部因子有效时计算乘积；任一输入缺失时返回 None，不默认补零。"""
    return math.prod(values) if all(_number(value) for value in values) else None


def _sum(values: list[Any]) -> Any:
    """对完整的有效数列求和；空列表或缺失分项保留为未记录。"""
    if not values or not all(_number(value) for value in values):
        return None
    return sum(values) if all(isinstance(value, int) for value in values) else math.fsum(values)


def _divide(numerator: Any, denominator: Any) -> Any:
    """仅对有效非零分母执行除法；未知资源不能被解释为零时延或零功率。"""
    if _number(numerator) and _number(denominator) and denominator != 0:
        return numerator / denominator
    return None


def _per_uav(value: Any, uav: int) -> Any:
    """读取每机字段；当前等量采集的标量对各机共用，也兼容逐机列表。"""
    if isinstance(value, (list, tuple)):
        return value[uav] if uav < len(value) else None
    return value


def _text(value: Any, *, compact: bool = False) -> str:
    """统一显示缺失和结构值；CSV 保留浮点完整精度，Markdown 用12位有效数。"""
    if value is None:
        return MISSING
    if isinstance(value, float):
        return format(value, ".12g") if compact else str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _cell(value: Any) -> str:
    """转义 Markdown 表格单元格，避免路径或口径中的竖线/换行破坏表格。"""
    return _text(value, compact=True).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _metrics(report: Mapping[str, Any]) -> list[_Metric]:
    """整理训练、资源、样本、载荷和物理计量为共享的扁平指标集合。

    优先读取实际执行台账；资源公式只使用 summary 内配置。显式标注由固定
    资源/样本数推导的 FLOP、周期和功率，不把这些量称为实测电能。各机时延
    的和只表示设备工作量，关键路径分解单独使用 physical_execution 记录。
    """
    cfg = _mapping(report.get("config"))
    run = _mapping(report.get("run"))
    physical = _mapping(report.get("physical_execution"))
    training = _mapping(cfg.get("training"))
    uav_cfg, server_cfg = _mapping(cfg.get("uav")), _mapping(cfg.get("server"))
    radio, mission = _mapping(cfg.get("radio")), _mapping(cfg.get("mission"))
    participation = _mapping(report.get("data_participation"))
    collection = _mapping(_mapping(report.get("inputs")).get("collection"))
    profile = _mapping(_mapping(report.get("split_profiles")).get(run.get("split")))
    epochs = report.get("epoch_records", [])
    epochs = epochs if isinstance(epochs, list) else []
    ledgers = physical.get("uavs", [])
    ledgers = ledgers if isinstance(ledgers, list) else []
    n = mission.get("num_uavs", len(ledgers) or None)
    n_int = int(n) if _number(n) and n >= 1 and int(n) == n else 0
    rows: list[_Metric] = []

    def add(section: str, key: str, label: str, value: Any, unit: str = "",
            scope: str = "summary 记录", entity: str = "系统", epoch: Any = "") -> None:
        """追加一条指标；None 原样保留，在写出阶段明确显示为未记录。"""
        rows.append(_Metric(section, key, label, entity, epoch, value, unit, scope))

    section = "运行概况"
    for key, label, value, unit in (
        ("created_at_utc", "报告生成时间（UTC）", report.get("created_at_utc"), ""),
        ("epochs", "实际完成训练 epoch", run.get("epochs"), "epoch"),
        ("configured_total_epochs", "配置的理论/任务训练轮数", training.get("total_epochs"), "epoch"),
        ("aggregation_period", "聚合间隔", run.get("aggregation_period"), "epoch/次"),
        ("aggregations", "实际聚合次数", run.get("aggregations"), "次"),
        ("num_uavs", "无人机数量", n, "架"),
        ("split", "本次固定切点", run.get("split"), ""),
        ("batch_size", "批大小上限", training.get("batch_size"), "样本/批"),
        ("device", "本机训练执行设备", run.get("device"), ""),
        ("training_blocks", "模拟批同步块数", physical.get("training_blocks"), "块"),
    ):
        add(section, key, label, value, unit)
    add(section, "resource_policy", "实际资源策略", run.get("resource_policy"))
    add(section, "synchronization", "同步方式", run.get("synchronization"))
    add(section, "endpoint", "最终模型位置及下发约定", run.get("endpoint"))

    section = "实际资源与计算功率"
    up_hz, down_hz = _divide(radio.get("total_up_hz"), n), _divide(radio.get("total_down_hz"), n)
    f_uav, f_server = uav_cfg.get("f_ref_hz"), server_cfg.get("frequency_hz")
    p_compute_uav = _product(uav_cfg.get("kappa"), _product(f_uav, f_uav, f_uav))
    p_compute_server = _product(server_cfg.get("kappa"), _product(f_server, f_server, f_server))
    add(section, "total_up_hz", "上行总带宽", radio.get("total_up_hz"), "Hz", "每个方向独立的带宽预算")
    add(section, "total_down_hz", "下行总带宽", radio.get("total_down_hz"), "Hz", "每个方向独立的带宽预算")
    for uav in range(n_int):
        entity = f"UAV {uav}"
        for key, label, value, unit, scope in (
            ("frequency_hz", "实际计算频率 f", f_uav, "Hz", "固定参考资源：config.uav.f_ref_hz"),
            ("kappa", "动态计算能耗系数 kappa", uav_cfg.get("kappa"), "J/(cycle·Hz²)", "硬件能耗参数，不是理论收敛系数"),
            ("cycles_per_flop", "FLOP 到周期的换算系数", uav_cfg.get("cycles_per_flop"), "cycle/FLOP", "配置的等效计算周期"),
            ("compute_power_w", "计算有效功率", p_compute_uav, "W", "由 kappa*f³ 推导，仅在计算期间使用"),
            ("up_power_w", "上行发射功率", uav_cfg.get("p_up_ref_w"), "W", "每机上行链路；与计算功率分别计量"),
            ("up_bandwidth_hz", "实际上行带宽", up_hz, "Hz", "上行总带宽/N，未用份额不重分配"),
            ("down_bandwidth_hz", "实际下行带宽", down_hz, "Hz", "下行总带宽/N，未用份额不重分配"),
        ):
            add(section, key, label, value, unit, scope, entity)
    branches = server_cfg.get("parallel_branches")
    add(section, "server_branches", "服务器独立分支数", n if branches is None else branches,
        "分支", "parallel_branches=None 时使用 N 个独立分支", "服务器")
    for key, label, value, unit, scope in (
        ("frequency_hz", "每分支计算频率 f", f_server, "Hz", "各分支独立使用该频率，不将它按 N 均分"),
        ("kappa", "每分支动态计算能耗系数 kappa", server_cfg.get("kappa"), "J/(cycle·Hz²)", "服务器硬件参数"),
        ("cycles_per_flop", "FLOP 到周期的换算系数", server_cfg.get("cycles_per_flop"), "cycle/FLOP", "服务器后缀等效周期"),
        ("compute_power_w", "每分支计算有效功率", p_compute_server, "W", "由 kappa*f³ 推导，仅在该分支计算期间使用"),
        ("active_compute_power_w", "N 个活跃分支同时计算功率", _product(p_compute_server, n), "W", "N*kappa*f³；不含额外未使用的分支"),
        ("downlink_power_w", "每条下行链路发射功率", server_cfg.get("downlink_power_w"), "W/link", "每链路发射功率，发射时长分别累计"),
        ("aggregation_time_s", "每次聚合固定时延", server_cfg.get("aggregation_time_s"), "s/次", "聚合开销与后缀计算分别记录"),
        ("aggregation_power_w", "聚合计算功率", server_cfg.get("aggregation_power_w"), "W", "只用于固定聚合阶段"),
    ):
        add(section, key, label, value, unit, scope, "服务器")

    section = "逐 epoch 快照与切点"
    batch_limit = training.get("batch_size")
    for record in epochs:
        record = _mapping(record)
        epoch = record.get("epoch")
        add(section, "snapshot_time_s", "冻结样本快照时刻", record.get("start_s"), "s", "共同模拟物理时间", epoch=epoch)
        add(section, "training_end_s", "本 epoch 训练结束时刻", record.get("training_end_s"), "s", "不含本 epoch 之后的聚合", epoch=epoch)
        add(section, "end_with_aggregation_s", "包含聚合的结束时刻", record.get("end_with_aggregation_s"), "s", "完成该 epoch 及适用的聚合后", epoch=epoch)
        for uav in range(n_int):
            entity = f"UAV {uav}"
            count = _per_uav(record.get("samples_per_uav"), uav)
            tail = _per_uav(record.get("tail_batch_samples"), uav)
            batches = _per_uav(record.get("batches_per_uav"), uav)
            cut = _per_uav(record.get("splits_per_uav", run.get("split")), uav)
            add(section, "split", "实际切点", cut, "", "固定切分诊断，每机同一切点", entity, epoch)
            add(section, "snapshot_samples", "冻结快照样本数", count, "样本", "本 epoch 完整遍历快照；包含保留的旧样本", entity, epoch)
            add(section, "batches", "实际批次数", batches, "批", "含最后一个批次，drop_last=False", entity, epoch)
            add(section, "tail_samples", "最后一批样本数", tail, "样本", "可能等于批大小上限，此时没有不满批", entity, epoch)
            full_batches = None
            if _number(count) and _number(batch_limit) and batch_limit > 0:
                full_batches = int(count // batch_limit)
            add(section, "full_batches", "满批数", full_batches, "批", "floor(快照样本数/批大小上限)", entity, epoch)

    section = "逐 epoch 时延与能量"
    for record in epochs:
        record = _mapping(record)
        epoch = record.get("epoch")
        costs = _mapping(record.get("costs"))
        critical = _mapping(costs.get("critical_path"))
        for key, label in (("uav_compute_s", "关键路径 UAV 前缀计算"),
                           ("server_compute_s", "关键路径服务器后缀计算"),
                           ("communication_s", "关键路径通信"), ("aggregation_s", "关键路径聚合")):
            add(section, key, label, critical.get(key), "s", "本 epoch 执行台账增量，包含适用的聚合；旧版未保存则不推算", epoch=epoch)
        for key, label in (("uav_compute_j", "所有 UAV 前缀计算能量"),
                           ("uav_transmit_j", "所有 UAV 上行发射能量"),
                           ("server_compute_j", "服务器后缀计算能量"),
                           ("server_transmit_j", "服务器下行发射能量"),
                           ("aggregation_j", "服务器聚合计算能量"),
                           ("modeled_system_training_j", "系统已建模训练总能量")):
            add(section, key, label, costs.get(key), "J", "本 epoch 前后累计能量之差，包含适用的聚合；各并行分支能量相加", epoch=epoch)

    section = "样本参与统计"
    for key, label, unit, scope in (
        ("max_samples_per_uav", "每机采集容量", "样本", "固定、互不重复的每机采集序列容量"),
        ("initial_samples_per_uav", "每机初始缓存", "样本", "已计入采集容量"),
        ("rate_samples_per_s", "每机持续采集速率", "样本/s", "训练、通信和等待期间均继续采集"),
    ):
        add(section, key, label, collection.get(key), unit, scope, "每机")
    for key, label, unit, scope in (
        ("unique_trained_samples_per_uav", "参与训练的不同样本", "样本", "累计前缀保留，取最后已训练快照；不按 epoch 重复计数"),
        ("sample_visits_per_uav", "训练样本访问次数", "样本次", "所有 epoch 快照样本数之和；旧样本再次训练再次计数"),
        ("available_samples_per_uav_at_end", "结束时已采集样本", "样本", "包括晚到且尚未进入 epoch 快照的样本"),
    ):
        value = participation.get(key)
        add(section, key, label, value, unit, scope, "每机")
        total = _sum(list(value)) if isinstance(value, (list, tuple)) else _product(value, n)
        add(section, "system_" + key, "系统" + label, total, unit,
            scope + "；各机使用不同训练索引，按 N 架求和", "系统")

    section = "切分画像与通信载荷"
    for key, label, unit, scope in (
        ("client_flops_per_sample", "每样本前缀计算量", "FLOP/样本", "包括前向与反向；按画像约定计数"),
        ("server_flops_per_sample", "每样本后缀计算量", "FLOP/样本", "包括前向与反向；每个独立服务器分支"),
        ("activation_bytes_per_sample", "每样本激活", "B/样本", "上行激活，不含标签"),
        ("label_bytes_per_sample", "每样本标签", "B/样本", "随激活上行"),
        ("gradient_bytes_per_sample", "每样本边界梯度", "B/样本", "服务器向 UAV 下行"),
        ("prefix_state_bytes", "一次单机前缀状态", "B/次", "含参数及 BN buffers，聚合时上传"),
    ):
        add(section, key, label, profile.get(key), unit, scope, str(run.get("split", "画像")))
    add(section, "flop_count_scope", "FLOP 计数范围", profile.get("flop_count_scope"), "", "沿用离线画像记录")
    up_per_sample = _sum([profile.get("activation_bytes_per_sample"), profile.get("label_bytes_per_sample")])
    for payload_name, per_sample, label in (
        ("activations_and_labels_up", up_per_sample, "激活与标签上行"),
        ("gradients_down", profile.get("gradient_bytes_per_sample"), "梯度下行"),
    ):
        add(section, payload_name + "_full_batch_bytes", "满批" + label, _product(per_sample, batch_limit), "B/批", "每机、按实际样本数线性缩放", "每机")
        for uav in range(n_int):
            add(section, payload_name + "_sample_visits_bytes", "训练批累计" + label,
                _product(per_sample, _per_uav(participation.get("sample_visits_per_uav"), uav)), "B",
                "每样本载荷×本机样本访问次数，不含聚合状态", f"UAV {uav}")
        for record in epochs:
            record = _mapping(record)
            for uav in range(n_int):
                add(section, payload_name + "_tail_bytes", "最后一批" + label,
                    _product(per_sample, _per_uav(record.get("tail_batch_samples"), uav)), "B/批",
                    "每机、使用尾批实际样本数", f"UAV {uav}", record.get("epoch"))
                add(section, payload_name + "_epoch_bytes", "本 epoch 累计" + label,
                    _product(per_sample, _per_uav(record.get("samples_per_uav"), uav)), "B",
                    "每样本载荷×本 epoch 冻结快照样本数", f"UAV {uav}", record.get("epoch"))
    names = {"activations_and_labels_up": "激活与标签上行", "gradients_down": "边界梯度下行",
             "prefix_state_up": "聚合前缀状态上传", "prefix_state_down": "聚合前缀状态下发"}
    communication = _mapping(physical.get("communication"))
    for category, label in names.items():
        ledger = _mapping(communication.get(category))
        for key, suffix, unit in (("bytes", "载荷总量", "B"), ("transmissions", "传输次数", "次"),
                                  ("csi_segments", "累计 CSI 分段数", "段"),
                                  ("cross_boundary_transmissions", "跨 CSI 边界传输数", "次")):
            add(section, category + "_" + key, label + suffix, ledger.get(key), unit,
                "所有 UAV、所有批/聚合累加；纯张量载荷，不含协议开销", "系统")
    add(section, "total_payload_bytes", "通信载荷总量", _sum([_mapping(communication.get(key)).get("bytes") for key in names]),
        "B", "四类实际通信载荷求和，上下行分别发送均计入")

    section = "每机实际计量"
    for ledger in ledgers:
        ledger = _mapping(ledger)
        entity = f"UAV {ledger.get('uav_id', '?')}"
        for key, label, unit, scope in (
            ("compute_s", "前缀计算工作时长", "s", "本机各批前向/反向时间累加，不是系统墙钟时长"),
            ("up_s", "上行发送工作时长", "s", "本机激活/标签及前缀状态上传时长累加"),
            ("down_s", "下行传输工作时长", "s", "本机链路接收梯度及适用的前缀下发时长；接收能量未标定"),
            ("barrier_wait_s", "同步等待时长", "s", "批/通信块末屏障等待，含服务器聚合等待；不含后缀计算期间的协议依赖"),
            ("compute_j", "前缀计算能量", "J", "kappa*C*f² 的模型估计；C 为累计等效周期"),
            ("transmit_j", "上行发射能量", "J", "只计本机发射功率×上行发送时长"),
            ("modeled_training_j", "已建模训练总能量", "J", "前缀计算+上行发射；未含未知的接收/空闲等能量"),
        ):
            add(section, key, label, ledger.get(key), unit, scope, entity)
        add(section, "compute_energy_over_time_w", "计算能量/计算时长", _divide(ledger.get("compute_j"), ledger.get("compute_s")),
            "W", "交叉核对 kappa*f³；不是除以系统模拟时长的平均功率", entity)

    section = "服务器与系统总量"
    server = _mapping(physical.get("server"))
    for key, label, scope in (("compute_j", "后缀计算能量", "所有独立分支能量相加"),
                               ("transmit_j", "下行发射能量", "所有下行链路 p_down×发送时长相加"),
                               ("aggregation_j", "聚合计算能量", "聚合次数×固定聚合时延×聚合功率")):
        add(section, "server_" + key, label, server.get(key), "J", scope, "服务器")
    add(section, "server_total_j", "服务器已建模总能量", _sum([server.get(key) for key in ("compute_j", "transmit_j", "aggregation_j")]),
        "J", "后缀计算+下行发射+聚合", "服务器")
    for key, label, unit, scope in (
        ("compute_j", "所有 UAV 前缀计算能量", "J", "各机计算能量之和"),
        ("transmit_j", "所有 UAV 上行发射能量", "J", "各机上行发射能量之和"),
        ("compute_s", "所有 UAV 计算工作量时长", "设备·s", "各机计算时长之和；不能当作并行系统时延"),
    ):
        add(section, "all_uavs_" + key, label, _sum([_mapping(ledger).get(key) for ledger in ledgers]), unit, scope)
    add(section, "simulated_elapsed_s", "系统模拟总时延", physical.get("simulated_elapsed_s"), "s", "并行独立分支、每批同步；取最慢完成时间推进物理时钟")
    add(section, "modeled_system_training_j", "系统已建模训练总能量", physical.get("modeled_system_training_j"), "J", "各 UAV 计算/发射+服务器计算/下行发射/聚合")
    for key, label, value in (
        ("all_compute_j", "系统前后缀计算能量", _sum([_sum([_mapping(ledger).get("compute_j") for ledger in ledgers]), server.get("compute_j")])),
        ("all_transmit_j", "系统通信发射能量", _sum([_sum([_mapping(ledger).get("transmit_j") for ledger in ledgers]), server.get("transmit_j")])),
    ):
        add(section, key, label, value, "J", "按设备台账直接求和，聚合能量单列")

    section = "并行关键路径时延分解"
    critical = _mapping(physical.get("critical_path"))
    for key, label in (("uav_compute_s", "关键路径 UAV 前缀计算"),
                        ("server_compute_s", "关键路径服务器后缀计算"),
                        ("communication_s", "关键路径通信"), ("aggregation_s", "关键路径聚合")):
        add(section, key, label, critical.get(key), "s", "由模拟执行时记录；各同步块仅取关键分支，不累加并行分支时长")
    critical_sum = _sum([critical.get(key) for key in ("uav_compute_s", "server_compute_s", "communication_s", "aggregation_s")])
    add(section, "critical_path_sum_s", "关键路径四项之和", critical_sum, "s", "应等于系统模拟总时延；旧版无 critical_path 时不推算")
    elapsed = physical.get("simulated_elapsed_s")
    difference = critical_sum - elapsed if _number(critical_sum) and _number(elapsed) else None
    add(section, "critical_path_minus_elapsed_s", "关键路径合计减系统总时延", difference, "s", "允许浮点舍入级差异；缺少任何分项则显示未记录")

    section = "本机墙钟与训练验证"
    wall = _mapping(report.get("wall_timing"))
    for key, label in (("training_loop_including_aggregation_evaluation_s", "本机训练循环墙钟（含聚合评价）"),
                        ("total_before_report_write_s", "本机执行墙钟（报告写出前）")):
        add(section, key, label, wall.get(key), "s", "本机真实执行计时，与模拟物理时延分开")
    add(section, "parameter_change_l2", "模型参数变化 L2", report.get("parameter_change_l2"), "", "最终聚合模型与统一初始化参数之差")
    for stage in ("before", "after"):
        values = _mapping(report.get("validation_" + stage))
        for key, label in (("mean_loss", "验证平均损失"), ("accuracy", "验证准确率"), ("samples", "验证样本数")):
            add(section, "validation_" + stage + "_" + key, ("训练前" if stage == "before" else "训练后") + label,
                values.get(key), "样本" if key == "samples" else "", "同一固定验证子集；准确率为 0 到 1 的比例")
    return rows


def _markdown(report: Mapping[str, Any], rows: list[_Metric]) -> str:
    """将全量指标压缩为面向阅读的表格；同构资源只列一次，每台设备占一行。

    CSV 保留所有细项和完整精度。Markdown 把同口径的指标横向合并，不重复
    五份相同硬件参数；epoch 内同时列出快照、切点、关键路径与能量增量。
    """
    lines = ["# UAV-SFL 训练结果报告", "",
             "参数和台账来自本次 summary；能量为模型估计。时延为秒，能量为 J，字节为 B；未记录的字段不补零。", ""]

    def metric(key: str, entity: str | None = None, epoch: Any = None) -> Any:
        """按键、对象和轮次读取已经整理的指标；不存在时返回 None。"""
        for row in rows:
            if row.key == key and (entity is None or row.entity == entity) and (epoch is None or row.epoch == epoch):
                return row.value
        return None

    def table(title: str, headers: list[str], values: list[list[Any]]) -> None:
        """追加简洁的 Markdown 表格，单元格沿用统一缺失及精度规则。"""
        lines.extend(["## " + title, "", "| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"])
        lines.extend("| " + " | ".join(_cell(value) for value in record) + " |" for record in values)
        lines.append("")

    def pair(first: Any, second: Any) -> str:
        """格式化同一表格列中的两个有序数值，顺序由列名明确给出。"""
        return _text(first, compact=True) + " / " + _text(second, compact=True)

    run = _mapping(report.get("run"))
    physical = _mapping(report.get("physical_execution"))
    cfg = _mapping(report.get("config"))
    server_cfg, uav_cfg = _mapping(cfg.get("server")), _mapping(cfg.get("uav"))
    epoch_records = report.get("epoch_records", [])
    epoch_records = epoch_records if isinstance(epoch_records, list) else []
    table("运行概况", ["项目", "结果"], [
        ["实际 epoch / 聚合间隔 / 聚合次数", " / ".join(_text(run.get(key)) for key in ("epochs", "aggregation_period", "aggregations"))],
        ["无人机数量 / 固定切点 / 批大小上限", " / ".join(_text(metric(key)) for key in ("num_uavs", "split", "batch_size"))],
        ["本机设备 / 报告时间（UTC）", pair(run.get("device"), report.get("created_at_utc"))],
        ["系统模拟总时延（s）", metric("simulated_elapsed_s")],
        ["系统已建模训练总能量（J）", metric("modeled_system_training_j", "系统", "")],
        ["资源策略", run.get("resource_policy")],
        ["最终模型约定", run.get("endpoint")],
    ])

    epoch_values = []
    for record in epoch_records:
        record = _mapping(record)
        costs = _mapping(record.get("costs"))
        critical = _mapping(costs.get("critical_path"))
        start, end = record.get("start_s"), record.get("end_with_aggregation_s")
        duration = end - start if _number(start) and _number(end) else None
        epoch_values.append([
            record.get("epoch"), record.get("splits_per_uav", run.get("split")),
            " / ".join(_text(record.get(key), compact=True) for key in ("samples_per_uav", "batches_per_uav", "tail_batch_samples")),
            record.get("start_s"), pair(critical.get("uav_compute_s"), critical.get("server_compute_s")),
            pair(critical.get("communication_s"), critical.get("aggregation_s")), duration,
            _sum([costs.get("uav_compute_j"), costs.get("server_compute_j")]),
            _sum([costs.get("uav_transmit_j"), costs.get("server_transmit_j")]),
            costs.get("aggregation_j"), costs.get("modeled_system_training_j"),
        ])
    table("逐 epoch 快照、时延与能量", ["epoch", "每机切点", "每机样本/批数/尾批", "快照时刻 s", "前缀/后缀 s",
          "通信/聚合 s", "本轮总时延 s", "计算 J", "通信 J", "聚合 J", "总能量 J"], epoch_values)
    lines.extend(["样本数是轮初冻结的累计快照；旧数据保留。相同标量适用于所有 UAV，逐机列表按 UAV 编号排列。时延分项取同步块关键路径；能量对所有设备求和。", ""])

    table("样本参与", ["口径", "每机", "系统"], [
        ["参与训练的不同样本", metric("unique_trained_samples_per_uav"), metric("system_unique_trained_samples_per_uav")],
        ["样本访问次数（含旧样本重复训练）", metric("sample_visits_per_uav"), metric("system_sample_visits_per_uav")],
        ["结束时已采集样本（含未训练晚到样本）", metric("available_samples_per_uav_at_end"), metric("system_available_samples_per_uav_at_end")],
    ])

    table("实际资源（同构参数只列一次）", ["对象", "计算频率 f（Hz）", "计算功率（W）", "上行发射（W）", "下行发射（W/link）",
          "每链路上/下行带宽（Hz）", "kappa", "cycle/FLOP"], [
        [f"每架 UAV（共 {_text(metric('num_uavs'))} 架）", metric("frequency_hz", "UAV 0"), metric("compute_power_w", "UAV 0"),
         metric("up_power_w", "UAV 0"), "—", pair(metric("up_bandwidth_hz", "UAV 0"), metric("down_bandwidth_hz", "UAV 0")),
         uav_cfg.get("kappa"), uav_cfg.get("cycles_per_flop")],
        [f"服务器每分支（共 {_text(metric('server_branches'))} 个）", metric("frequency_hz", "服务器"), metric("compute_power_w", "服务器"),
         "—", metric("downlink_power_w", "服务器"), "同对应 UAV 链路", server_cfg.get("kappa"), server_cfg.get("cycles_per_flop")],
    ])
    lines.extend([f"计算有效功率为 `kappa*f³`，与发射功率分别计量。上下行各自总带宽为 {_cell(metric('total_up_hz'))} / {_cell(metric('total_down_hz'))} Hz，分别均分给 N 条链路；每个服务器分支独立使用所列频率。",
                  f"聚合配置：每次 {_cell(metric('aggregation_time_s'))} s，计算功率 {_cell(metric('aggregation_power_w'))} W。", ""])

    ledgers = physical.get("uavs", [])
    ledgers = ledgers if isinstance(ledgers, list) else []
    table("每机累计工作与能量", ["UAV", "计算 s", "上行 s", "下行 s", "屏障等待 s", "计算 J", "上行发射 J", "合计 J"], [
        [ledger.get("uav_id"), *[ledger.get(key) for key in ("compute_s", "up_s", "down_s", "barrier_wait_s", "compute_j", "transmit_j", "modeled_training_j")]]
        for raw_ledger in ledgers for ledger in [_mapping(raw_ledger)]
    ])
    lines.extend(["设备累计工作时长会重叠，不能相加作为系统时延；屏障等待含聚合等待，不含后缀计算的协议依赖时间。下行时长对应链路接收，UAV 接收能量未标定。", ""])

    table("系统关键路径与能量分解", ["分项", "关键路径时延（s）", "能量（J）"], [
        ["UAV 前缀计算", metric("uav_compute_s", "系统", ""), metric("all_uavs_compute_j")],
        ["服务器后缀计算", metric("server_compute_s", "系统", ""), metric("server_compute_j", "服务器")],
        ["通信发射", metric("communication_s", "系统", ""), metric("all_transmit_j")],
        ["其中：所有 UAV 上行", "—", metric("all_uavs_transmit_j")],
        ["其中：服务器下行", "—", metric("server_transmit_j", "服务器")],
        ["服务器聚合", metric("aggregation_s", "系统", ""), metric("server_aggregation_j", "服务器")],
        ["服务器能量小计", "—", metric("server_total_j", "服务器")],
        ["系统总计", metric("simulated_elapsed_s"), metric("modeled_system_training_j", "系统", "")],
    ])
    lines.extend([f"关键路径四项合计 {_cell(metric('critical_path_sum_s'))} s，与系统总时延之差为 {_cell(metric('critical_path_minus_elapsed_s'))} s。旧报告未记录分解时显示“未记录”。", ""])

    table("通信载荷", ["方向/类别", "每样本 B", "满批 B", "累计 B", "传输次数"], [
        ["激活与标签上行", _sum([metric("activation_bytes_per_sample"), metric("label_bytes_per_sample")]),
         metric("activations_and_labels_up_full_batch_bytes"), metric("activations_and_labels_up_bytes"), metric("activations_and_labels_up_transmissions")],
        ["梯度下行", metric("gradient_bytes_per_sample"), metric("gradients_down_full_batch_bytes"), metric("gradients_down_bytes"), metric("gradients_down_transmissions")],
        ["前缀状态上传", "—", "—", metric("prefix_state_up_bytes"), metric("prefix_state_up_transmissions")],
        ["前缀状态下发", "—", "—", metric("prefix_state_down_bytes"), metric("prefix_state_down_transmissions")],
        ["总载荷", "—", "—", metric("total_payload_bytes"), "—"],
    ])
    tails = []
    for record in epoch_records:
        record = _mapping(record)
        tail = record.get("tail_batch_samples")
        if _number(tail):
            up = _sum([metric("activation_bytes_per_sample"), metric("label_bytes_per_sample")])
            tails.append(f"epoch {_cell(record.get('epoch'))}：{_cell(_product(up, tail))} / {_cell(_product(metric('gradient_bytes_per_sample'), tail))} B")
    lines.extend([f"一次单机前缀状态 {_cell(metric('prefix_state_bytes'))} B；累计载荷对所有 UAV 求和。尾批每机上行/下行载荷：" + ("；".join(tails) if tails else MISSING) + "。", ""])

    table("验证与本机计时", ["阶段", "验证样本", "平均损失", "准确率（0—1）"], [
        [label, metric("validation_" + stage + "_samples"), metric("validation_" + stage + "_mean_loss"), metric("validation_" + stage + "_accuracy")]
        for stage, label in (("before", "训练前"), ("after", "训练后"))
    ])
    lines.extend([f"本机训练循环墙钟（含聚合评价）：{_cell(metric('training_loop_including_aggregation_evaluation_s'))} s；报告写出前总墙钟：{_cell(metric('total_before_report_write_s'))} s；参数变化 L2：{_cell(metric('parameter_change_l2'))}。", "",
                  "## 统计口径", "",
                  "- 能量：`E=kappa*C*f²`；`C=FLOP×cycle/FLOP`。各设备能量可求和，独立分支时延按同步关键路径计量。",
                  "- 表中合计只含已建模的计算、发射和聚合能量；不是整机实测电能或电池可行性结论。",
                  "- 数据/训练未要求所有晚到样本补训；短诊断验证结果不构成总体损失收敛证明。",
                  "- key_metrics.csv 保留完整逐机、逐轮、画像和通信明细，数值使用完整浮点精度。"])
    excluded = physical.get("excluded")
    if isinstance(excluded, list) and excluded:
        lines.append("- 未计入：" + "；".join(_text(entry) for entry in excluded) + "。")
    return "\n".join(lines) + "\n"


def write_run_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """写出 key_metrics.csv 和 report.md，返回两个绝对 Path 的字典。

    report 为已完成训练的 summary 字典；不重新训练或加载图片。output_dir
    是本次训练目录，调用方通常传入 summary.json 的父目录。若该目录不存在
    则创建；已经存在的同名表格会更新，summary/checkpoint 本身保持不变。
    返回键固定为 key_metrics_csv、report_md，方便训练入口打印产物路径。
    """
    if not isinstance(report, Mapping):
        raise ValueError("训练报告必须为 JSON 对象/映射")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = _metrics(report)
    csv_path, markdown_path = destination / "key_metrics.csv", destination / "report.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["分组", "指标键", "指标", "对象", "epoch", "数值", "单位", "统计口径"])
        for row in rows:
            writer.writerow([row.section, row.key, row.label, row.entity, row.epoch, _text(row.value), row.unit, row.scope])

    markdown_path.write_text(_markdown(report, rows), encoding="utf-8")
    return {"key_metrics_csv": csv_path, "report_md": markdown_path}


def find_latest_report(root: str | Path = ROOT) -> Path:
    """按时间戳目录名字查找最新版 summary，失败时回退旧 diagnostic 路径。

    只搜索 ROOT/output/output_*/summary.json，不使用文件修改时间，避免旧
    结果被复制后误当作最新一次运行；新版候选不存在才尝试旧版目录。
    """
    output = Path(root).expanduser().resolve() / "output"
    candidates = sorted(output.glob("output_*/summary.json"), key=lambda path: path.parent.name, reverse=True)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    legacy = output / "diagnostic" / "summary.json"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"未找到 {output}/output_*/summary.json 或 {legacy}；可用 --report 指定已有报告")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    """处理命令行并就地生成表格；输入不存在/JSON无效时明确退出。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="已有 summary.json 路径；默认查找本模块 output 内最新训练")
    args = parser.parse_args(argv)
    try:
        source = args.report.expanduser().resolve() if args.report is not None else find_latest_report()
        report = json.loads(source.read_text(encoding="utf-8-sig"))
        paths = write_run_report(report, source.parent)
    except (OSError, ValueError) as error:
        parser.exit(1, f"生成结果表失败：{error}\n")
    print(f"读取训练报告：{source}")
    print(f"关键指标 CSV：{paths['key_metrics_csv']}")
    print(f"中文结果报告：{paths['report_md']}")
    return paths


if __name__ == "__main__":
    main()
