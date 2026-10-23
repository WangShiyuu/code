"""作用：使用求解资源核算块物理时延与能量，供优化预演和训练执行共用。
运行指令：由 optimize_sfl.py 或 train_sfl.py 调用，不单独运行。
输入是活动载荷、块初CSI和实际资源；不执行求解、不读取图片、不更新网络。
"""
from __future__ import annotations
import math

class DynamicExecution:
    """将已核验资源映射为物理账本；不再读取任何参考频率/功率/带宽。"""

    def __init__(self, config, trace):
        """建立每机和服务器的零成本台账；物理时钟从0秒开始，无悬停或电池假设补值。"""
        self.config, self.trace = config, trace
        self.time_s = 0.0
        self.blocks = 0
        self.uavs = [dict(compute_j=0.0, transmit_j=0.0, barrier_wait_s=0.0,
                          compute_s=0.0, up_s=0.0, down_s=0.0)
                     for _ in range(config["mission"]["num_uavs"])]
        self.server = dict(compute_j=0.0, transmit_j=0.0, aggregation_j=0.0)
        self.critical_path = dict(uav_compute_s=0.0, server_compute_s=0.0,
                                  communication_s=0.0, aggregation_s=0.0)
        self.communication = {name: dict(bytes=0, transmissions=0, csi_segments=0,
                                         cross_boundary_transmissions=0)
                              for name in ("activations_and_labels_up", "gradients_down",
                                           "prefix_state_up", "prefix_state_down")}

    def commit(self, problem, decision):
        """重算原模型阶段开销后计账，批末或聚合末推进共同物理时钟。"""
        resources = decision["resources"]
        cfg = self.config
        from transmission import rate_bps
        client, server, ups, downs = [], [], [], []
        for i, uav in enumerate(problem.uav_ids):
            f, p = resources["frequency_hz"][i], resources["upload_power_w"][i]
            up = (problem.up_bits[i] / rate_bps(resources["uplink_bandwidth_hz"][i], p,
                   problem.channel_gains[i], cfg["channel"]["noise_psd_w_hz"]) if problem.up_bits[i] else 0.0)
            down = (problem.down_bits[i] / rate_bps(resources["downlink_bandwidth_hz"][i],
                     cfg["server"]["downlink_power_w"], problem.channel_gains[i],
                     cfg["channel"]["noise_psd_w_hz"]) if problem.down_bits[i] else 0.0)
            c = problem.client_cycles[i] / f if problem.client_cycles[i] else 0.0
            s = problem.server_cycles[i] / cfg["server"]["frequency_hz"]
            client.append(c); server.append(s); ups.append(up); downs.append(down)
            ledger = self.uavs[uav]
            ledger["compute_s"] += c
            ledger["compute_j"] += cfg["uav"]["kappa"] * problem.client_cycles[i] * f * f
            ledger["up_s"] += up; ledger["down_s"] += down
            ledger["transmit_j"] += p * up
            self.server["compute_j"] += cfg["server"]["kappa"] * problem.server_cycles[i] * cfg["server"]["frequency_hz"] ** 2
            self.server["transmit_j"] += cfg["server"]["downlink_power_w"] * down
            categories = (("activations_and_labels_up", "gradients_down") if problem.kind == "training"
                          else ("prefix_state_up", "prefix_state_down"))
            for category, bits in zip(categories, (problem.up_bits[i], problem.down_bits[i])):
                if bits:
                    self.communication[category]["bytes"] += int(round(bits / 8))
                    self.communication[category]["transmissions"] += 1
                    self.communication[category]["csi_segments"] += 1
        if problem.kind == "training":
            totals = [c + s + u + d for c, s, u, d in zip(client, server, ups, downs)]
            slow = max(range(len(totals)), key=totals.__getitem__)
            elapsed = totals[slow]
            for key, value in zip(("uav_compute_s", "server_compute_s", "communication_s"),
                                   (client[slow], server[slow], ups[slow] + downs[slow])):
                self.critical_path[key] += value
            waits = [elapsed - value for value in totals]
            self.blocks += 1
        else:
            tau = problem.aggregation_time_s
            elapsed = max(ups, default=0) + tau + max(downs, default=0)
            self.critical_path["communication_s"] += elapsed - tau
            self.critical_path["aggregation_s"] += tau
            self.server["aggregation_j"] += tau * cfg["server"]["aggregation_power_w"]
            waits = [elapsed - u - d for u, d in zip(ups, downs)]
        if abs(elapsed - decision["duration_s"]) > 1e-6 * max(1, elapsed):
            raise RuntimeError("求解时延与物理执行账本不一致")
        for uav, wait in zip(problem.uav_ids, waits):
            self.uavs[uav]["barrier_wait_s"] += wait
        self.time_s += elapsed

    def report(self):
        """返回实际资源产生的账本，电池可行性保留为未核验。"""
        return dict(simulated_elapsed_s=self.time_s, training_blocks=self.blocks,
                    critical_path=dict(self.critical_path),
                    uavs=[dict(uav_id=i, **ledger, modeled_training_j=ledger["compute_j"] + ledger["transmit_j"])
                          for i, ledger in enumerate(self.uavs)], server=dict(self.server),
                    modeled_system_training_j=math.fsum(v["compute_j"] + v["transmit_j"] for v in self.uavs)
                                              + math.fsum(self.server.values()),
                    communication=self.communication, battery_feasibility=None, mission_deadline_enforced=True,
                    channel_policy="每个训练/聚合块起始CSI冻结，块末重新观测",
                    excluded=["起飞前统一初始化下发", "固定飞行及识别/采集/基础运行能量",
                              "尚未标定的接收、空闲、内存及数据处理功耗", "无线协议开销与重传"])
