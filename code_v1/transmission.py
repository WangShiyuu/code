"""作用：按环境 CSV 的信道保持区间，计算一次传输的完成时间与发射能耗。

运行与使用方式（在 code_v1 目录执行）：
    python -X utf8 -B train_sfl.py --help

本文件由训练入口调用，没有独立命令行任务。也可在 Python 中使用：
    from environment import EnvironmentTrace
    from transmission import transmit
    trace = EnvironmentTrace('output/scenario/environment.csv')
    result = transmit(trace, payload_bits=8192, start_s=0.0, uav_id=0,
                      bandwidth_hz=4e6, power_w=0.5,
                      noise_psd_w_hz=3.981e-21)
    print(result.end_s, result.energy_j, result.segments)

输入载荷以 bit 计，时间为 s，带宽为 Hz，功率为 W，噪声功率谱密度为
W/Hz。输出 TransmissionResult 给出起止时间、时延、发射能量和使用的
CSI 区间数；它只描述一次链路传输，不推进环境对象的内部状态。

功率和带宽由调用方在传输开始时确定，本函数在整次传输中保持它们不变。
每到下一个 CSV 时刻，执行器读取该时刻的新 CSI，继续累计已传输比特。
这用于回放传输实际经历的信道过程，不向资源决策器暴露未来 CSI。上下行
可使用同一接口，并分别传入发射方的功率；energy_j 是该发射方的能耗，
不自动计入 UAV 接收、计算、固定飞行或服务器其他能耗。

环境末行只定义终点状态，不能提供终点之后的保持区间。如果载荷尚未传完
就用尽 CSV，抛出 TraceHorizonError；调用方应准备足够长的诊断环境，
不能静默循环 CSV、冻结最后一个信道，或把未完成传输当作成功。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import EnvironmentTrace


def _finite_number(name, value, *, allow_zero=False):
    """校验一个有限物理量，并按参数约定要求正值或非负值。

    bool 虽然属于 Python 的 int 子类，但不接受其作为时间、功率或载荷。
    此辅助函数只验证输入，不把负值截成零，避免掩盖上游单位或符号错误。
    """
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (not allow_zero and value == 0)):
        requirement = '有限非负数' if allow_zero else '有限正数'
        raise ValueError(f'{name} 必须为{requirement}')


def rate_bps(bandwidth_hz, power_w, channel_gain, noise_psd_w_hz):
    """计算当前信道与固定资源下的 Shannon 速率，单位为 bit/s。

    R = B log2(1 + p h / (N0 B))，其中 h 是线性功率增益，不是 dB 值。
    零发射功率或零增益对应零速率；带宽与噪声功率谱密度必须为正数。
    对极大或极小的信噪比采用稳定的对数表达式，避免中间乘除法溢出。
    该函数计算速率，不执行带宽或功率优化，也不读取环境。
    """
    _finite_number('bandwidth_hz', bandwidth_hz)
    _finite_number('power_w', power_w, allow_zero=True)
    _finite_number('channel_gain', channel_gain, allow_zero=True)
    _finite_number('noise_psd_w_hz', noise_psd_w_hz)
    if power_w == 0 or channel_gain == 0:
        return 0.0
    # log(SNR) 避免先算 p*h 或 N0*B 时出现不必要的浮点上溢或下溢。
    log_snr = (math.log(power_w) + math.log(channel_gain)
               - math.log(noise_psd_w_hz) - math.log(bandwidth_hz))
    if log_snr >= 0:
        log_one_plus_snr = log_snr + math.log1p(math.exp(-log_snr))
    else:
        log_one_plus_snr = math.log1p(math.exp(log_snr))
    rate = bandwidth_hz * log_one_plus_snr / math.log(2.0)
    if not math.isfinite(rate):
        raise ValueError('当前资源产生了无法用有限浮点数表示的传输速率')
    return rate


@dataclass(frozen=True)
class TransmissionResult:
    """一次已经完成的传输结果；所有时间为 s，载荷为 bit，能量为 J。

    segments 是实际经历的 CSI 保持区间数，而不是 CSV 的完整行数。
    零载荷的 segments 为零；非零载荷恰好在某个边界完成时，不额外计入
    后一个区间。energy_j = power_w * duration_s，仅为发射能耗。
    """

    start_s: float
    end_s: float
    duration_s: float
    payload_bits: float
    energy_j: float
    segments: int


class TraceHorizonError(ValueError):
    """CSV 时域不足以完成传输；保留剩余载荷，供调用方诊断或扩展环境。

    此异常只说明回放数据的时间范围不足，不表示系统施加了时延约束。
    本执行器是无状态的，抛出后不会修改 CSV，也不会提交部分通信结果。
    """

    def __init__(self, *, start_s, horizon_s, payload_bits, remaining_bits):
        """保存起始时间、环境终点和剩余载荷，并生成包含单位的错误信息。"""
        self.start_s = start_s
        self.horizon_s = horizon_s
        self.payload_bits = payload_bits
        self.remaining_bits = remaining_bits
        super().__init__(
            f'环境 CSV 在 {horizon_s:g} s 结束：从 {start_s:g} s 开始的 '
            f'{payload_bits:g} bit 传输还剩 {remaining_bits:g} bit；'
            '请延长环境回放时域后重新执行该传输。')


def transmit(trace: EnvironmentTrace, payload_bits, start_s, uav_id,
             bandwidth_hz, power_w, noise_psd_w_hz):
    """在给定链路上累计各 CSI 区间容量，直到完整载荷传输结束。

    trace 为 EnvironmentTrace，uav_id 从零开始。调用方可将上行激活、
    下行梯度或聚合参数的字节数乘以 8，作为 payload_bits 传入。本函数
    不插入协议开销，也不重传；这些均是当前仿真的理想链路假设。

    在 [t_i, t_{i+1}) 内使用 t_i 的信道和固定 B、p 计算速率。若载荷
    大于该区间剩余容量，先消耗该容量，再从 t_{i+1} 的新 CSI 继续。
    剩余载荷可以在区间中间完成，因此结束时刻不强制对齐 CSV 网格。
    start_s 必须位于已存储时域内；非零载荷不能越过 trace.duration_s。
    """
    _finite_number('payload_bits', payload_bits, allow_zero=True)
    _finite_number('start_s', start_s, allow_zero=True)
    _finite_number('bandwidth_hz', bandwidth_hz)
    _finite_number('power_w', power_w, allow_zero=True)
    _finite_number('noise_psd_w_hz', noise_psd_w_hz)
    # 即使载荷为零，也先核对 UAV 编号及起始时间，避免错误输入静默通过。
    state = trace.state_at(start_s, uav_id)
    start = float(start_s)
    if payload_bits == 0:
        return TransmissionResult(start, start, 0.0, 0.0, 0.0, 0)

    remaining = float(payload_bits)
    current = start
    segments = 0
    while remaining > 0:
        boundary = state['next_change_time_s']
        if boundary is None:
            raise TraceHorizonError(start_s=start, horizon_s=trace.duration_s,
                                    payload_bits=float(payload_bits), remaining_bits=remaining)
        interval_s = boundary - current
        if interval_s <= 0:
            raise ValueError('环境必须提供严格递增的下一个 CSI 时间')
        rate = rate_bps(bandwidth_hz, power_w, state['channel_gain'], noise_psd_w_hz)
        capacity_bits = rate * interval_s
        segments += 1
        if rate > 0 and remaining <= capacity_bits:
            # 当前区间只使用传完剩余载荷所需的时间，不向上取整到下一个节点。
            end = min(boundary, current + remaining / rate)
            duration = end - start
            energy = power_w * duration
            if not math.isfinite(energy):
                raise ValueError('发射能耗无法用有限浮点数表示')
            return TransmissionResult(start, end, duration, float(payload_bits), energy, segments)
        remaining -= capacity_bits
        current = boundary
        # 到达新时刻后才查询该时刻 CSI。固定带宽、功率没有依据未来值重选。
        state = trace.state_at(current, uav_id)

    # 正常完成均在区间内直接返回；保留这一保护，便于发现异常浮点输入。
    raise RuntimeError('传输累计过程异常结束')
