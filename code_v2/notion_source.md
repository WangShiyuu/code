Here is the result of "fetch" for the Page with URL https://app.notion.com/p/3b5427f94a0b8040a213d5d9b277c8df as of 2026-10-29T18:45:37.698Z:
<page url="https://app.notion.com/p/3b5427f94a0b8040a213d5d9b277c8df">
<ancestor-path>
<parent-page url="https://app.notion.com/p/2f3427f94a0b802fa943f94e4d054a5a" title="第一个工作"/>
<ancestor-2-page url="https://app.notion.com/p/31e427f94a0b800a9f51dabcc059554a" title=""/>
<ancestor-3-page url="https://app.notion.com/p/31e427f94a0b801eabbbde94d5cf6312" title=""/>
<ancestor-4-page url="https://app.notion.com/p/26f427f94a0b8030ae09cd15f1f4adb3" title="研究生"/>
</ancestor-path>
<properties>
{"title":"<p>\n动态UAV-SFL \n</p>"}
</properties>
<iconMetadata>null</iconMetadata>
<content>
# Ⅰ. 研究背景与问题描述

无人机巡逻任务需要在持续飞行中采集图像并开展模型训练，但机载算力、电池容量及无线链路条件限制了训练任务的执行效率。分割联邦学习（split federated learning，SFL）将完整神经网络划分为无人机侧前缀和边缘服务器侧后缀，使多架无人机能够利用服务器算力完成协同训练，并将原始样本保留在本地。对于沿给定轨迹运动的无人机，链路距离和传播条件随时间变化；持续采集又使后续训练轮次的数据规模增加。因此，模型切分、计算与通信资源需要随执行状态协调配置。

本文考虑由多架无人机和一台地面边缘服务器构成的巡逻训练系统。无人机按照预定轨迹持续飞行和采集，在样本积累达到规定数量后，从统一的预训练模型开始训练。每个本地 epoch 使用轮初已获得的数据，轮内新增样本在下一 epoch 加入；每完成两个 epoch，服务器执行一次完整模型聚合。总训练轮数 $K_t$ 由第Ⅲ节的一阶精度条件确定，并在仿真执行前固定；本文在该训练预算下研究系统能耗最小化问题。

模型切分决定计算在无人机与服务器之间的分布，同时影响激活值、激活梯度及模型参数的传输开销。本文在任务时限和无人机可用训练电量约束下，联合优化切分点、无人机计算功率、上行发射功率及上下行带宽。切分点在聚合边界更新，连续资源在每个 epoch 开始时配置，形成双时间尺度的执行机制。飞行轨迹和采集规则作为给定输入，实际执行时间进一步决定后续的无人机位置、链路状态和可用样本量。

# Ⅱ. 系统模型

## A. 网络、飞行与任务时间

无人机集合为 $\mathcal N=\{1,\ldots,N\}$，地面边缘服务器记为 $e$。巡逻任务的物理时间为 $t\in[0,T]$，其中 $T$ 为实际任务结束时刻。无人机 $n$ 的位置由给定航迹确定，其三维位置向量为

$$
\boldsymbol q_n(t)
=\big[x_n(t),\,y_n(t),\,z_n(t)\big]
\in\mathbb R^{1\times3}.
$$

地面边缘服务器的固定位置向量为

$$
\boldsymbol q_e
=\big[x_e,\,y_e,\,z_e\big]
\in\mathbb R^{1\times3}.
$$

其中，$x$、$y$ 表示水平坐标，$z$ 表示高度坐标。

无人机飞行、采集、初始模型预部署和基础运行能耗可合并为常数 $E_{\mathrm{fix}}$。无人机 $n$ 在扣除上述固定任务消耗后可用于训练的电量记为 $E_n^{\mathrm{cap}}$，直接作为已知预算。

任务开始前，无人机与服务器预先部署候选切分所需的模型模块和预训练参数。

## B. 连续采集与 epoch 定义

以 $k=1,\ldots,K_t$ 表示本地训练轮次，每轮对应一个 epoch。$t_k$ 为第 $k$ 轮的共同开始时刻，$t_1$ 由给定采集规则和训练启动条件确定。令 $D(t)$ 表示无人机在物理时刻 $t$ 已获取的累计训练样本数，则该轮冻结的训练集为 $\mathcal D_{k}$，样本数满足

$$
D_{k}=|\mathcal D_{k}|=D(t_k),
\qquad
\mathcal D_{k}\subseteq\mathcal D_{k+1}.
\tag{1}
$$

$D(t)$ 是给定采集计划产生的非减计数函数。不同无人机具有相同的累计样本数。

一个 epoch 完整遍历一次轮初训练集。该轮内新采集的样本先进入缓存，在下一轮开始时加入训练；模型参数在相邻轮次之间连续更新，不因数据集扩大而重新初始化。所有无人机在轮末同步。

## C. 模型切分与周期聚合

完整模型采用 ResNet18，其四个阶段各含两个残差块，共八个残差块 \[1\]。在残差块边界设置九个候选切分点 $\mathcal S=\{0,\ldots,8\}$。对于切分点 $s$，无人机执行输入端基础模块（stem）及前 $s$ 个残差块，服务器执行剩余残差块和池化、分类头。因此，$s=0$ 位于首个残差块之前，$s=8$ 位于最后一个残差块之后，任一残差块均保持完整。客户端前缀与服务器后缀满足

$$
\boldsymbol w
=\left(\boldsymbol w_{\mathrm{UAV}}^{(s)},
\boldsymbol w_e^{(s)}\right),
\qquad s\in\mathcal S.
\tag{2}
$$

两端预部署支持候选切分的模型模块，实际训练仅执行当前切分对应的互补前缀和后缀。服务器为每架无人机保留配套的后缀参数，从而不同分支均对应同一架构的完整模型。切分点 $s$ 的离线画像定义为

$$
\boldsymbol\phi_s=
\left(C_s^{\mathrm c},C_s^{\mathrm e},A_s,G_s,M_s\right).
\tag{3}
$$

其中，$C_s^{\mathrm c}$ 和 $C_s^{\mathrm e}$ 分别为处理单个样本时客户端、服务器完成前向和反向传播所需的等效计算周期数；$A_s$ 为单样本激活值及标签的上传比特数；$G_s$ 为单样本激活梯度的下传比特数；$M_s$ 为客户端完整前缀参数的比特数。

每个批次依次完成无人机前向计算、激活值及标签上传、服务器前向与反向计算、激活梯度返回及无人机反向计算。两端使用同一更新前模型计算梯度，并采用一致的普通 SGD 更新规则。

以 $s_{n,k}$ 表示无人机 $n$ 在第 $k$ 轮使用的切分点。系统每 $K=2$ 个 epoch 聚合一次，聚合指示量及周期内切分约束为

$$
\delta_k=\mathbf 1\{k\bmod K=0\},\qquad K=2,
\qquad
s_{n,k+1}=s_{n,k}\quad
(\delta_k=0,\ k<K_t).
\tag{4}
$$

第Ⅲ节给出的总训练轮数 $K_t$ 为 $K$ 的整数倍。聚合时，各无人机上传当前前缀，服务器将其与对应后缀拼接为完整模型 $\widetilde{\boldsymbol w}_{n,k}$，再执行按当前样本数加权的模型平均 \[2\]：

$$
\boldsymbol w_k=
\sum_{n=1}^{N}
\frac{D_{n,k}}{\sum_{i=1}^{N}D_{i,k}}
\widetilde{\boldsymbol w}_{n,k},
\qquad \delta_k=1.
\tag{5}
$$

各分支样本数相同，式（5）退化为等权平均。服务器聚合后根据下一周期切分点下发相应新前缀，并保留对应后缀。

## D. 动态信道与通信速率

本文采用基于概率 LoS/NLoS 传播条件的平均大尺度信道模型。在第 $k$ 个 epoch 开始时，根据给定轨迹确定无人机位置，计算该 epoch 使用的信道增益，并配置计算功率、发射功率及上下行带宽。无人机与服务器之间的距离及仰角为

$$
\begin{aligned}
d_{n,k}
&=\|\boldsymbol q_n(t_k)-\boldsymbol q_e\|\\
&=\sqrt{
\big(x_n(t_k)-x_e\big)^2
+\big(y_n(t_k)-y_e\big)^2
+\big(z_n(t_k)-z_e\big)^2},\\
\theta_{n,k}
&=\frac{180}{\pi}
\arcsin\left(
\frac{z_n(t_k)-z_e}{d_{n,k}}
\right).
\end{aligned}
\tag{6}
$$

其中，仰角以度计。采用概率 LoS/NLoS 传播模型 \[3\]，两种传播条件下的大尺度功率增益分别为

$$
h_{n,k}^{\mathrm L}
=\beta_{\mathrm L}d_{n,k}^{-\alpha_{\mathrm L}},
\qquad
h_{n,k}^{\mathrm N}
=\beta_{\mathrm N}d_{n,k}^{-\alpha_{\mathrm N}}.
$$

其中，$\beta_{\mathrm L},\beta_{\mathrm N}$ 为参考功率增益，$\alpha_{\mathrm L},\alpha_{\mathrm N}$ 为相应的路径损耗指数。

LoS 概率由链路仰角确定：

$$
P_{n,k}^{\mathrm L}
=\frac{1}{1+a\exp[-b(\theta_{n,k}-a)]},
$$

其中，$a,b$ 为环境参数，NLoS 概率为 $1-P_{n,k}^{\mathrm L}$。由全期望公式，对两种传播条件的功率增益按发生概率加权，得到平均大尺度功率增益

$$
h_{n,k}
=P_{n,k}^{\mathrm L}h_{n,k}^{\mathrm L}
+(1-P_{n,k}^{\mathrm L})h_{n,k}^{\mathrm N}

\tag{7}
$$

$h_{n,k}$ 以线性功率增益表示，并根据每个 epoch 开始时的距离和仰角更新。

上下行采用正交带宽分配。以 $p_{n,k}$ 表示无人机上行发射功率，$p_e$ 表示服务器对每条下行链路的发射功率，$N_0$ 为噪声功率谱密度，则上下行速率为

$$
\begin{aligned}
R_{n,k}^{\mathrm u}
&=B_{n,k}^{\mathrm u}
\log_2\left(1+\frac{p_{n,k}h_{n,k}}{N_0B_{n,k}^{\mathrm u}}\right),\\
R_{n,k}^{\mathrm d}
&=B_{n,k}^{\mathrm d}
\log_2\left(1+\frac{p_eh_{n,k}}{N_0B_{n,k}^{\mathrm d}}\right).
\end{aligned}
\tag{8}
$$

其中，$B_{n,k}^{\mathrm u}$ 和 $B_{n,k}^{\mathrm d}$ 分别为无人机 $n$ 在第 $k$ 个 epoch 的上行与下行带宽。式（8）以平均大尺度功率增益计算等效通信速率，后续时延和能耗均按该速率计算。

## E. 时延与能耗

训练过程包括每个 epoch 执行的分割训练，以及每 $K$ 个 epoch 执行一次的模型聚合 \[8\]。计算开销按无人机侧和服务器侧分别统计，两端均合并前向与反向计算；通信开销包括激活值上传和激活梯度下传。以下时延与能耗均按一个 epoch 内全部批次累计，处理样本总数为 $D_k$。采集与训练并行进行，飞行、采集及基础运行的固定开销沿用第Ⅱ节A的 $E_{\mathrm{fix}}$。

无人机采用动态电压频率调节模型 \[4\]，计算功率与频率满足

$$
P_{n,k}^{\mathrm c}=\kappa_n f_{n,k}^{3},
\qquad
f_{n,k}=\left(\frac{P_{n,k}^{\mathrm c}}{\kappa_n}\right)^{1/3}.
\tag{9}
$$

客户端和服务器的单样本计算量分别为 $C_s^{\mathrm c}$ 和 $C_s^{\mathrm e}$，均包含对应模型的前向与反向计算。

<b>1) 无人机计算：</b> 无人机完成本地前缀的前向传播、反向传播及参数更新，其合计计算时延与能耗为

$$
T_{n,k}^{\mathrm c}
=\frac{D_kC_{s_{n,k}}^{\mathrm c}}{f_{n,k}},
\qquad
E_{n,k}^{\mathrm c}
=P_{n,k}^{\mathrm c}T_{n,k}^{\mathrm c}.
\tag{10a}
$$

<b>2) 激活值上传：</b> 无人机将激活值及标签发送至服务器。单样本上传量为 $A_{s_{n,k}}$，因此

$$
T_{n,k}^{\mathrm A}
=\frac{D_kA_{s_{n,k}}}{R_{n,k}^{\mathrm u}},
\qquad
E_{n,k}^{\mathrm A}
=p_{n,k}T_{n,k}^{\mathrm A}.
\tag{10b}
$$

<b>3) 服务器计算：</b> 服务器使用对应后缀完成前向传播、损失计算和反向传播，得到切分层激活梯度。各分支计算频率为 $f_e$，计算功率系数为 $\kappa_e$，其时延与能耗为

$$
T_{n,k}^{\mathrm{e,c}}
=\frac{D_kC_{s_{n,k}}^{\mathrm e}}{f_e},
\qquad
E_{n,k}^{\mathrm{e,c}}
=\kappa_e f_e^3T_{n,k}^{\mathrm{e,c}}.
\tag{10c}
$$

<b>4) 激活梯度下传：</b> 服务器将单样本大小为 $G_{s_{n,k}}$ 的激活梯度返回无人机。对应的下行时延与服务器发射能耗为

$$
T_{n,k}^{\mathrm G}
=\frac{D_kG_{s_{n,k}}}{R_{n,k}^{\mathrm d}},
\qquad
E_{n,k}^{\mathrm G}
=p_eT_{n,k}^{\mathrm G}.
\tag{10d}
$$

将每个批次的两端计算与上下行通信时延相加，并累计全部批次，得到无人机 $n$ 的分割训练总时延：

$$
\begin{aligned}
T_{n,k}^{\mathrm{SL}}
&=T_{n,k}^{\mathrm c}+T_{n,k}^{\mathrm A}
+T_{n,k}^{\mathrm{e,c}}+T_{n,k}^{\mathrm G}\\
&=D_k\left(
\frac{C_{s_{n,k}}^{\mathrm c}}{f_{n,k}}
+\frac{A_{s_{n,k}}}{R_{n,k}^{\mathrm u}}
+\frac{C_{s_{n,k}}^{\mathrm e}}{f_e}
+\frac{G_{s_{n,k}}}{R_{n,k}^{\mathrm d}}
\right).
\end{aligned}
\tag{11}
$$

各无人机及其对应服务器后缀构成并行训练分支，分支内部顺序执行，完成整个 epoch 后同步。因此，分割训练阶段的持续时间为 $\max_nT_{n,k}^{\mathrm{SL}}$。

<b>5) 周期模型聚合：</b> 当 $\delta_k=1$ 时，依次执行前缀上传、完整模型聚合和新前缀下发。当前前缀上传量为 $M_{s_{n,k}}$；下发量由下一周期切分点确定：

$$
M_{n,k}^{+}=
\begin{cases}
M_{s_{n,k+1}},&k<K_t,\\
0,&k=K_t.
\end{cases}
\tag{12}
$$

上传阶段，无人机发送当前前缀，其时延和发射能耗为

$$
T_{n,k}^{\mathrm{mu}}=\frac{M_{s_{n,k}}}{R_{n,k}^{\mathrm u}},
\qquad
E_{n,k}^{\mathrm{mu}}=p_{n,k}T_{n,k}^{\mathrm{mu}}.
\tag{12a}
$$

服务器收到全部前缀后，将其与对应后缀拼接并执行式（5）的聚合，处理时延和能耗分别为 $\tau_{\mathrm{agg}}$ 和 $E_{\mathrm{agg}}$。聚合完成后，按新切分向各无人机下发前缀，其时延和服务器发射能耗为

$$
T_{n,k}^{\mathrm{md}}=\frac{M_{n,k}^{+}}{R_{n,k}^{\mathrm d}},
\qquad
E_{n,k}^{\mathrm{md}}=p_eT_{n,k}^{\mathrm{md}}.
\tag{12b}
$$

上述参数通信仅在聚合 epoch 发生，最终聚合后不再下发模型。

<b>6) 单轮与任务总开销：</b> 分割训练、前缀上传、聚合处理和新前缀下发依次进行，因此第 $k$ 个 epoch 的总时延为

$$
T_k=\max_nT_{n,k}^{\mathrm{SL}}
+\delta_k\left(
\max_nT_{n,k}^{\mathrm{mu}}
+\tau_{\mathrm{agg}}
+\max_nT_{n,k}^{\mathrm{md}}
\right).
\tag{13}
$$

给定单轮时限 $T_{\max}$，第Ⅳ节要求 $T_k\le T_{\max}$。实际时间按下式递推：

$$
t_{k+1}=t_k+T_k,
\qquad
t_{K_t+1}=t_1+\sum_{k=1}^{K_t}T_k.
\tag{14}
$$

下一轮的样本数和信道分别由 $D(t_{k+1})$ 及给定轨迹上的位置确定。

无人机能耗由本地计算、激活值上传及聚合时的前缀上传构成：

$$
E_{n,k}^{\mathrm U}
=E_{n,k}^{\mathrm c}+E_{n,k}^{\mathrm A}
+\delta_kE_{n,k}^{\mathrm{mu}}.
\tag{15}
$$

服务器能耗为各分支后缀计算、激活梯度下传、新前缀下发和聚合处理能耗之和：

$$
E_k^e=
\sum_n\left(
E_{n,k}^{\mathrm{e,c}}+E_{n,k}^{\mathrm G}
+\delta_kE_{n,k}^{\mathrm{md}}
\right)+\delta_kE_{\mathrm{agg}}.
\tag{16}
$$

各阶段能耗按实际计算或发射时间计量。单轮能耗及完成 $K_t$ 个 epoch 的系统总能耗为

$$
E_k=\sum_nE_{n,k}^{\mathrm U}+E_k^e,
\qquad
E_{\mathrm{sys}}=E_{\mathrm{fix}}+\sum_{k=1}^{K_t}E_k.
\tag{17}
$$

其中，$E_{\mathrm{fix}}$ 在报告系统总能耗时计入，在能耗最小化目标中可省略。

# Ⅲ. 总体训练目标与训练预算

第 $k$ 个 epoch 的联合经验损失为

$$
F_k(\boldsymbol w)=\frac{1}{N}\sum_{n=1}^{N}
\frac{1}{D_k}\sum_{i=1}^{D_k}\ell(\boldsymbol w;\xi_{n,k,i}).
\tag{18}
$$

各无人机完整遍历轮初数据，采用普通 SGD 更新：

$$
\boldsymbol w^+=\boldsymbol w-\eta\frac{|\mathcal B|}{D_k}
\nabla\ell_{\mathcal B}(\boldsymbol w).
\tag{19}
$$

其中，$\ell_{\mathcal B}$ 为批次平均损失，$\eta$ 为每个 epoch 的累计步长，尾批按实际样本数计入。两端使用同一更新前参数并精确传输激活值与梯度，分割更新保持完整模型的梯度关系。分析采用固定逐样本损失，BatchNorm 使用固定统计量。一个训练轮对应一个本地 epoch，每 $K=2$ 个 epoch 聚合一次，总聚合次数为 $J=K_t/K$。

Zhai 等在文献 \[9\] 定理1、式（8）中给出

$$
\frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla f(\bar{\boldsymbol x}^{(t)})\|^2
\le
\frac{\Delta/(\lambda T)+\lambda\omega\alpha^2/(2M)
+(\alpha^2+9\beta^2)\mathcal G}{1/2-9\mathcal G}.
\tag{20}
$$

其中，$\bar{\boldsymbol x}^{(t)}=\boldsymbol X^{(t)}\boldsymbol1/M$，$\Delta=f_0-f^\star$；$M,\lambda,\omega$ 分别为原文的设备数、步长和光滑常数，$\alpha^2,\beta^2$ 分别为梯度方差和数据异质性的上界。原文的聚合谱系数及步长条件为

$$
\mathcal G=
\frac{M\lambda^2\omega^2}
{(1-\sqrt{\rho(\overline{W^2})})^2-18M\lambda^2\omega^2},
\qquad
0<\lambda<
\frac{1-\sqrt{\rho(\overline{W^2})}}{6\sqrt M\,\omega}.
\tag{21}
$$

该条件保证 $d=1/2-9\mathcal G>0$。记 $A_{\mathrm Z}=\Delta/(\lambda d)$、$C_{\mathrm Z}=[\lambda\omega\alpha^2/(2M)+(\alpha^2+9\beta^2)\mathcal G]/d$。由最小值不大于平均值，取有效上界 $\overline A_{\mathrm Z}\ge A_{\mathrm Z}$、$C_{\mathrm Z}\le\overline C_{\mathrm Z}<\varepsilon$，则

$$
\begin{aligned}
\min_{0\le t<T}\mathbb E\|\nabla f(\bar{\boldsymbol x}^{(t)})\|^2
&\le\frac{A_{\mathrm Z}}T+C_{\mathrm Z}
\le\frac{\overline A_{\mathrm Z}}T+\overline C_{\mathrm Z},\\
\frac{\overline A_{\mathrm Z}}T+\overline C_{\mathrm Z}\le\varepsilon
&\quad\Longleftrightarrow\quad
T\ge\frac{\overline A_{\mathrm Z}}{\varepsilon-\overline C_{\mathrm Z}}.
\end{aligned}
\tag{22}
$$

原式的 $T$ 计去中心化 SGD 更新次数。本文参照其“轮数衰减项＋误差底限”结构，以 $\overline A/K_t+\overline C\le\varepsilon$ 作为 epoch 预算条件；$\overline A,\overline C$ 须为覆盖本文周期聚合与持续采样的有效系数，原文的谱系数不直接代入本文协议。若该同型界成立且 $\overline C<\varepsilon$，考虑 $K_t$ 为 $K$ 的整数倍，取

$$
K_t=K\max\left\{1,\left\lceil
\frac{\overline A}{K(\varepsilon-\overline C)}
\right\rceil\right\},
\qquad J=\frac{K_t}{K}.
\tag{23}
$$

例如，有效系数取 $\overline A=3.85$、$\overline C=0.0115$、$\varepsilon=0.05$ 时，预算为100个 epoch、50次聚合。该数值为条件性示例；仿真执行前固定选定的 $K_t$。

# Ⅳ. 系统能耗优化与双时间尺度求解

## A. 任务级优化问题

记 $\boldsymbol x_k=\{P_{n,k}^{\mathrm c},p_{n,k},B_{n,k}^{\mathrm u},B_{n,k}^{\mathrm d}\}_{n=1}^{N}$。给定第Ⅲ节确定的 $K_t$，省略目标中的常数 $E_{\mathrm{fix}}$，任务级优化问题为

$$
\begin{aligned}
\mathbf P_0:\quad\min_{\{s_{n,k},\boldsymbol x_k\}}\;&\sum_{k=1}^{K_t}E_k\\
\mathrm{s.t.}\;&s_{n,k}\in\mathcal S,\\
&s_{n,k+1}=s_{n,k},\quad\delta_k=0,\ k<K_t,\\
&T_k\le T_{\max},\quad\sum_{k=1}^{K_t}E_{n,k}^{\mathrm U}\le E_n^{\mathrm{cap}},\\
&0<P_{n,k}^{\mathrm c}\le P_n^{\mathrm{c,max}},\quad0<p_{n,k}\le p_n^{\max},\\
&B_{n,k}^{\mathrm u}>0,\quad B_{n,k}^{\mathrm d}>0,\\
&\sum_nB_{n,k}^{\mathrm u}\le B_{\mathrm{tot}}^{\mathrm u},\quad
\sum_nB_{n,k}^{\mathrm d}\le B_{\mathrm{tot}}^{\mathrm d}.
\end{aligned}
\tag{24}
$$

上述约束对全部适用的无人机 $n$ 和轮次 $k$ 成立。

其中，$D_k=D(t_k)$、$h_{n,k}=h_n(t_k)$，且 $t_{k+1}=t_k+T_k$。切分点在聚合后更新，连续资源在每个 epoch 开始时更新。采用窗口信道选择切分，并在当前数据量和信道下求解资源子问题，在线逼近任务级问题。

## B. 切分点选择子问题及求解

在新周期开始前，固定参考资源，利用最近至多 $W$ 个已完成 epoch 的平均信道估计上下行速率。首周期使用初始信道和给定可行参考资源。记新周期切分向量为 $\boldsymbol s$，参考状态下的切分子问题为

$$
\begin{aligned}
\mathbf P_1:\quad
\min_{\boldsymbol s}\quad
&\sum_n\mathcal E_n^{\mathrm{ref}}(s_n)\\
\mathrm{s.t.}\quad
&s_n\in\mathcal S,\quad \forall n,\\
&\widehat T_\ell(\boldsymbol s)\le T_{\max},
\quad \ell\in\mathcal H,\\
&\sum_{\ell\in\mathcal H}\widehat E_{n,\ell}^{\mathrm U}(\boldsymbol s)
\le E_n^{\mathrm{rem}},\quad \forall n.
\end{aligned}
\tag{25}
$$

$E_n^{\mathrm{rem}}$ 为决策时刻的剩余训练电量；$\mathcal H$ 包含待进入周期及尚未执行的前一聚合 epoch，首周期仅包含本周期。参考时延和能耗按式（13）、（15）计算，资源取最近一次可行配置，样本数取当前 $D(t)$。若预测范围末端仍有下一周期，其下发量按保持候选切分估计；最终聚合后取零。

令 $\widehat m=KD(t)$，$\chi=0$ 表示首周期，其他周期取 $\chi=1$。与候选 $s$ 有关的参考能耗评分为

$$
\begin{aligned}
\mathcal E_n^{\mathrm{ref}}(s)
={}&\widehat m
\left[\kappa_n(f_n^{\mathrm{ref}})^2C_s^{\mathrm c}
+\kappa_e f_e^2C_s^{\mathrm e}\right]\\
&+\frac{p_n^{\mathrm{ref}}(\widehat m A_s+M_s)}
{R_n^{\mathrm{u,ref}}}
+\frac{p_e(\widehat m G_s+\chi M_s)}
{R_n^{\mathrm{d,ref}}}.
\end{aligned}
\tag{26}
$$

该评分包含周期内两端计算、激活值与梯度通信，以及周期末前缀上传和周期开始前的新前缀下发。上一周期的旧前缀上传及聚合处理与新候选无关，省略于评分，但保留在可行性检验中。

固定其他无人机的切分，逐一枚举本机九个候选。仅接受满足式（25）且使 $\Delta\mathcal E_n=\mathcal E_n^{\mathrm{ref}}(s_{\mathrm{new}})-\mathcal E_n^{\mathrm{ref}}(s_{\mathrm{old}})<0$ 的更新，循环扫描至无改进；评分相同时保留原切分。该 trade-off 同时比较机载计算增加、服务器计算减少以及通信负载变化。首周期枚举取得可行初值；后续优先沿用原切分。

非最终聚合 epoch 开始时预选下一周期切分，使本轮下发量 $M_{n,k}^{+}$ 在资源求解前确定。参考筛选用于候选排序，最终按当前状态求解资源子问题。

## C. 资源分配子问题及求解

固定当前切分、适用的下一周期切分及当前 $D_k,h_{n,k}$。省略下标 $k$，记 $E_n^{\mathrm{rem}}=E_n^{\mathrm{cap}}-\sum_{\ell<k}E_{n,\ell}^{\mathrm U}$。单 epoch 的资源分配问题为

$$
\begin{aligned}
\mathbf P_2:\quad
\min_{\boldsymbol x}\quad &E_k\\
\mathrm{s.t.}\quad
&T_k\le T_{\max},\\
&E_n^{\mathrm U}\le E_n^{\mathrm{rem}},\quad \forall n,\\
&0<P_n^{\mathrm c}\le P_n^{\mathrm{c,max}},\quad \forall n,\\
&0<p_n\le p_n^{\max},\quad \forall n,\\
&B_n^{\mathrm u}>0,\quad B_n^{\mathrm d}>0,\quad \forall n,\\
&\sum_n B_n^{\mathrm u}\le B_{\mathrm{tot}}^{\mathrm u},\quad
\sum_n B_n^{\mathrm d}\le B_{\mathrm{tot}}^{\mathrm d}.
\end{aligned}
\tag{27}
$$

<b>切比雪夫变换。</b> 对式（13）的各阶段最大时延作切比雪夫型上图变换，引入非负辅助时间 $\tau^{\mathrm{SL}},\tau^{\mathrm u},\tau^{\mathrm d}$。约束 $T_k\le T_{\max}$ 等价于

$$
\begin{aligned}
&T_n^{\mathrm{SL}}\le\tau^{\mathrm{SL}},\quad \forall n,\\
&\frac{M_{s_n}}{R_n^{\mathrm u}}\le\tau^{\mathrm u},\quad
\frac{M_n^{+}}{R_n^{\mathrm d}}\le\tau^{\mathrm d},
\quad \forall n,\ \delta=1,\\
&\tau^{\mathrm{SL}}+
\delta(\tau^{\mathrm u}+\tau_{\mathrm{agg}}+\tau^{\mathrm d})
\le T_{\max}.
\end{aligned}
\tag{28}
$$

各阶段分别引入辅助时间，以保持同步执行顺序。非聚合 epoch 删除参数通信阶段，最终 epoch 删除无负载的下发阶段。

将计算功率重参数化为 $f_n=(P_n^{\mathrm c}/\kappa_n)^{1/3}$，计算能耗变为 $\kappa_nDC_{s_n}^{\mathrm c}f_n^2$，计算时延为 $DC_{s_n}^{\mathrm c}/f_n$。令 $\Lambda_n^{\mathrm u}=DA_{s_n}+\delta M_{s_n}$、$\Lambda_n^{\mathrm d}=DG_{s_n}+\delta M_n^{+}$。固定切分后，服务器计算和聚合能耗为常数；余下目标包含计算能耗及上下行能耗比式之和。

<b>比式和变换与迭代。</b> 对上行能耗比式，采用逆二次变换 \[6\]：

$$
\frac{a_n}{b_n}=
\min_{\substack{y_n>0\\2y_n\sqrt{b_n}-y_n^2a_n>0}}
\frac1{2y_n\sqrt{b_n}-y_n^2a_n},
\quad
a_n=\Lambda_n^{\mathrm u}p_n,\quad b_n=R_n^{\mathrm u}.
\tag{29}
$$

令 $\boldsymbol z$ 汇集频率、功率、带宽及活动阶段辅助时间。从原问题可行点 $\boldsymbol z^0$ 出发，第 $i$ 次迭代更新

$$
\begin{aligned}
y_n^i&=\frac{\sqrt{R_n^{\mathrm u}(\boldsymbol z^i)}}{\Lambda_n^{\mathrm u}p_n^i},\\
\psi_n^i(\boldsymbol z)&=
2y_n^i\sqrt{R_n^{\mathrm u}(\boldsymbol z)}
-(y_n^i)^2\Lambda_n^{\mathrm u}p_n.
\end{aligned}
\tag{30}
$$

固定 $\boldsymbol y^i$，以 $f_n^{\max}=(P_n^{\mathrm{c,max}}/\kappa_n)^{1/3}$ 表示频率上限，求解

$$
\begin{aligned}
\boldsymbol z^{i+1}\in\arg\min_{\boldsymbol z}\;&\sum_n\left[\kappa_nDC_{s_n}^{\mathrm c}f_n^2+\frac1{\psi_n^i(\boldsymbol z)}+\frac{p_e\Lambda_n^{\mathrm d}}{R_n^{\mathrm d}}\right]\\
\mathrm{s.t.}\;&\text{(28)},\quad\tau^{\mathrm{SL}},\tau^{\mathrm u},\tau^{\mathrm d}\ge0,\\
&\kappa_nDC_{s_n}^{\mathrm c}f_n^2+\frac1{\psi_n^i(\boldsymbol z)}\le E_n^{\mathrm{rem}},\\
&\psi_n^i(\boldsymbol z)>0,\quad0<f_n\le f_n^{\max},\\
&0<p_n\le p_n^{\max},\quad B_n^{\mathrm u}>0,\ B_n^{\mathrm d}>0,\\
&\sum_nB_n^{\mathrm u}\le B_{\mathrm{tot}}^{\mathrm u},\quad\sum_nB_n^{\mathrm d}\le B_{\mathrm{tot}}^{\mathrm d}.
\end{aligned}
\tag{31}
$$

式（31）中的单机约束对所有 $n$ 成立。固定 $\boldsymbol y^i$ 后，$\psi_n^i$ 为凹函数，其倒数、计算能耗及式（28）的时延项均为凸函数，故式（31）可由 CVX 求解 \[5,7\]。交替执行式（30）、（31），至真实能耗的相对变化低于容差或达到迭代上限。该变换在当前点与原比式相切，从可行点出发保持原约束可行并使真实能耗不增。

求解后恢复 $P_n^{\mathrm c}=\kappa_nf_n^3$，核验原始时延与电量约束，并按式（13）推进实际时间。单轮可行性不保证剩余全部训练轮次均可完成。

## D. 双时间尺度执行流程

<b>算法1　窗口切分与逐轮资源分配</b>

```txt
输入：预训练模型、轨迹、采集计划、模型画像、资源与电量边界、
      训练预算 K_t、聚合间隔 K=2、窗口长度 W、容差 ε_r、迭代上限 I_max
输出：最终模型、切分与资源配置、累计能耗及任务完成状态

 1  初始化首周期切分 s、参考资源、信道窗口、t ← t_1、E_n^rem ← E_n^cap
 2  for k = 1, …, K_t do
 3      更新 D_k = D(t)、h_{n,k} 及 δ_k
 4      if δ_k = 1 and k < K_t then
 5          Q_k ← 按参考能耗排序的下一周期候选切分（式25、26）
 6      else
 7          Q_k ← {∅}
 8      end if
 9      feasible ← false
10      for q ∈ Q_k do
11          设置 M_n^+ ← M_{q_n}（q ≠ ∅），否则 M_n^+ ← 0
12          构造固定当前切分 s 的可行初值 z^0；若未获得则 continue
13          i ← 0
14          repeat
15              按式（30）更新 y^i；固定 y^i，用 CVX 求解式（31）
16              若求解失败，转至下一候选 q；否则 i ← i + 1
17          until 相对能耗变化 ≤ ε_r or i = I_max
18          若原始时延和电量约束满足：feasible ← true；break
19      end for
20      if feasible = false then return 已执行结果与“任务未完成” end if
21      按当前切分 s 和所得资源执行本 epoch 分割训练
22      if δ_k = 1 then
23          上传当前前缀并聚合完整模型
24          if k < K_t then 下发新前缀；s ← q end if
25      end if
26      t ← t + T_k；E_n^rem ← E_n^rem − E_{n,k}^U
27      更新信道窗口及新增样本缓存
28  end for
29  return 最终模型、配置、累计能耗与“任务完成”
```

# Ⅴ. 仿真数据与评价设置

- [ ] 确定数据集、类别与独立划分单位，固定训练／验证／测试集。基准比例为 $80\%:10\%:10\%$，避免同一独立对象跨集合泄漏。
- [ ] 固定无人机数及客户端划分。若采用 27000 个样本、5 架无人机，则训练／验证／测试样本数为 21600／2700／2700，每架无人机分配 4320 个训练样本；该配置须按实际数据集核对。
- [ ] 生成并保存按物理时间定义的样本释放表，各无人机同一时刻具有相同样本数。若初始释放比例为 $10\%$，上述示例每机先释放 432 个样本；其余样本按统一采集计划释放。
- [ ] 实现轮初冻结数据集、轮内新增样本入缓存、下一 epoch 加入的机制，并记录每轮 $t_k$、$D_k$ 及样本索引。
- [ ] 固定 ResNet18 预训练参数、批量大小、损失和式（19）的 SGD 步长规则；按理论设置固定 BatchNorm 统计量。验证同一批次下不同切分的完整模型梯度和参数更新一致。
- [ ] 实测九个切分点的 $\{C_s^{\mathrm c},C_s^{\mathrm e},A_s,G_s,M_s\}$，两端计算量均合并计入前向与反向计算；核对计算周期、bit、Hz、W、J、s 等单位，并区分尾批。
- [ ] 固定飞行轨迹、采集计划、LoS/NLoS 参数、上下行总带宽、功率上限、服务器频率、聚合开销、$T_{\max}$ 和单机训练电量。
- [ ] 核对文献 \[9\] 的原式与迭代计数，验证同型 epoch 级界对本文周期聚合和持续采样的适用性，给出有效 $\overline A,\overline C,\varepsilon$ 到 $K_t$ 的计算表。参数估计与严格上界分别记录。
- [ ] 以 $K_t=100$、$K=2$ 作为待验证的主实验配置；仿真运行期间固定预算。若同型有效界尚未验证，将其标为经验仿真预算；取得有效系数后另报理论预算。
- [ ] 实现窗口切分的逐机枚举与 trade-off 评分，记录每次候选的计算、特征传输和模型同步能耗，以及接受或拒绝原因。
- [ ] 实现切比雪夫上图变换、式（30）的辅助变量更新及 CVX 凸子问题，设置可行初始化、停止容差和最大迭代次数；记录原始约束残差、目标下降曲线、求解状态与耗时。
- [ ] 核对非聚合轮、切分切换轮和最终聚合轮的通信负载：分别处理各阶段最大时延，计入新前缀下发，最终轮下发量为零。
- [ ] 设置固定切分＋固定资源、固定切分＋资源优化、窗口切分＋资源优化三类基线；补充无窗口、忽略同步开销等消融。各方案统一预训练模型、数据释放表、轨迹、训练预算和随机种子。
- [ ] 同时记录系统总能耗、无人机／服务器能耗及各组成、训练完成时刻、每轮时延、单机剩余电量、切分轨迹与资源配置。
- [ ] 在统一验证集与测试集上报告损失、准确率及适用的分类指标；分别记录最终模型和按验证集选择的模型，测试集不参与模型选择。
- [ ] 分别统计不同样本数与累计样本处理次数 $N\sum_kD_k$。不同资源策略可能使用不同的轮初数据快照，比较能耗时同时报告这些差异；增加统一数据快照的对照实验以区分资源和数据效应。
- [ ] 改变 $T_{\max}$、电量、带宽、无人机数、采集速率、信道窗口和数据异质性，评估敏感性；采用多个随机种子报告均值与波动。
- [ ] 单独汇总未完成训练、超时、超电量及求解失败的情况，报告任务完成率，不将未完成方案的较低能耗计作节能收益。

# 参考文献

\[1\] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, 2016, pp. 770–778. [出版页](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). <mention url="https://app.notion.com/p/397427f94a0b8093a889ec0a27adaed5" />

\[2\] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in <i>Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS)</i>, PMLR, vol. 54, 2017, pp. 1273–1282. [出版页](https://proceedings.mlr.press/v54/mcmahan17a.html). <mention url="https://app.notion.com/p/3ac427f94a0b80869773f04aa928445d" />

\[3\] M. Ren, J. Chen, Y. Zhou, and L. Yang, “Exploiting UAV-emitted jamming to improve physical-layer security: A 3D trajectory control perspective,” <i>IET Communications</i>, vol. 15, no. 6, pp. 780–789, 2021. [出版页](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12120).

\[4\] Y. Mao, J. Zhang, and K. B. Letaief, “Dynamic Computation Offloading for Mobile-Edge Computing With Energy Harvesting Devices,” <i>IEEE Journal on Selected Areas in Communications</i>, vol. 34, no. 12, pp. 3590–3605, 2016. [DOI](https://doi.org/10.1109/JSAC.2016.2611964). [作者稿](https://ira.lib.polyu.edu.hk/bitstream/10397/96181/1/Dynamic_ComputatiOffloading_Mobile-Edge.pdf).

\[5\] S. Boyd and L. Vandenberghe, <i>Convex Optimization</i>. Cambridge University Press, 2004. [作者开放全文](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

\[6\] Y. Chen, L. Zhao, Y. Zhang, and K. Shen, “Inverse Quadratic Transform for Minimizing a Sum of Ratios,” in <i>Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)</i>, 2023. [作者稿](https://kaimingshen.github.io/doc/ICASSP23_FP.pdf).

\[7\] CVX Research, Inc., <i>The CVX Users' Guide</i>. [函数与凸性规则](https://cvxr.com/cvx/doc/funcref.html).

\[8\] X. Hou, J. Wang, J. Wang, J. Du, Z. Zhang, C. Jiang, and Y. Ren, “Energy-Efficient Federated Semi-Supervised Learning for UAV-Enabled Integrated Sensing, Computation, and Communication,” in <i>Proc. IEEE Int. Conf. Communications (ICC)</i>, 2025, pp. 6680–6686. [DOI](https://doi.org/10.1109/ICC52391.2025.11161423).

\[9\] Z. Zhai, X. Yuan, X. Wang, and G. Y. Li, “Decentralized Federated Learning With Distributed Aggregation Weight Optimization,” <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, vol. 48, no. 3, pp. 3899–3910, Mar. 2026. [DOI](https://doi.org/10.1109/TPAMI.2025.3640709).

<page url="https://app.notion.com/p/402427f94a0b8181ba40c71c42f7181e">
实验记录
</page>
</content>
</page>
