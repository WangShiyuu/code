# 连续飞行 UAV-SFL：动态切分、资源优化与训练执行

本目录实现固定轨迹飞行、持续采集样本的 UAV-SFL。模型由起飞前统一初始化，各机在飞行中执行前缀计算，服务器执行后缀计算；批末同步等待期间继续沿轨迹飞行和采集。正式入口默认完成 **100 epoch，每 2 epoch 聚合，共 50 次聚合**。周期切分与块资源优化已经接入实际训练，比较算法留待独立模块实现。

优化目标为已建模的训练计算与通信总能量最小化，使用独立校准预演确定宽松时限。电量目前仅统计，不作为硬约束。`code_v0` 保存原代码归档，已有数据划分和历史实验保持原样。

## 1. 文件分别负责什么

| 文件 | 作用 | 日常运行方式 |
|---|---|---|
| `config.py` | 统一物理、数据、训练、切分和求解器参数及其校验 | 修改参数或通过 JSON 覆盖，由入口读取 |
| `data.py` | ZIP/XML 解析、区域裁剪、场景隔离、IID 分配、缓存和采集快照 | 一般由 `prepare_scenario.py` 调用 |
| `environment.py` | 固定蛇形轨迹、空地信道、原单 CSV 场景生成和读取 | 由场景入口调用，无独立命令行入口 |
| `long_environment.py` | 长环境流式分片生成、有限缓存读取、独立校准信道 | 由动态训练准备阶段调用 |
| `prepare_scenario.py` | 数据与原 600 s 场景的统一准备入口 | `python -X utf8 -B prepare_scenario.py --stage all` |
| `example_read.py` | 检查位置、CSI、可用样本数量和索引读取 | `python -X utf8 -B example_read.py --with-data` |
| `sfl_model.py` | ResNet18、P0—P8 切分、画像、完整模型等权聚合 | 由训练入口调用 |
| `transmission.py` | Shannon 速率及旧诊断的跨 CSV 边界通信 | 由执行模块调用 |
| `dynamic_controller.py` | 周期逐机候选扫描、参考预演、固定逐轮与逐块预算 | 由动态训练入口调用 |
| `resource_solver.py` | CVXPY / Clarabel 块资源求解、原目标线搜索、原约束与 KKT 审计 | 由每个训练块和聚合块调用 |
| `scenario_preparation.py` | 画像、独立校准和长环境准备，供两类入口共享 | `train_sfl.py --prepare-only` |
| `run_output.py` | 时间戳、Git提交哈希及代码脏状态 | 两类入口共用 |
| `execution.py` | 实际资源的物理时序和能量核算，不求解、不更新网络 | 优化预演与训练共用 |
| `optimize_sfl.py` | 独立运行周期选择和块优化，保存决策 | `python -X utf8 -B optimize_sfl.py --prepared-run 运行目录` |
| `decision_replay.py` | 校验并流式读取优化结果，不调用求解器 | 训练的 `--optimization-results` 模式调用 |
| `dynamic_training.py` | 连接准备包、周期控制、块求解、梯度提交与实际成本台账 | 由 `train_sfl.py` 调用 |
| `train_sfl.py` | 正式统一入口；默认动态，兼容固定诊断 | `python -X utf8 -B train_sfl.py --save-checkpoint` |
| `report_results.py` | 从实际运行记录生成中文表格，兼容旧报告 | `python -X utf8 -B report_results.py --report 结果目录/summary.json` |

使用顺序：修改 `config.py` → 运行 `prepare_scenario.py` → 用 `example_read.py` 检查输入 → 准备动态长环境 → 启动训练 → 查看本次 `report.md`。数据与环境模块一般不需要逐个直接运行。各 Python 文件开头说明用途与运行指令，函数注释注明主要变量及单位。

使用 Python 3.10 及以上版本。数据准备需要 NumPy、Pillow；环境生成和 CSV 查询只依赖标准库；真实训练和画像需要 PyTorch，动态优化另需 CVXPY、Clarabel、SciPy，统一列在 `requirements-training.txt`。程序使用本地 CBCL 数据，不下载原始数据或预训练权重。

## 2. 在另一台电脑运行：完整 PowerShell 命令

在 VS Code 打开仓库并打开终端。已有 Conda 环境为 `sfl0416`，先确认已激活，再安装依赖：

```powershell
cd "D:\Wangshiyu\Work1SFL\code"
git pull --ff-only
conda activate sfl0416
python -c "import sys; print(sys.executable)"
cd code_v1
python -m pip install -r requirements-training.txt
```

数据目录应包含 `Images.zip` 和 `Annotations.zip`。首次使用或还没有本地缓存时运行：

```powershell
python -X utf8 -B prepare_scenario.py --stage all --data-root "D:\Wangshiyu\Work1SFL\DataSet\CBCL StreetScenes"
python -X utf8 -B example_read.py --with-data
```

完整匹配的缓存直接复用，不重新裁剪。上述命令保留原 600 s 场景，正式动态训练会在独立运行目录准备完整长环境。若只需要数据，可以使用 `--stage data`；动态准备本身不依赖旧的 `output/scenario/environment.csv`。

先只准备 100 epoch 的参考成本预演和正式长环境，**此步骤不进行 SGD 更新**：

```powershell
python -X utf8 -B train_sfl.py --prepare-only
```

程序打印 `Prepared: ...output_时间戳_git哈希`。用该目录先执行独立优化，再读取结果训练；下面自动选择名字最新的运行目录：

```powershell
$runDir = Get-ChildItem "output" -Directory -Filter "output_*" | Sort-Object Name -Descending | Select-Object -First 1
python -X utf8 -B optimize_sfl.py --prepared-run "$($runDir.FullName)"
python -X utf8 -B train_sfl.py --optimization-results "$($runDir.FullName)" --save-checkpoint
```

也可以把打印的具体目录写入命令。`optimize_sfl.py` 不读取训练图片、不执行梯度更新；它按模型画像及同一物理时钟输出 `optimization/optimization.json`、逐块资源和周期切点。训练使用 `--optimization-results` 后不重新选择切点、不运行 CVXPY，而是逐块核对时刻、CSI、样本载荷、预算和文件指纹，再执行网络更新。优化失败的结果不能进入正式训练。也可直接传入 `optimization/optimization.json`。准备与执行使用相同的控制参数；更改物理、训练或优化设置后重新准备。准备包会核对数据清单、初始化权重、画像、校准记录和环境参数的内容指纹。一个准备包只启动一次训练；当前没有断点恢复功能，重复实验重新准备即可。

保留直接在线调用方式；优化模块与训练模块仍各自独立，统一入口按块调用求解器：

```powershell
python -X utf8 -B train_sfl.py --save-checkpoint
```

先进行 4 epoch 的完整短运行，覆盖中间聚合和周期重选：

```powershell
python -X utf8 -B train_sfl.py --epochs 4 --save-checkpoint
```

`--epochs` 必须为聚合间隔的正整数倍。`--device auto` 默认有 CUDA 时用 GPU，否则用 CPU；可显式增加 `--device cpu` 或 `--device cuda`。求解器在 CPU 上执行。GPU 版本 PyTorch 按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装，已有适用版本无需替换。动态模式不接受 `--split` 或旧 `--scenario`，切点由控制器逐机选择，环境来自本次准备包。

保留原固定切点、参考资源的短诊断入口：

```powershell
python -X utf8 -B train_sfl.py --mode diagnostic --epochs 2 --split P4 --save-checkpoint
```

诊断使用旧场景和跨 CSV 节点传输规则，不施加动态逐轮预算。动态与固定诊断采用不同的 CSI 执行约定，不能直接把两者的成本差异解释成算法节能收益。

已完成的实验可重新生成表格，无需重训：

```powershell
python -X utf8 -B report_results.py
python -X utf8 -B report_results.py --report output/diagnostic/summary.json
```

不带 `--report` 时选择最新 `output/output_*/summary.json`；没有新版记录才回退旧诊断路径。命令行帮助分别用 `python train_sfl.py --help` 和 `python prepare_scenario.py --help` 查看。

## 3. 参数与默认设定

参数集中在 `config.py` 的 `CONFIG`，也可用 `--config 路径.json` 局部覆盖。JSON 中的相对路径相对于该 JSON 所在目录解析；命令行相对路径相对于当前工作目录。

| 配置组 | 主要内容 |
|---|---|
| `mission` | UAV 数量、区域、高度、速度、原采集任务时长、扫描间距、环境网格 |
| `data` | 数据与缓存路径、图像大小、场景划分、共同容量、初始缓存、固定采集速率 |
| `training` | epoch、聚合间隔、epoch 步长、batch 大小、SGD、固定 BN、初始化和设备 |
| `uav` / `server` | 计算频率、cycles/FLOP、有效电容、功率、并行分支与聚合开销 |
| `radio` / `channel` | 双向带宽、最低带宽、载频、噪声谱密度、损耗和衰落 |
| `selection` | 候选切点、初始切点、历史块观测窗口、候选扫描次数与平局容差 |
| `optimization` | 独立校准种子、期限倍率、分片大小、近端和 Armijo 参数及容差 |
| `energy` | 尚待标定的初始电量、固定飞行能耗、基础能耗与储备；当前无电量硬约束 |
| `theory` | 平均期望梯度精度预算的条件性参数界 |
| `output` | 原共享场景的准备目录 |

默认 5 架 UAV，x、y 工作范围各 −500～500 m，高度 100 m，服务器位于地面中心；速度 10 m/s，扫描间距 100 m，原采集时长 600 s，环境采样间隔 0.1 s。按 x 方向等宽分区，各机在自己的分区蛇形扫描，闭合后继续循环；默认一圈 2200 m、220 s。训练等待不会改变轨迹，也不添加悬停能耗。

轨迹是固定高度折线运动学模型，转角不另外模拟加速度、转弯半径或驻留。`scenario.json` 保存分区、航点、累计距离、单圈时间和总航程。扫描间距是几何设置，不代表已验证相机覆盖率。

默认 100 个本地 epoch、每 2 个 epoch 聚合，epoch 累计步长 0.18。实际批步长为 `epoch_step_size * batch_samples / epoch_snapshot_samples`。普通 SGD、固定 BN、IID 数据划分和原有参数更新规则保持一致。`theory_budget()` 可复算条件性理论预算；这里的 100 epoch 是任务设置，不能仅凭分类准确率确认理论平均期望梯度条件已满足。

## 4. 数据划分、采集快照与缓存

CBCL 任务为九类 polygon 外接框区域分类：bicycle、building、car、pedestrian、road、sidewalk、sky、store、tree。先按原始场景约 70%/15%/15% 隔离训练、验证和测试，再把训练区域无放回、等量 IID 分配给 UAV。不同 UAV 不持有相同区域样本，有限随机划分不强制类别比例完全相同。

当前本地原始包解析出 27,662 个有效区域样本，默认训练 19,365、验证 4099、测试 4198；5 架 UAV 各 3873 个训练样本。数字以实际生成清单为准。指定共同容量时按该容量分配；省略时取训练池大小除以机数向下取整。未选样本和余数在清单生成前排除并记录，不复制补齐。

`data.json` 的 `clients[n].indices` 是训练子集局部索引，其固定顺序就是到达顺序。每个 epoch 开始用 `snapshot_indices(time_s)` 取得当时已到达的前缀副本，打乱该副本后完整训练，尾批保留。epoch 内新增样本从下一 epoch 使用，旧样本持续保留；不设每张图片最低训练遍数。

默认初始缓存为每机容量的 10% 向下取整，即当前 387 张；剩余样本在原 600 s 采集设定下得到固定速率 **5.81 样本/s**。动态长环境只延长可查询时间，不重新降低该速率；达到 3873 张后封顶。末尾未进入快照的图片不额外补训。

改变控制器参数、截止时间或求解器设置不会使图片缓存失效。缓存复用仍核对原始包、预处理参数、数据指纹及 PNG 完整性。改变数据分配、UAV 数量或图像大小时使用相应的新缓存目录。

```powershell
# 仅在需要另一组数据/场景时运行，避免覆盖原配置。
python -X utf8 -B prepare_scenario.py --stage all --num-uavs 3 --samples-per-uav 1000 --data-output output/data_n3_1000 --output output/scenario_n3_1000
```

训练这组新配置时，还应通过同一份 `--config` 对齐机数、容量与数据路径；准备命令的局部参数不会自动改写 `config.py`。

## 5. 校准预演、长环境与信道读取

校准采用正式信道种子加 1，默认 **43**；正式环境种子为 **42**。预演使用正文周期切分评分、参考计算频率、参考上传功率及均分带宽，模拟完整采集、训练与聚合成本，**不更新神经网络**。第 e 个 epoch 固定预算为该轮参考时长的 1.2 倍，总期限为参考总时长的 1.2 倍。每轮独立使用预算，不临时挪用其他轮的份额。

正式训练前一次生成覆盖完整总期限的环境，终点向上对齐采样网格，轨迹持续循环。CSV 按完整共同时间组分片，单文件不超过 **20,000,000 字节（20 MB）**。`scenario.json` 的 `trace.parts` 保存文件名、首末时刻、记录数、字节数和 SHA-256。读取器仅缓存至多两个分片，支持跨分片随机查询，不把全部长环境载入内存；训练过程中不续写环境。

信道由距离与仰角决定 LoS 概率，自由空间损耗叠加 LoS/NLoS 额外损耗，再乘莱斯/瑞利功率衰落。默认 LoS 线性莱斯因子为 10，NLoS 取 0；上下行共用同一等效线性功率增益。固定种子只在生成时抽样，查询不重新抽样。

| CSV 字段 | 含义与单位 |
|---|---|
| `time_s`, `uav_id` | 物理时间 / s、零起始 UAV 编号 |
| `x_m,y_m,z_m` | 三维坐标 / m |
| `distance_m` | 本次观测到服务器的距离 / m |
| `los_probability`, `los` | LoS 概率、LoS 状态 0/1 |
| `path_loss_db` | 自由空间与状态附加损耗 / dB，不含小尺度衰落 |
| `channel_gain` | 包含衰落的线性功率增益 |

规则 0.1 s 网格与所有转角共同构成观测时间点。位置在相邻坐标间线性插值；CSI 使用不晚于查询时刻的最近观测。动态执行在每个训练块或聚合块开始时读取一次 CSI，**整个块内冻结，下一块重新观测**。传输可以跨越多个 CSV 时间点，该块仍使用块初 CSI；CSV 网格不是必须在其内完成传输的时限。

```python
from environment import EnvironmentTrace

# 旧单 CSV 与动态分片场景使用统一工厂接口。
trace = EnvironmentTrace.from_scenario("output/output_时间戳_git哈希/environment")
state = trace.state_at(12.345, 0)
position = state["position_m"]
gain = trace.channel_at(12.345, 0)
samples = trace.sample_count_at(12.345)
```

`window_mean(t, 20)` 为旧接口的最近 20 个 CSV 观测均值。动态控制器另行维护最近 **20 次块起始观测**，首周期使用初始 CSI，不能用 CSV 窗口替代。固定轨迹已知，因此位置插值可以读取下一航点；控制决策不读取未来信道或未来样本。

## 6. 周期决策、块求解与训练提交

每个周期按正文式（26）—（28）逐机扫描候选切点，周期内保持各机切点不变，平局保留原值。参考可行性只基于当前已知样本量、当前 CSI 和已分配预算；当前切点不可行时尝试逐机最快参考切分初值，仍不可行则明确停止。

每轮开始以当前快照、切点和轮初 CSI 估计参考批成本，按比例分配本轮固定预算；聚合单独分配，并按候选前缀最大载荷预留适用的下发开销。真正的带宽、频率和上传功率由每个块的优化器联合决定。

求解器使用 CVXPY 与 Clarabel；Shannon 速率采用相对熵表达进入指数锥。实现参数更新、带近端项的凸子问题，以及对原目标的 Armijo 回溯。默认近端系数 0.01、回溯比例 0.5、Armijo 系数 0.0001，最多 100 次外层迭代和 30 次回溯；方向、原约束、原问题 KKT 残差均按归一化变量的无穷范数核验，容差分别为 `1e-5`、`1e-6`、`1e-4`。参见 [CVXPY 求解器说明](https://www.cvxpy.org/tutorial/solvers/)。

优先热启动上一块资源，必要时执行可行性恢复。指数锥在平坦目标附近出现数值停滞时，以原目标梯度和原约束线性化的投影方向做额外停止核验；该小型凸QP只生成方向证书，不替代资源优化，不用缩小实际步长冒充方向收敛。只有通过原约束与收敛审计的块才进入模型更新；求解失败不会自动回退到固定 P4 或均分带宽。聚合零计算负载、末次聚合零下发负载不创建对应无效变量。

训练顺序为：前缀前向 → 激活和标签上传 → 后缀前后向 → 边界梯度下传 → 前缀反向。先求解并核验资源，再计算所有 UAV 梯度，确认梯度均有效后统一执行本批 SGD。较快分支在批末等待，物理时钟按最慢分支推进；本机依次执行各模型算子不作为模拟服务器排队时延。

不同 UAV 可以选择不同切点。完整模型权重和 Parameter 连续保留：聚合时上传旧前缀，完整模型等权聚合，再按下一周期新切点下发新前缀；末次聚合不再下发。初始统一前缀视为起飞前已下发。切点重选不重新初始化模型，也不强制制造切换。

## 7. 输出、失败处理与结果解释

```text
output/
├─ data/                         本机复用的 data.json 和 crops/
├─ scenario/                     原共享 600 s 单 CSV 场景
└─ output_时间戳_git哈希/
   ├─ preparation.json          准备包参数、来源与完整性校验
   ├─ calibration.json          独立信道预演、固定逐轮预算与总期限
   ├─ split_profiles.json       所有候选切点的计算量与通信画像
   ├─ environment/
   │  ├─ environment_00000.csv  每个分片不超过 20 MB
   │  ├─ environment_00001.csv
   │  ├─ scenario.json          完整索引、航点、固定采集速率和种子
   │  └─ patrol_map.svg
   ├─ optimization/             独立优化结果（采用分步运行时）
   │  ├─ optimization.json       优化状态、来源与完整性校验
   │  ├─ period_splits.json      周期切分结果
   │  └─ block_decisions.jsonl   逐块实际资源及求解诊断
   ├─ execution_started.json    标记此准备包已经启动过训练
   ├─ period_splits.json        周期候选扫描与每机切点
   ├─ block_decisions.jsonl     逐块资源、预算、负载与求解诊断
   ├─ summary.json              请求/完成轮数、状态、实际成本及失败原因
   ├─ report.md                 中文关键结果表
   ├─ key_metrics.csv           Excel 可直接读取的完整指标表
   └─ checkpoint.pt             指定 --save-checkpoint 时保存
```

目录格式为 `output_YYYYMMDD_HHMMSS_微秒_12位Git提交哈希/`。时间戳使用本机时间，精确到微秒；`source_git` 另记完整提交和代码是否有未提交修改。Git 不可用时明确标记 `nogit`。`--prepare-only` 先创建运行目录，`--prepared-run` 在同一目录完成训练，避免重复复制长环境。`--output` 指定父目录；历史结果不覆盖。报告、配置、长环境分片和可选模型允许随运行结果提交 Git，图片缓存和默认共享场景留在本机。

报告从 **实际已提交块** 汇总资源区间、通信字节、计算与通信时延和能量。参考资源只用于校准和切分评分，不显示成实际资源。逐块日志保留原目标下降、可行性、KKT 残差和求解状态；对称条件下可以得到相同分配，不要求带宽或切点人为出现差异。

求解、梯度或时限检查失败时，`run.status=failed` 并记录原因；失败块不计为完成训练。样本访问量、时间和能量只统计已提交批次。检查点保存最近一次完成聚合的完整模型，首次聚合前则保存统一初始模型。完成 epoch 数和完成聚合数分别报告，不以未聚合局部模型冒充最终全局结果。

系统模拟时延使用并行关键路径，分为 UAV 计算、服务器计算、通信与聚合，四项之和对应总时延。每机上下行和计算工作时长是设备工作量，不能简单累加当作并行系统完成时间。计算动态功率为 `kappa*f^3`，计算能量为 `kappa*C*f^2`，发射能量为功率乘发送时长；频率与功率是不同物理量。

求解和本机训练墙钟另行报告；读取独立优化结果时，训练阶段求解墙钟为零，独立优化耗时记入 `optimization_input.offline_solver_wall_s`。两类墙钟，不混入正文物理时延，也不需要真实等待或 `sleep`。成本是由 FLOP 画像、cycles/FLOP、频率、电容和信道推算的模型值，不是实际 UAV 电表测量。未知电池、固定飞行、接收、空闲等能量不按零补齐；当前不施加电量硬约束。

旧 `output/diagnostic/` 及其摘要、检查点保持原样。旧报告缺失新增指标时显示“未记录”，不从能量反推其未记录的时延。项目目录不保留临时测试脚本、构造数据或临时验证结果。

## 8. 已完成的验证范围

- 用真实缓存的微型配置完成4 epoch、2次聚合：5架UAV，初始32张，采集0.1张/s，batch 16，epoch累计步长0.01，候选P3/P5。该配置只用于验证执行链路，正式默认参数保持原值。
- 先独立优化再重放训练，训练阶段禁用求解与切分扫描后仍完整运行。样本快照为32、32、33、34张，含尾批；优化与训练的物理时长、能量账本一致。不同切点的四轮SGD/FedAvg另与完整网络逐项对照，验证Parameter及权重连续性。
- 检查原目标下降、原梯度/Jacobian有限差分、原约束及KKT、非对称信道带宽、零负载、末次无下发及不可行期限。旧报告重新生成后与原文件逐字节一致。
- 默认100 epoch仅执行成本预演与完整长环境准备，未运行100 epoch神经网络训练。当前参数参考时长约114492.132 s，期限137390.558 s，向上对齐至137390.6 s；实际生成41个CSV分片、约802 MB，全部小于20 MB，并验证边界、随机查询、有界缓存及5.81张/s采集速率。

临时脚本、环境和微型实验结果均已删除。成功块日志保留原目标历史、原约束/KKT/方向残差和实际资源，省略重复的全量乘子及逐次方向向量；失败块保留完整求解诊断。
