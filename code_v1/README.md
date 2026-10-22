# 连续飞行 UAV-SFL：数据、环境与训练诊断

本目录统一管理 CBCL StreetScenes 数据划分、样本采集顺序、固定巡逻轨迹、动态信道及首次 SFL 训练诊断。系统参数集中在 `config.py`；本次诊断规模及本机设备可通过训练命令行指定。

同级 `code_v0` 为原代码的完整复制归档，原路径没有改动。

## 1. 日常使用顺序与运行

1. **修改 `config.py`**：设定无人机数量、样本量、工作范围、飞行速度和信道参数等。
2. **运行 `prepare_scenario.py`**：统一准备数据、分配采集顺序，并生成轨迹和信道环境。
3. **运行 `example_read.py`**：查看任意时刻的位置、当前信道及可用样本数量，确认读取符合预期。
4. **运行 `train_sfl.py`**：先用固定切分及参考资源完成短训练诊断，观察损失、计算量、通信量、模拟时延和估计能耗。
5. **查看本次 `output/output_时间戳/report.md`**：集中查看切分点、样本数、计算与通信开销、带宽和功率；也可使用 `key_metrics.csv`。

`data.py` 和 `environment.py` 是由统一入口调用的具体功能模块，一般不需要逐个运行；其中 `environment.py` 没有独立命令行入口，直接运行该文件不会生成环境。`sfl_model.py` 提供模型与画像，`transmission.py` 执行分段通信，`train_sfl.py` 串联真实训练和模拟时钟，`report_results.py` 汇总运行记录。目前未接入切分点优化、资源分配优化和比较算法，比较方案后续放在单独代码中。

各Python文件开头均说明用途、输入输出和运行指令，函数内注释解释关键步骤与单位。

使用 Python 3.10 及以上版本。准备图片需要 NumPy、Pillow；环境生成和 CSV 回放仅依赖标准库。运行真实训练与模型画像时，另安装 `requirements-training.txt` 中的 PyTorch 依赖。首次训练的完整步骤见第7节。

```powershell
cd E:\Work1SFL\code\code_v1
python -m pip install -r requirements.txt
python -X utf8 -B prepare_scenario.py --stage all
python -X utf8 -B example_read.py
python -X utf8 -B example_read.py --with-data
```

原始目录应包含 `Images.zip` 和 `Annotations.zip`。默认向上查找 `DataSet/CBCL StreetScenes`，当前机器可自动定位；程序不会下载或修改原始包。

默认输出：

```text
output/
├─ data/
│  ├─ data.json
│  └─ crops/                  实际数据裁剪缓存
└─ scenario/
   ├─ environment.csv         共享物理时间环境
   ├─ scenario.json           参数、容量、航点和文件校验值
   └─ patrol_map.svg          工作范围和巡逻轨迹图
```

每次训练在同一 `output` 下自动创建一个新目录，不覆盖先前运行：

```text
output/
├─ data/                     可复用图片缓存与清单，留在本机
├─ scenario/                 可复用的共享环境输入，留在本机
├─ output_20261022_143000_123456/
│  ├─ summary.json           完整配置、训练与计量记录
│  ├─ report.md              中文关键指标表
│  ├─ key_metrics.csv        可用Excel读取的指标表
│  ├─ checkpoint.pt          仅指定 --save-checkpoint 时生成
│  └─ environment/           本次CSV、场景参数和轨迹图的归档副本
└─ output_20261022_150000_654321/
   └─ ...
```

时间戳以运行机器的本地时间生成，精确到微秒，重名时附加序号。每次训练目录内的报告、参数、环境副本及可选模型均允许Git提交；图片缓存、默认共享 `output/data/` 和 `output/scenario/` 仍忽略，避免在各电脑之间相互覆盖可复用输入。脚本只生成文件，使用VS Code源代码管理提交、同步结果。已有旧版 `output/diagnostic/` 不自动移动或删除。

同参数且完整的缓存可复用；已有目录中参数不一致、文件缺失或内容不一致时会报错，不覆盖旧实验。改变无人机数量、样本容量或环境参数时，请使用新的输出目录。

```powershell
# 示例：改变无人机数量及每机样本上限。只在需要这组配置时运行。
python -X utf8 -B prepare_scenario.py --stage all --num-uavs 3 --samples-per-uav 1000 --data-output output/data_n3_1000 --output output/scenario_n3_1000

# 分开准备与生成，使用相同的配置。
python -X utf8 -B prepare_scenario.py --stage data
python -X utf8 -B prepare_scenario.py --stage environment

# 已有code_v1数据清单时，环境阶段无需NumPy、Pillow或PyTorch。
python -X utf8 -B prepare_scenario.py --stage environment --manifest output/data/data.json
```

也可运行 `python -X utf8 -B data.py` 仅准备数据。完整参数见 `python prepare_scenario.py --help`。

## 2. 参数与默认场景

直接修改 `config.py` 的 `CONFIG`，或通过 `--config 路径.json` 局部覆盖；显式命令行参数最后生效。JSON 内的相对路径相对于该 JSON 所在目录解析，命令行路径相对于当前工作目录。

| 配置组 | 主要参数 |
|---|---|
| `mission` | 无人机数量、区域边界、高度、速度、任务时长、航线间距、环境采样间隔 |
| `data` | 原始目录、缓存目录、图像尺寸、场景划分比例、种子、每机容量、初始缓存及采集速率 |
| `training` | 总epoch、聚合间隔、epoch累计步长、batch大小、模型初始化、固定BN与普通SGD规则 |
| `uav` / `server` | 计算频率、cycles/FLOP、有效电容、功率和独立服务器分支 |
| `radio` / `channel` | 双向总带宽、最低带宽、载频、噪声谱密度、传播损耗和衰落参数 |
| `selection` | 切分候选、初始切分及历史信道窗口观测数 |
| `energy` | 初始电量、固定飞行能耗、识别/采集/基础运行能量、安全储备 |
| `theory` | 正文收敛预算中的条件性参数界 |
| `output` | 环境输出目录 |

默认场景为 5 架无人机，工作区域 x、y 均为 −500～500 m，高度 100 m，地面服务器位于 `(0,0,0)` m。飞行速度 10 m/s，任务持续 600 s，扫描间距 100 m，环境网格间隔 0.1 s。

参数中 `samples_per_uav=None` 表示按实际训练池等量分配；指定整数表示每架无人机共同的容量。`initial_samples_per_uav=None` 时取容量的10%向下取整，至少为1。`sample_rate_per_s=None` 时，将剩余样本均匀释放至任务结束；也可填写非负的固定采集速率。

服务器默认每架无人机一个独立后缀分支。`parallel_branches` 和服务器总下行功率上限的自动解析结果记录在 `scenario.json` 的 `resolved_hardware` 中。

`energy` 的尚未校准项为 `None`，不影响轨迹和信道生成。调用 `config.training_energy_capacity(config)` 核算训练可用电量前必须填全，计算为：初始电量减固定飞行能耗、固定运行能量和安全储备。不能把旧版悬停功率直接作为固定飞行能耗。

默认训练协议参数为100个本地epoch、每2个epoch聚合、epoch累计步长0.18、普通SGD和固定BN。批次实际步长为 `epoch_step_size * batch_samples / epoch_snapshot_samples`。`theory_budget()` 可复算正文的条件性100 epoch预算，但不证明给定参数界适用于实际网络。首次诊断不调用该预算，默认只完成一个聚合周期；通过模型副本自动统计画像，不要求预先填写 `split_profile_path`，也不下载预训练权重。

## 3. CBCL 数据划分与持续采集

CBCL 本项目任务为九类 polygon 外接框区域分类。27,662 是当前原始包中的有效区域样本数，不是独立原始街景照片数。

处理先按原始场景约70%/15%/15%划分训练、验证和测试，再从训练场景的区域中无放回选样本，随机等量分配给无人机。验证和测试场景始终固定，不参与采集流。有限IID划分不强制每架无人机类别比例完全相同。

默认种子42的现有数据得到19,365个训练区域、4099个验证区域和4198个测试区域；5架无人机各3873个训练样本。实际数量每次从原始数据解析，程序不硬编码这些统计。其他无人机数量无法整除训练池时，余数在生成清单前排除；显式容量不足训练池时，未选样本也不混入验证或测试。排除数量和样本ID记录在 `allocation_report`。

`data.json` 保留 schema 1、类别、归一化参数、样本列表、每机索引和指纹。`clients[n].indices` 始终指向筛选后的训练子集局部索引，其固定排列就是样本的到达顺序；不同无人机不重复分配同一个区域样本。

```python
import json
from pathlib import Path
from data import AcquisitionSnapshots, load_data

scenario = json.loads(Path("output/scenario/scenario.json").read_text(encoding="utf-8"))
source = AcquisitionSnapshots(scenario["manifest_path"], scenario["collection"])
epoch_indices = source.snapshot_indices(12.345)  # 所有无人机的前缀副本
uav0_indices = source.snapshot_indices(12.345, uav_id=0)
train, validation, test, full_indices, manifest = load_data(scenario["manifest_path"])
```

一个epoch开始时读取并冻结索引快照，完整遍历该快照，尾批不丢弃。epoch期间新样本加入下一epoch；已有样本保留。只能打乱快照副本，不能原地修改采集序列。初始缓存包含在最终容量中；末尾尚未进入快照的样本不额外补训。

缓存复用检查原始包的大小/修改时间、预处理参数、清单指纹及PNG完整性；清单指纹绑定参数、样本元数据和UAV映射，不是每个PNG像素的逐文件哈希。

## 4. 工作范围与轨迹

矩形沿x方向等宽分区，每架无人机在自己的分区内沿y方向往返扫描。扫描线数量为 `ceil(分区宽度 / lane_spacing_m)`，均匀置于各子带中心。末端以轴向折线连接首点，单扫描线时采用沿原线往返。所有线段按同一指定速度飞行，闭合后重复，无驻留训练段。

`scenario.json` 保存每条轨迹的分区、航点、累计距离、航点时间、单圈长度/时间、完整圈数及任务飞行距离。默认每机一圈2200 m、220 s，600 s内完成2圈并继续飞行，总航程6000 m。

轨迹为无障碍平面区域的固定高度折线模型，转角按瞬时方向变化处理，不模拟转弯半径和起降。扫描间距是几何设置，不表示已验证相机覆盖率。区域大小、飞行速度、机数与任务时长改变后，不保证任务内一定完成一整圈，可从输出的圈数检查。

## 5. 信道与 CSV

每个共同物理时间点由当前位置计算距离和仰角，按仰角生成LoS状态；自由空间损耗叠加LoS/NLoS附加损耗，再乘单位均值的莱斯/瑞利功率衰落。LoS线性莱斯因子默认10，NLoS取0。上下行共用同一等效功率增益。

CSV时间节点合并0.1 s规则网格、所有轨迹拐点及任务终点，按 `(time_s,uav_id)` 排序，编号为0到N−1。所有算法使用同一文件、同一物理时钟。

| 字段 | 含义与单位 |
|---|---|
| `time_s` | 环境观测时间，秒 |
| `uav_id` | 零起始无人机编号 |
| `x_m,y_m,z_m` | 三维位置，米 |
| `distance_m` | 观测时刻到服务器的距离，米 |
| `los_probability` | LoS概率 |
| `los` | 1为LoS，0为NLoS |
| `path_loss_db` | 自由空间与状态附加损耗之和，dB，不含小尺度衰落 |
| `channel_gain` | 含小尺度衰落的线性功率增益，无量纲 |

信道在共同观测节点生成，此后分段保持，LoS及小尺度衰落在节点间按当前简化模型独立抽样；不模拟多普勒相关性。采样间隔是仿真环境分辨率，不自动等于真实物理信道相干时间。

CSV不预写速率、功率、带宽、epoch或训练能耗。速率由当前信道和实际资源计算，预先固定这些值会将环境与待比较算法混在一起。

## 6. 读取与后续训练接入

```python
from environment import EnvironmentTrace

env = EnvironmentTrace("output/scenario/environment.csv")
state = env.state_at(12.345, uav_id=0)
position = state["position_m"]
gain = state["channel_gain"]
next_time = state["next_change_time_s"]
means = env.window_mean(12.345, window_size=20)
count = env.sample_count_at(12.345)
```

- 位置在相邻时间节点间线性插值；因为保留全部转角，与固定折线轨迹对应。
- CSI只取不晚于查询时刻的最近观测，不插值未来CSI。重复或乱序查询不改变结果。
- `channel_time_s` 是CSI观测时间，`distance_m`/`channel_distance_m` 是该观测的距离；`current_distance_m` 是查询时刻的实际位置距离。
- `window_mean` 对截至当前时刻最近W个CSV观测取算术均值，不是W秒窗口；不足W个时使用已有观测。
- `next_change_time_s` 表示下一信道更新边界，任务终点为 `None`。`transmission.transmit()` 在节点之间保持CSI，跨边界逐段累计传输比特，不将一次CSI用于整个长epoch。
- 越界时间、缺失UAV、重复行、非有限数和非正增益会报错，不静默外推。

`train_sfl.py` 共用物理模拟时间：epoch开始时冻结样本前缀，按计算、传输及同步模型的执行时间推进时钟；较快分支等待时继续飞行采集。本次资源使用固定参考值；后续资源决策只可使用决策时的当前或历史CSI。电脑实际运行耗时、图片加载和评价耗时不作为飞行模拟时间，也不需要 `sleep`。

环境预生成便于公平比较：不同策略按自身模拟时刻读取同一外部过程。其完成时间和实际纳入训练的样本量可以不同，需要共同报告；不能仅凭环境生成成功就宣称100 epoch已满足时限、电量或测试精度要求。

正式目录不保留临时测试脚本、测试数据或测试报告。`output/data` 是后续使用的真实数据缓存，`output/scenario` 是实际环境输入。

## 7. 在另一台机器运行首次训练

在 VS Code 中拉取仓库，打开 `code_v1` 目录的终端。先把原始 CBCL 数据放到默认位置，或修改 `config.py` 的 `data.root`。另一台机器首次训练需准备本地图片缓存和对应环境；每次训练结果内会保留环境副本，允许随结果一起上传Git。

```powershell
cd E:\Work1SFL\code\code_v1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -X utf8 -B prepare_scenario.py --stage all
.\.venv\Scripts\python.exe -X utf8 -B train_sfl.py
```

目录放在其他位置时修改第一行路径；后续命令均在 `code_v1` 内执行。不需要激活虚拟环境，也不需要修改 PowerShell 执行策略。VS Code 选择 `.venv/Scripts/python.exe` 解释器后，也可使用对应的 Python 运行按钮。

有 NVIDIA GPU 时，先根据 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 为该虚拟环境安装匹配的 CUDA 版 PyTorch。CPU 版可明确安装为：

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

默认 `device=auto`：可用CUDA时使用GPU，否则使用CPU；CPU也能完成诊断。显式指定GPU或改变诊断规模：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B train_sfl.py --device cuda --epochs 2
.\.venv\Scripts\python.exe -X utf8 -B train_sfl.py --epochs 4 --save-checkpoint
```

`--epochs` 必须是聚合间隔的正整数倍，默认值为一个聚合周期（当前2个epoch），与 `training.total_epochs=100` 的正式理论预算分开。所有无人机完整遍历每个epoch冻结的数据，不截取几个batch冒充完整epoch。默认初始化种子42、固定P4、参考机载频率1GHz、上行功率0.5W、每机上/下行各4MHz，具体读取配置，未执行资源优化。

不用填电池或固定飞行能量，不调用电量可行性检查，不施加正式epoch预算或任务截止时间。环境文件仍必须覆盖执行所需的物理时间；超过CSV终点将明确报覆盖不足，不冻结末行信道。需要更长环境时，增加 `mission.duration_s` 后另生成环境，并将原采集速率显式写入 `data.sample_rate_per_s`，以免默认自动速率随时长改变。更多epoch不保证现有600s环境仍够用。

每个训练批次按以下顺序执行：各机前缀前向 → 激活与标签上传 → 对应服务器分支后缀前后向 → 边界梯度下传 → 前缀反向及统一步长SGD更新。服务器分支在模拟中并行、批末同步；本机为节约显存依次执行各独立模型的算子，实际执行顺序不作为模拟排队时延。每个epoch开始使用共同物理时钟，因此同采集速率下各机样本数量一致。

通信载荷不要求在一个0.1s区间内完成。例如传输开始于1.05s，先按1.0s记录的CSI传至1.1s，未完成的比特再按1.1s的CSI继续，最后可在区间中途结束。整次传输的功率与带宽保持不变，不在每个CSV节点重新优化；仿真器读取后来发生的信道仅用于执行，不用于提前选择资源。激活和梯度按真实float32张量字节统计，标签为int64，无压缩、协议开销或重传。

初始化模型的首个前缀视为起飞前已统一下发。每次聚合只上传各机前缀状态（包括BN buffers），服务器持有独立后缀，拼接后对完整模型等权聚合。非末次聚合再下发前缀；末次以服务器得到最终完整模型为终点。等待不改变巡逻轨迹，不增加悬停能耗。

每次运行自动创建 `output/output_时间戳/`，其中 `summary.json` 保存完整运行记录：

- `split_profiles`：P0—P8四段前后向FLOPs、每样本激活/梯度/标签字节、前缀状态字节。
- `epoch_records` / `aggregation_records`：各epoch快照样本数、尾批、损失、准确率、物理时间和聚合记录。
- `validation_before` / `validation_after`：同一固定验证子集（默认256张）的训练前后表现；测试集未参与。
- `physical_execution`：模拟时延、各机计算/发射能量、服务器能量、通信载荷及跨CSI节点次数。`critical_path` 将总模拟时延分成无人机计算、服务器计算、通信、聚合四项，沿每个并行块的最慢分支累计，四项之和为总时长。
- `wall_timing`：本机墙钟耗时，与模拟时延分开；`parameter_change_l2` 检查模型实际发生更新。
- `config` / `inputs` / `run`：完整参数、数据与CSV指纹、设备和协议；不会把未知电量填为零。

`report.md` 和 `key_metrics.csv` 自动汇总epoch、切分点、各轮样本数、各机带宽、计算频率/功率、上传功率、计算/通信时延及能耗。频率单位为Hz；计算动态功率根据 `kappa * f^3` 得到，不能把1GHz当作1W。每机上/下行累计工作时间是该链路的工作量，不将5条并行链路的时长相加作为系统完成时间。`barrier_wait_s` 只记录批末和聚合同步等待。

重复执行同一训练命令会自动生成新的时间戳目录，无需手动改名。`--output` 指定**父目录**，例如 `--output output/my_experiment` 会写到 `output/my_experiment/output_时间戳/`。仅添加 `--save-checkpoint` 才保存 `checkpoint.pt`；该文件可作为以后统一初始化的 `initialization_checkpoint`，当前不提供恢复中断训练的功能。

已经完成的旧版训练也可以生成关键指标表，无需重训：

```powershell
python -X utf8 -B report_results.py --report output/diagnostic/summary.json
```

不带 `--report` 时优先汇总默认 `output` 下最新时间戳目录中的 `summary.json`，没有新目录时回退到旧版路径。旧记录缺少新增的关键路径分项时，表格显示未记录；原有每机计算、上传、下行及能耗统计仍可读取。

FLOPs来自计数器支持的算子，不能覆盖所有内存与算子开销。模拟时间由配置的cycles/FLOP和频率得到；估计能耗由有效电容及发射功率得到，均不是另一台机器或真实UAV的测量值。短诊断只用于观察参数和执行流程，不证明100epoch收敛、任务可行或优化算法优于其他方法。训练出现非有限损失/梯度会明确终止，应据诊断调整步长，不自动加入裁剪或其他未定义的优化器。
