# 动态 UAV-SFL · code_v2

## 运行指令（Windows PowerShell）

新电脑上的项目目录为 `D:\Wangshiyu\Work1SFL\code`。先安装 Python 3.10 或更高版本，然后在 PowerShell 中进入 `code_v2`，首次运行时在本目录创建虚拟环境并安装依赖。

```powershell
cd 'D:\Wangshiyu\Work1SFL\code\code_v2'
python -m venv .venv
$sflPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $sflPython -m pip install -r .\requirements.txt
```

以后新开 PowerShell 时，从项目目录重新设置解释器变量。以下命令可分别运行，用 `config` 核对 `data.root` 和 `output_root`，用 `prepare` 验证两个原始 ZIP 包能读取，再用 `check` 和 `smoke` 检查求解器及训练流程。

```powershell
cd 'D:\Wangshiyu\Work1SFL\code\code_v2'
$sflPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $sflPython -X utf8 -B .\uav_sfl.py config
& $sflPython -X utf8 -B .\uav_sfl.py prepare
& $sflPython -X utf8 -B .\uav_sfl.py check
& $sflPython -X utf8 -B .\uav_sfl.py smoke
```

在新电脑复核运行时，使用 `verification_config.json` 做一次真实 CBCL 四轮成本仿真。该配置保留正式实验的四机硬件与切分策略，并将采样间隔固定为 13.574144486692015 秒，使短实验沿用 120 轮主实验的采样频率。

```powershell
$runDir = Join-Path '.\output' ('verification_release_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $sflPython -X utf8 -B .\uav_sfl.py simulate --config .\verification_config.json --output $runDir
Get-Content -Raw -Encoding UTF8 (Join-Path $runDir 'summary.json')
```

使用本目录已准备的 ResNet18 权重进行正式训练：

```powershell
& $sflPython -X utf8 -B .\uav_sfl.py train --checkpoint .\resnet18-f37072fd.pth
```

仅计算资源配置、物理时延和能耗时，选择：

```powershell
& $sflPython -X utf8 -B .\uav_sfl.py simulate
```

以上命令**不需要示例 JSON**，直接读取 `uav_sfl.py` 的 `default_config()`。`-X utf8` 使用 UTF-8 编码，`-B` 避免生成 Python 字节码缓存；`--checkpoint` 是本地预训练文件的位置。各命令可独立运行，不必先执行 `prepare` 或 `profile`，主流程会生成所需数据结构。

`smoke` 使用四机、4 epoch、32×32 合成图像执行真实反向传播、资源优化和两次完整聚合。`smoke` 和 `check` 使用同质设备诊断参数和 `reference_score` 策略。正式 `train` 和 `simulate` 默认使用下表的四种设备配置，计划执行 120 个本地 epoch（60 次聚合）。`train` 使用 CBCL 数据和指定的预训练骨干训练九分类模型，`simulate` 计算资源和物理成本。

正式实验默认使用 `optimization.split_objective="optimized_window"`，每次决策比较各组合分别分配资源后的预测窗口总能耗。固定参考评分对照使用 `"reference_score"`。

在这台电脑上，默认结果目录是 `D:\Wangshiyu\Work1SFL\code\code_v2\output`。`simulate` 和 `train` 为每次运行新建子目录，终端会打印实际位置。子目录中的 `config.json` 是此次运行最终生效的完整参数快照，`summary.json` 记录完成状态和失败原因，`history.csv` 记录逐轮指标，`decisions.jsonl` 记录候选筛选和求解过程。需要自行指定此次结果目录时，在子命令后追加 `--output .\output\run_001` 并使用新的目录名。PowerShell 的 `$LASTEXITCODE` 为 0 表示成功，1 表示输入或执行错误，2 表示任务未完成。

其他入口按需运行：

| 用途 | 在上述 PowerShell 会话中执行 | 主要结果 |
|---|---|---|
| 查看参数 | `& $sflPython -X utf8 -B .\uav_sfl.py config` | 打印完整参数和训练预算，不训练 |
| 验证实现 | `& $sflPython -X utf8 -B .\uav_sfl.py check` | 梯度、约束、尾批和聚合检查报告 |
| 准备数据 | `& $sflPython -X utf8 -B .\uav_sfl.py prepare` | 独立场景划分、客户端序列和释放表 |
| 生成画像 | `& $sflPython -X utf8 -B .\uav_sfl.py profile` | 五个切分点的计算/通信画像 |
| 查看帮助 | `& $sflPython -X utf8 -B .\uav_sfl.py --help` | 子命令、路径和覆盖参数说明 |

## 项目与环境

所有参数、数据准备、环境、优化、ResNet18、训练和检查均在 **`uav_sfl.py` 一个代码文件**。没有项目内部跨文件依赖，也不导入 v0/v1。实现依据为 [Notion 动态UAV-SFL](https://app.notion.com/p/3b5427f94a0b8040a213d5d9b277c8df)，源页面编辑时间为北京时间 **2026-10-30 02:45:37**，完整读取快照保存在 `notion_source.md`。

默认使用 CPU。若选择 GPU，在 `code_v2\.venv` 中安装与设备匹配的 PyTorch，并在 `training.device` 中指定设备。

## 可选参数覆盖

需要在多组实验间切换参数时，可以使用 JSON 局部覆盖。以下命令使用本目录虚拟环境中的解释器。

```powershell
# 按场景准备数据清单、等量客户端序列和完整释放时间表
& $sflPython -X utf8 -B .\uav_sfl.py prepare --config .\example_config.json

# 生成五个切分点画像
& $sflPython -X utf8 -B .\uav_sfl.py profile --config .\example_config.json

# 仅仿真资源/能耗，不更新神经网络参数
& $sflPython -X utf8 -B .\uav_sfl.py simulate --config .\example_config.json

# 保持默认的上下行各 20 MHz，每个组合分别求预测窗口资源，再选能耗最低的可行组合
& $sflPython -X utf8 -B .\uav_sfl.py simulate --config .\optimized_window_example.json

# 带宽敏感性对照：将上下行共享带宽分别设为 10 MHz
& $sflPython -X utf8 -B .\uav_sfl.py simulate --config .\bandwidth_10mhz_example.json

# 在同一份覆盖配置下进行真实训练，仍显式提供预训练 checkpoint
& $sflPython -X utf8 -B .\uav_sfl.py train --config .\example_config.json --checkpoint .\resnet18-f37072fd.pth

# 显式启用参考资源重建扩展，4 epoch 物理成本诊断
& $sflPython -X utf8 -B .\uav_sfl.py simulate --config .\reference_rebuild_example.json
```

`default_config()` 是唯一默认参数表；`example_config.json` 只是局部覆盖示例。相对数据路径和 checkpoint 路径相对配置文件，默认输出锚定本目录。`--output` 指定本次实验的新目录，已有实验不覆盖。

覆盖优先级为：`default_config()` → `--config` JSON → 显式命令行参数。比如 JSON 中 `checkpoint` 为 `null`，而命令行提供了 `--checkpoint`，则以命令行路径为准。`smoke` 还会覆盖自己的诊断规模、步长、随机初始化及模型保存选项；`check` 使用固定的小规模验证配置。

正式 `train` 默认要求提供预训练模型，不会静默下载、随机初始化或冒称已有预训练权重。支持标准 ResNet18 的 `state_dict`、`{"state_dict": ...}` 或本程序保存的 `{"params": ..., "buffers": ...}`；ImageNet 分类头维数不符时，保留预训练骨干并用统一种子重建九分类头，记录 `head_reinitialized`。v1 自定义 `residual_blocks.block*` 格式未做自动转换。只有显式 `--allow-random-init` 才允许随机初始化实验。

本目录现已准备 `resnet18-f37072fd.pth`，来自 [PyTorch 官方 ResNet18 IMAGENET1K_V1](https://download.pytorch.org/models/resnet18-f37072fd.pth)，文件大小 46,830,571 字节（约 44.66 MiB）。SHA-256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`，已匹配官方文件名中的哈希前缀。它在 ImageNet 上预训练，CBCL 九分类头仍需本研究训练。该官方文件采用旧 BN 格式，加载器按 PyTorch 兼容规则为缺失的 `num_batches_tracked` 补零，保留所有预训练权重和 BN 统计量，并记录到初始化来源中。

## 四架无人机的资源设定

数组按 UAV 1 至 UAV 4 排列，对应代码下标 0 至 3。下表是仿真场景假设，实验时可替换为具体设备的训练预算与测量值。[四机配置](heterogeneous_example.json)

| 无人机 | 训练电量上限（J） | 计算功率上限（W） | 最高频率（GHz） | 每 FLOP 所需周期 | 上行发射功率上限（W） |
|---|---:|---:|---:|---:|---:|
| UAV 1 | 12,000 | 0.1000 | 1.0 | 1.000 | 0.3 |
| UAV 2 | 16,000 | 0.1728 | 1.2 | 0.750 | 0.5 |
| UAV 3 | 20,000 | 0.3375 | 1.5 | 0.500 | 0.8 |
| UAV 4 | 24,000 | 0.5832 | 1.8 | 0.375 | 1.0 |

- `uav.energy_cap_j` 是预留给训练的可用电量。各机独立累计本地计算和上行发射能耗，逐轮扣减本机余额。
- `uav.compute_power_max_w` 与 `uav.kappa` 通过 `f_max = (compute_power_max_w / kappa)^(1/3)` 决定本机频率上限。示例中四机共用 `kappa = 1e-28`。`uav.cycles_per_flop` 将前缀 FLOPs 换算成本机所需周期。相同切点、相同样本量下，每 FLOP 的周期数越少、可用频率越高，纯计算时间就越短。
- `uav.power_max_w` 限制各机的上行发射功率。求解器每轮联合分配频率、发射功率和带宽。各机的硬件配置影响切点的可行性与能耗，四机可以选出相同或不同的切点。

每个 `uav` 字段均可填一个正数，表示所有无人机共用，也可填长度等于 `num_uavs` 的正数列表。显式配置的列表长度应与机数一致。旧式局部配置仅修改机数且机数不为 4 时，未显式配置的硬件字段沿用原同质默认值。在 `code_v2` 目录、已设置 `$sflPython` 的 PowerShell 会话中运行：

```powershell
& $sflPython -X utf8 -B .\uav_sfl.py config --config .\heterogeneous_example.json
& $sflPython -X utf8 -B .\uav_sfl.py simulate --config .\heterogeneous_example.json
```

运行后在 `config.json` 查看输入上限，在 `decisions.jsonl` 查看逐机实际分配的 `resources.frequency_hz`、`resources.power_w`、轮初和轮末电量 `remaining_before_j`、`remaining_after_j`，并与当轮 `split` 对照。

## 函数式结构及返回值

Model 是参数 Tensor 和固定 BN buffers 的字典。前向函数只读取它；`sgd_step` 和 `aggregate_models` 产生新字典/参数，不原地修改输入模型。没有自定义 `nn.Module`、训练器类或隐藏优化器状态。文件读取、写结果和打印集中在 I/O、主流程及 CLI 函数；`run_experiment` 会设置 PyTorch 线程数。

| 层次 | 主要函数 | 返回值 |
|---|---|---|
| 配置 | `default_config`, `load_config`, `validate_config` | 独立的嵌套 `dict` |
| 训练预算 | `training_budget(config)` | `epochs, aggregations, conditional_epochs, mode` |
| 数据 | `prepare_data(config)` | 记录、客户端索引、验证/测试索引、释放表、指纹组成的 `Data` |
| 轮初冻结 | `epoch_snapshot(data, t, k, config)` | `count, indices[N][D_k], time_s, mode` |
| 环境 | `channel_at(t, config)` | 位置 `[N,3]`、距离/仰角/LoS概率/增益 `[N]` |
| 模型 | `initialize_model(config, require_pretrained)` | `(Model, provenance)` |
| 前后向 | `forward_prefix`, `forward_suffix`, `split_gradients` | 激活、logits，或 `(完整梯度字典, 批次统计)` |
| 参数更新 | `sgd_step(model, gradients, step)` | 新 `Model` |
| 单机训练 | `train_client_epoch(...)` | `(新 Model, loss/accuracy/samples/step_sum/通信量)` |
| 聚合 | `aggregate_models(models, counts)` | 新的完整全局 `Model` |
| 画像 | `profile_splits(model, config)` | 按切点 `[0,2,4,6,8]` 排列的五列画像 |
| 原始成本 | `evaluate_resources(problem, resources, profile, config)` | 阶段时延、每机能耗、服务器能耗及组成 |
| 约束检查 | `audit_resources(...)` | `ok, normalized_violation, residuals` |
| 切分决策 | `select_splits(...)` | 候选向量列表、参考评分矩阵 `[N,5]`、各组合在所选目标下的筛选结果 |
| 资源优化 | `solve_resources(...)` | `ok, status, resources, metrics, trace, initialization, wall_s` |
| 主流程 | `run_experiment(config, train=False, ...)` | `summary, history, final_model, best_model, branches, output_dir` |

```python
import uav_sfl as s

cfg = s.load_config("example_config.json")
# 也可以在内存修改参数；函数将返回深拷贝。
cfg = s.validate_config(s.merge_config(cfg, {
    "training": {"epochs": 4},
    "optimization": {"strategy": "fixed_opt", "fixed_split": 4},
}))
result = s.run_experiment(cfg, train=False)
print(result["summary"]["completed"])
print(result["summary"]["system_energy_j"])
```

数组按 UAV 编号 0..N−1 排列。代码头部列出所有核心数据结构，各函数 docstring 说明输入和返回。`Model.params` 与 `Model.buffers` 中的 Tensor 作为只读输入使用；调用方也不应原地改写共享 buffers。

阅读代码时，可以沿着下面的调用顺序对照注释：

1. `main` 读取参数并选择模式；`run_experiment` 准备数据、初始模型及画像。
2. 每轮先由 `epoch_snapshot` 和 `channel_at` 冻结本轮数据与信道，再由 `select_splits` 在适用的聚合边界生成切点候选。
3. `make_problem` 明确本轮与下一周期的切点、全部载荷和剩余电量；`solve_resources` 找可行初值并交替求凸子问题；`audit_resources` 用原公式核验。
4. 可行资源确定后，`train_client_epoch` 返回各分支新模型；偶数轮通过 `aggregate_models` 得到全局模型，并用验证集选择最佳模型。
5. 本轮执行成功后才推进物理时钟、扣除电量、更新已完成信道窗口和下一周期切点，再进入下一轮。

三个容易混淆的量在代码中分别保存：`count` 是**每机本轮训练样本数**，`processed_samples` 是**所有轮次所有机器的累计处理次数**；`duration_s` 是**系统模型预测时延**，`training_wall_s`/`solver_wall_s` 是**本机执行耗时**；`split` 是**本轮切点**，`next_split` 是**本轮聚合后要下发的新前缀切点**。最后一轮的 `next_split=None`，避免额外下发。

资源优化结果应先读 `ok` 再取 `resources`。`ok=True` 表示当前返回点通过原约束检查；是否已收敛还要看 `status`。例如 `iteration_limit_feasible` 可以执行当前轮，但只表示达到迭代上限时仍可行。完整任务是否完成应看 `RunResult.summary.completed`，不能用某一轮的 `ok` 代替。

## 与 Notion 的对应

- 式(1)：每个 epoch 用 `D(t_k)` 的索引前缀。默认在物理时间 0 以已有巡逻样本启动。各机按固定秒数定时采样，轮中到达的样本在下一轮快照中可用。
- 式(2)—(5)：标准 ResNet18 的八个残差块按顺序两两分组。切点编号表示客户端执行的残差块数：`0` 在 stem 后，`2、4、6、8` 分别在 layer1、layer2、layer3、layer4 后。切点 8 的池化和分类头仍由服务器执行。每架 UAV 保留独立完整参数字典，其前缀/后缀由函数互补执行。每两轮进行完整模型平均，切换切点保留参数。
- 式(6)—(8)：预定偏心圆巡逻轨迹随物理时间变化，使用概率加权的**线性功率增益**；无旧版随机小尺度衰落。轨迹是可替换的场景输入。
- 式(9)—(17)：UAV/服务器前后向计算、激活+int64标签上传、梯度下传、旧前缀上传、新前缀下发、聚合开销均计入。三个阶段分别取最大时延后相加；最终聚合不下发。
- 式(19)：普通 SGD，实际批次步长为 `eta * batch_size_actual / D_k`，尾批同样计入。BN 统计量固定、仿射参数可训练。不使用动量、衰减、随机增强或混合精度。
- 式(23)：默认 `K_t=120` 为经验预算。`A_bar=3.85,C_bar=0.0115,epsilon=0.05` 得到条件性 100 epoch，但代码不会据此宣布理论保证。`budget.mode=theoretical` 要求显式设置有效界已经验证。
- 式(25)—(26) 对应 `reference_score` 模式。窗口只含已完成轮次信道，首次决策采用初始信道。该模式穷举五个切点的笛卡尔积，按参考资源筛选时限和电量，再按参考能耗排列候选。默认 `optimized_window` 的评分步骤见下节。
- 式(28)—(31)：CVXPY 用相对熵表示 Shannon perspective、按阶段引入上图变量；逆二次变换更新上行能量上界。频率/带宽使用比例变量，速率使用 Mbit/s，目标和电池约束做数值缩放。求解后必须重算原始约束及真实能耗，失败不会静默执行参考资源。

若第一次固定参考资源不可行，初始化在若干固定发射功率下求凸 Phase-I。未找到可行点的状态是 `feasible_initialization_not_found`，不是全局不可行证明。旧配置中的 `max_initial_candidates`、`max_coordinate_sweeps` 和 `score_tolerance_j` 在加载时忽略。逆二次迭代达到上限但原约束可行时返回 `iteration_limit_feasible`，不冒称已收敛或全局最优。

## 比较各切分组合优化资源后的能耗

默认 `optimization.split_objective="optimized_window"` 对同一预测窗口内的每个候选单独分配资源。首轮比较第 1、2 轮；在非最终偶数轮 k，比较 k、k+1、k+2 轮，并计入当轮下发新前缀的成本。所有候选共用当前样本数、窗口信道和同一组逐机剩余电量。对每个候选按时间顺序求单轮资源、扣除电量，再累加原公式的系统能耗。实际执行仍用当轮信道审计和求解。

四机每次决策检查 625 种组合，参考筛选失败的组合也会尝试求解。日志中的 `forecast_energy_j` 为成功解的预测累计能耗；空值与对应状态表示求解未成功。连续求解是逐轮迭代，因此这个排名只比较获得的可行解。示例使用 6 个并行进程，完整的 120 轮有 60 次切分决策。

仅复核已有仿真的一次决策时，可运行 `python -B compare_split_energy.py <已有运行目录> --epoch 1 --workers 6`。结果按能耗升序写入该目录的 `optimized_split_comparison_epoch_1.csv`，失败组合保留在表尾。

## 已发现的参考筛选边界与可选扩展

严格版本使用最近一次可行资源作为下一次 P1 的固定参考。在旧版恒速释放（每机 1 样本/秒）的四机 CBCL 仿真中，首轮每机处理 555 个样本，耗时约 600 秒。第 2 轮样本增长后，625 个切分组合在上一轮参考资源下均超过预测时限，运行返回 `no_reference_feasible_split`。这个状态只说明给定参考资源下的 P1 筛选结果。

在 `reference_score` 模式下，`optimization.reference_recovery="rebuild"` 允许在所有组合均被筛掉后，根据窗口平均信道重建参考资源，重新运行 P1，再按当前信道求解 P2。日志记录原失败状态和重建结果。复现严格筛选时，将 `optimization.reference_recovery` 设为 `"none"`。

## 数据、基线及参数标定

将 `Images.zip` 和 `Annotations.zip` 放到 `D:\Wangshiyu\Work1SFL\DataSet\CBCL StreetScenes`。默认数据路径根据 `uav_sfl.py` 所在目录解析为 `..\..\DataSet\CBCL StreetScenes`；整棵项目目录迁移时，保留 `code` 和 `DataSet` 的平级结构即可。每个 polygon 的外接矩形是一个九分类区域。先按原始街景隔离训练、验证和测试集，再将训练区域 IID 等量、不放回分给无人机，余数放入 `unused_train`。80/10/10 指独立场景比例。

本机实际读取到 3,547 个场景、27,662 个有效区域，跳过 5 个删除/空标注区域。默认四机划分为每机 5,552 个训练区域，共 22,208 个，验证 2,751 个、测试 2,700 个，剩余 3 个训练区域不用。初始每机释放 555 个。数据不会被复制为 Notion 中尚待核对的 27,000 样本示例。

### 巡逻过程中如何获得新样本

首轮每机已有 555 个区域。四架机以 10 m/s 匀速飞行，使用相同的采样间隔 Δ 和每次新增的有效区域数 B。每经过 Δ 秒，新的一批立即进入本机库存。在物理时刻 t，每机可用区域数为 `min(容量, 启动库存 + B × floor(t / Δ))`。各机的首批新增区域在 t = Δ 入库，最后一批取剩余数量。

默认 `data.samples_per_capture=1`，`data.sample_interval_s=null`。程序按规划时段 `(epochs − 1) × patrol_epoch_s × patrol_collection_fraction` 除以剩余批数，算出固定间隔。120 轮的规划时段为 67,830 秒，每机剩余 4,997 个区域，每 13.574 秒新增 1 个，相当于每飞约 136 米获得 1 个有效训练区域。该间隔为仿真假设，可按巡逻实测数据标定。区域索引来自各机预先划分的不重复序列。

固定采样设备参数时，可以直接设置 `data.sample_interval_s` 和 `data.samples_per_capture`。例如分别设为 27 和 2，表示每机在第 27、54、81 秒各添加 2 个区域。显式间隔独立于训练轮数。四机的库存数量同步，每机使用自己的区域索引。训练只在轮初读取一次库存。`data_manifest.json` 中的 `release_plan` 记录实际间隔与批量，`release_s` 给出各样本的入库时间，`history.csv` 的 `count_per_uav` 记录每轮实际使用的数量。

更改训练轮数会相应调整规划时段。比较不同轮时限或资源策略时，固定轮数、名义轮长及采集比例，即可共用同一释放表；实际完成较快的策略在同一轮次可能看到较少样本。把 `data.rate_per_s` 设为正数会切回旧的恒速释放，并等到初始样本攒齐后才开始训练。默认该值为 `null`。

`data.partition="dirichlet"` 支持类别异质性，并强制等客户端容量；容量约束会改变最终分布，实际类别计数保存在清单。更换独立划分单位时需调整 `prepare_data`，不能直接对同场景区域随机切分 holdout。

| strategy | 切分 | 连续资源 |
|---|---|---|
| `fixed_fixed` | 固定 `fixed_split` | 固定参考资源，仍检验时限/电池 |
| `fixed_opt` | 固定 `fixed_split` | 逐轮优化 |
| `dynamic_opt` | 最近 W 个已完成 epoch 平均信道 | 逐轮优化 |
| `no_window` | 最近 1 个已完成 epoch | 逐轮优化 |

多种子、带宽/电池/采集率等敏感性可用配置循环调用 `run_experiment`。不同策略必须共用数据划分、释放表、模型 checkpoint 和种子。正常模式允许不同物理执行时间导致不同 `D_k`，结果同时统计唯一已用样本量与累计处理次数。`data.fixed_epoch_counts=[...]` 可提供共同的非减轮次样本数，属于**统一快照消融**，此时明确绕过物理释放条件，不能混作主实验。

图像尺寸默认 64；本地批量大小、SGD 步长需在验证集调参。硬件 `kappa`、`cycles_per_flop`、服务器每分支频率、带宽、LoS 参数、轨迹、电池和 600 秒单轮时限都是场景假设。画像通过真实前后向 FLOP 计数乘配置的周期换算系数获得，**不是设备实测周期/能耗**；未被计数器支持的算子、SGD 更新和内存/I/O开销未计入。实际测量画像可通过 `run_experiment(..., profile=measured_profile)` 传入同结构数据并标明元数据。五列画像的列顺序为 `[0,2,4,6,8]`；传入旧版按块数 0 到 8 排列的九列画像时，程序提取五个对应列。

异构设备自动生成的 `client_cycles` 可为 `[N,5]`，行对应无人机，列对应五个切点。导入实测画像时，二维 `client_cycles` 按对应行取值，一维数组作为所有无人机共享的已知周期数。导入的周期数直接用于成本计算。

`prefix_bits` 计可训练参数；固定且各分支相同的 BN buffers 已预部署，不重复上传。服务器频率和下行功率均按每条并行分支解释。正资源下界 `positive_floor=1e-5` 是数值近似，带宽/频率/功率的可行域因此比严格开区间略小。功耗仅为通信/计算模型预测，本机串行执行时间、真实硬件功耗不混入系统模型。

## 输出与验证

每次输出到 `output/<模式>_<北京时间>_seed.../`：

- `config.json`、`data_manifest.json`、`split_profiles.json`、`initialization.json`：全部输入、样本释放和来源。
- 新版 `simulate` 和 `train` 运行保存 `uav_sfl_snapshot.py`，即启动本次实验的主程序源码。配套 `runtime.json` 记录 Python、操作系统平台、依赖包版本和源码 SHA-256；同一摘要也写入 `summary.json` 的 `code_sha256`。归档中早期运行使用的是当时的代码与参数，跨版本比较时先核对各目录的 `config.json` 和 `summary.json`。
- `decisions.jsonl`：每轮样本索引、切分候选/接受拒绝日志、资源、信道、载荷、能耗组成、原约束残差、优化目标轨迹、求解耗时。
- `history.csv`：便于后续绘图的逐轮指标。
- `summary.json`：完成状态、能耗、物理完成时刻、电池、累计样本处理数和失败原因；真实训练另有验证与测试指标。
- `last_aggregated_model.pt`、`best_validation_model.pt`：可选保存，验证集选择最佳模型，测试集仅在任务末评价。未完成时 `final_model=None`，最近聚合模型与未聚合分支不混淆；必要时另存 `partial_branches.pt`。

命令退出码：0=成功，1=输入/执行错误，2=任务未完成。未完成任务的低能耗不能作为节能收益。

已执行检查包括五点梯度及单步更新一致性、输入参数不变、尾批步长、加权聚合、快照冻结、数据互斥、同步载荷、最终零下发、原始约束、能量单调性和不可行拒绝。四机合成数据已完成四轮端到端训练，CBCL 的 `reference_rebuild_example.json` 配置已完成四轮成本仿真。完整 120 epoch 预训练模型实验没有在本次代码编写中执行。

API 依据：[CVXPY 原子函数](https://www.cvxpy.org/api_reference/cvxpy.atoms.elementwise.html)、[PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)、[标准 ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。
