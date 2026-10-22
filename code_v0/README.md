# CBCL 无人机训练代码

在 PowerShell 中先进入仓库的 `code` 目录并安装依赖。下面以仓库位于 `D:/Wangshiyu/Work1SFL/0721-SplitFed-When-Federated-Learning-Meets-Split-Learning-main` 为例；其他位置请替换为自己的路径：

```powershell
Set-Location 'D:\Wangshiyu\Work1SFL\0721-SplitFed-When-Federated-Learning-Meets-Split-Learning-main\code'
python -m pip install -r requirements.txt
```

使用 Python 3.10+。GPU 版 PyTorch 请按本机 CUDA 环境安装；`--device cpu` 可在 CPU 上运行。`data.py` 负责处理和划分数据，`models.py` 提供共用 ResNet18。五种训练方式各有独立脚本和自己的训练主流程；`training_common.py` 共享参数解析、单轮训练、评估、切点成本估计和输出保存等通用函数，避免重复代码。

## 1. 准备 CBCL 数据

首次训练前必须先运行 `data.py`。当前推荐 5 架 UAV 分配 1000、1250、1500、1750、2000 个训练样本，生成 `output/data_load_7500/data.json` 和图片缓存；数据已准备好时直接复用。仓库不包含原始 CBCL 数据；数据缓存和模型权重被 Git 忽略，需要在训练主机本地生成。训练指标与配置允许提交，另一台主机拉取后即可查看已提交的结果。

原始目录需含 `Images.zip` 和 `Annotations.zip`。下面以 `D:/Wangshiyu/Work1SFL/DataSet/CBCL StreetScenes` 为例；请将 `--data-root` 替换为实际包含这两个压缩包的目录。程序不会自动下载数据。

```powershell
# 第一次运行：准备 7500 个 CBCL 训练区域并分给 5 架 UAV
python data.py --data-root "D:/Wangshiyu/Work1SFL/DataSet/CBCL StreetScenes" --uavs 5 --partition iid --samples-per-uav 1000 1250 1500 1750 2000 --output output/data_load_7500
# 上一步成功生成 output/data_load_7500/data.json 后再开始训练
python centralized.py --rounds 100 --batch-size 32 --lr 0.01
```

后续训练可复用生成的数据，无需每次重新处理。省略 `--data-root` 时，程序沿本目录的祖先目录寻找 `DataSet/CBCL StreetScenes`。下面示例使用这个默认查找规则；自定义存放位置时也需加上 `--data-root "你的路径"`。

```powershell
# 另一份 IID 划分：3 架 UAV，各 1000 个训练样本；另存目录以保留首次划分
python data.py --uavs 3 --partition iid --samples-per-uav 1000 1000 1000 --output output/data_iid_3000

# non-IID：另存一份划分；alpha 越小，类别分布通常越不均衡
python data.py --uavs 3 --partition noniid --alpha 0.3 --train-samples 4500 --samples-per-uav 1000 1500 2000 --output output/data_noniid
# 使用这份 non-IID 清单训练；其他四种方式也用同一个 --data 参数
python centralized.py --data output/data_noniid/data.json --rounds 100 --batch-size 32 --lr 0.01
```

| 参数 | 作用 |
| --- | --- |
| `--uavs 3` | 无人机数量。 |
| `--train-samples 3000` | 选中的训练样本总量，省略时使用全部可用训练区域。 |
| `--samples-per-uav 100 200 300` | 按 UAV 编号指定精确容量；单值 `100` 表示每架 UAV 都有 100 个样本。 |
| `--partition iid` / `noniid` | 随机 IID 分配，或容量受限的 Dirichlet 类别偏好分配。 |
| `--alpha 0.5` | non-IID 浓度，默认 0.5；实际类别计数保存在清单中。 |
| `--val-fraction 0.15 --test-fraction 0.15` | 按原始街景数量设置验证和测试比例，默认各 15%。 |
| `--seed 42 --image-size 64` | 划分随机种子和缓存边长，默认 42 和 64。 |
| `--output output/data` | 数据清单和 PNG 缓存目录，默认本目录下的 `output/data`。 |

若只给总量，则自动均分，容量最多相差 1；同时给总量和逐 UAV 容量时，两者总和必须一致。请求量超过训练场景可提供的样本时直接报错，不通过重复样本补足。相同参数可复用完成缓存；更改参数需指定新的输出目录。

数据任务是 CBCL 的九类**区域分类**：bicycle、building、car、pedestrian、road、sidewalk、sky、store、tree。每个标注 polygon 的外接框裁成 RGB 图像，默认缩放为 64×64，保留框内背景，按固定均值和标准差归一化。

先隔离原始街景，再分配训练区域，因此同一原图不会跨训练、验证和测试集。为支持精确容量，同一训练街景中的不同区域可以分给不同 UAV，但每个区域只分配一次。划分是本项目的自定义场景划分；改变 UAV 数量或容量时，固定种子下的验证/测试场景不变。

## 2. 五种训练方式

所有方式使用同一份 `data.json` 和标准 ResNet18 九分类头。以下命令共用刚生成的 IID 数据；若使用 non-IID，将变量改为 `output/data_noniid/data.json`。

```powershell
$DataManifest = 'output/data_load_7500/data.json'
python centralized.py --data $DataManifest --rounds 100
python local.py --data $DataManifest --rounds 100
python fedavg.py --data $DataManifest --rounds 100 --aggregation-period 1
python split_fixed.py --data $DataManifest --rounds 100 --split P4 --aggregation-period 1
python split_dynamic.py --data $DataManifest --rounds 100 --client-gflops 2 3 4 5 6 --server-gflops 100 --bandwidth-mbps 10 15 20 25 30 --aggregation-period 1
```

- `centralized.py`：汇总全部 UAV 已分配的训练样本，训练一个完整模型。
- `local.py`：每架 UAV 独立训练自己的完整模型，不聚合；在相同验证/测试集分别评价。
- `fedavg.py`：各 UAV 完整本地训练，按样本数加权平均模型。
- `split_fixed.py`：各 UAV 的前缀与服务器上的独立后缀协同训练，切点固定。
- `split_dynamic.py`：训练流程同 SplitFed，根据计算能力和带宽选择切点，也可指定手动日程。

**1 个 SL round = 1 个完整 epoch。** 每轮随机排列并遍历每架 UAV 当前持有的全部样本，`drop_last=False` 保留不足 batch 大小的尾批。Centralized 每轮遍历所有 UAV 样本的并集。`--rounds 1` 就是完整执行一次上述遍历，`--epochs` 是同义参数。

`--aggregation-period K` 默认 1，用于 FedAvg 和两种 SplitFed。每 K 轮聚合一次，最后不足 K 轮也聚合，例如 5 轮、K=2 在第 2、4、5 轮聚合。K>1 时中间轮仅记录训练指标；聚合后才评价共享模型。Local 每轮独立评价。

训练默认值集中在 `experiment_config.py` 的 `CONFIG` 字典，五种训练入口会读取它；显式命令行参数优先。普通训练的默认清单为 `output/data_load_7500/data.json`，full 保留自己的 `output/data_full/data.json`。五种方式采用同一套默认训练设置，与 `resnet18_full.py` 的中心式基线一致：

| 设置 | 默认值 |
| --- | --- |
| 优化器 | SGD，`--momentum 0.9 --weight-decay 0.0005`。 |
| 学习率 | `--lr 0.01 --scheduler cosine`，按训练轮数进行余弦下降。 |
| BatchNorm | `--bn batch`，训练时更新统计量，评价时使用累计统计量。 |
| batch 大小 / 输入边长 | `--batch-size 32 --image-size 64`。 |
| 训练轮数 / 随机种子 | `--rounds 100 --seed 42`。 |
| 最佳模型 | 验证准确率最高；准确率相同时，选择验证损失较低的模型。 |

每轮使用本轮学习率，轮末更新调度器；余弦周期等于总训练轮数 `rounds`。各 UAV 使用相同的轮数进度，聚合不会重新开始学习率调度。`--scheduler constant` 可改为恒定学习率。

先按简单规则做对比：五种方式共用同一数据清单，固定 100 轮、batch 32；FedAvg 和两种 SplitFed 统一每轮聚合（K=1）。`best.pt` 仅用于保存和选择最佳模型，没有早停，仍跑完指定的全部轮次。累计通信量对应全部已执行轮次，不按最佳轮次截断。

`--device auto` 自动选择 CUDA 或 CPU，也可分别设置 `--client-device cpu --server-device cuda:0`。Windows 默认 `--workers 0`，CPU 默认 `--cpu-threads 1`。普通 BN 遇到单样本尾批时，临时使用已累计统计量以保留该样本；`--bn frozen` 可固定均值/方差，同时仍训练 gamma/beta。

FedAvg、固定切分和动态切分在每次聚合后广播共享模型，并清空各 UAV 的 SGD 动量，避免将聚合前的本地梯度方向继续带入共享权重；学习率进度保留。Centralized 和 Local 在连续 epoch 之间保留动量。动态切点移动网络层时，同时把对应的优化器状态移动到新设备。

三个聚合方法都按 UAV 样本数加权平均完整模型参数和浮点 BN 统计量，整数 BN 计数取最大值。这是统一的 BN 聚合规则，并非保留各 UAV 独立 BN 的 FedBN。

对比时五种方式需使用同一份数据清单、随机种子或初始权重，以及上述相同设置。中心式训练集是 UAV 训练样本的并集；验证集和测试集不参与训练。若要使用旧优化设置，可对五个入口统一添加 `--lr 0.001 --momentum 0 --weight-decay 0 --scheduler constant --bn frozen`，但最佳模型仍按统一的验证准确率规则选择。

## 3. 手动固定切点

P0 在 stem（初始卷积与池化）后，P1～P8 分别在第 1～8 个完整残差块后，不拆开残差分支。切点越靠后，UAV 承担的网络层越多。一个值由所有 UAV 共用，也可为 5 架 UAV 分别指定：

```powershell
python split_fixed.py --data $DataManifest --rounds 100 --split P0 P2 P4 P6 P8
```

SplitFed 在单进程中模拟激活、标签传向服务器和边界梯度返回 UAV。每架 UAV 都有独立服务器后缀；聚合时按样本数平均**完整模型**，即使切点不同也按相同层名合并。代码不建立真实无线网络。

## 4. 动态切点和手动日程

自动方式统计 ResNet18 前后向算子的 FLOPs，对各 UAV 估计一轮串行计算与通信耗时，再选择最小成本切点。`--client-gflops 2 3 4 5 6` 为逐 UAV 算力，`--server-gflops 100` 为服务器算力，`--bandwidth-mbps 10 15 20 25 30` 为各 UAV 上下行带宽；逐 UAV 参数也可只给一个值广播。

默认候选为 P0～P8，可用 `--candidates P2 P4 P6` 限制。每个 K 轮聚合区间开始时重新选择，区间内保持切点固定；换切点移动已有层，保留已训练权重，并迁移对应优化器状态。硬件条件和成本最优点不变时，动态方式可能一直选择同一位置。

`--bandwidth-trace 路径.json` 可读随轮数变化的带宽，JSON 形状为 `[rounds][UAV]`，每行也可用一个数字广播。下面是 2 轮、5 架 UAV 的文件内容示例：

```json
[[10, 15, 20, 25, 30], [30, 25, 20, 15, 10]]
```

```powershell
python split_dynamic.py --data $DataManifest --rounds 2 --aggregation-period 1 --bandwidth-trace bandwidth.json --client-gflops 2 3 4 5 6
```

`--split-schedule 路径.json` 可改为手动动态日程，此时跳过自动选点。下面是 2 轮、5 架 UAV 的文件内容示例；每行也可用一个 `"P4"` 字符串广播：

```json
[["P0", "P2", "P4", "P6", "P8"], ["P2", "P0", "P6", "P8", "P4"]]
```

```powershell
python split_dynamic.py --data $DataManifest --rounds 2 --aggregation-period 1 --split-schedule schedule.json
```

JSON 行数必须等于 rounds，逐 UAV 列数必须等于 UAV 数量。K>1 时，同一聚合区间内的手动切点必须相同；自动方式只在区间开始读取当前带宽作决策。

动态切点的目标是最小化估算耗时。成本基于给定算力和带宽，计算量仅包含 FLOPs 计数器已覆盖的前后向算子；没有优化器更新，也未完整覆盖 BN 等算子、数据加载和设备/切点迁移开销。它不是实际无线耗时或能耗。

通信载荷包含激活、标签、边界梯度与聚合上传/下发；动量留在本地，不额外上传。统计不计初始权重分发、数据收集、切点迁移、评价通信和协议开销。同样的数据量、轮数、batch、切点日程、聚合周期和张量精度下，改变 BN、学习率、动量或权重衰减不会改变这部分字节数；动态方式若选择不同切点日程，通信量可以变化。

本阶段比较测试准确率、累计通信字节和本机训练时间。普通 BN、动量和权重衰减可能改变实际计算/内存开销及能耗；现有真实训练入口没有接入功率测量或物理能耗模型，不报告焦耳。五种方式的新输出用 `energy_measured: false` 明确这一点。

## 5. 初始权重和输出

默认使用 seed 42 随机初始化，不自动下载预训练权重。同一轮对比使用相同的初始完整模型，各 UAV 从该模型开始。可加载兼容的本地 ResNet18；比较不同方式时传入同一个权重文件：

```powershell
python fedavg.py --data $DataManifest --checkpoint initial.pt
# 分类头类别数不同时，仅加载主干，九分类头仍随机初始化
python fedavg.py --data $DataManifest --checkpoint resnet18.pt --backbone-only
```

这些参数加载初始模型权重；保存的模型文件不包含恢复训练所需的优化器与完整进度状态。

训练默认输出到 `output/<method>/<时间戳_seed>`，每次使用新目录。也可添加 `--output output/my_experiment`，指定目录必须不存在或为空。

- `config.json`：训练参数、设备、UAV 容量和数据清单哈希，以及统一配置源代码与 CLI 覆盖后的配置快照。
- `data_manifest.json`：本次划分审计快照；图片仍在原数据缓存中，训练入口继续使用原 `data.json`。
- `history.csv`：逐轮损失、准确率、已处理样本数、batch 步数、切点、聚合状态和通信载荷。
- `best.pt` / `last.pt`：验证准确率最高（打平时损失较低）的模型 / 最后一轮模型；Local 放在各 `uav_00`、`uav_01` 等目录。
- `summary.json`：选中模型的测试准确率、Macro-F1、各类别与混淆矩阵；Local 汇总独立模型结果。五种方式还记录累计 `training_payload_bytes` 和 `training_seconds`。
- 自动动态方式另有 `split_profile.json`，动态方式记录 `split_schedule.json`。

五种方式的 `training_seconds` 汇总每轮训练、数据加载、设备迁移和聚合时间；`wall_seconds` 还包括动态 profiling、验证、最终测试和模型/日志保存。两者都是本机单进程仿真时长，不能直接当作真实 UAV 系统耗时或能耗。Centralized/Local 的训练通信量为 0 只表示该口径内无逐轮通信，数据收集和初始分发等成本没有计入。

`code/.gitignore` 允许 `output` 下所有层级的 `config.json`、`history.csv`、`summary.json`、`data_manifest.json`、`split_profile.json` 、`split_schedule.json` 和 `resource_history.json`。模型权重与裁剪缓存仍保留在训练主机。训练完成后，在训练主机的 `code` 目录正常提交即可同步结果：

```powershell
git add .
git commit -m "Save training results"
git pull --rebase origin main
git push origin main
```

另一台主机在仓库目录运行 `git pull --ff-only origin main`，结果会出现在相同的 `code/output` 子目录。换一次实验无需修改忽略规则；Git 只同步已提交并推送的文件。

验证集用于选择 best，测试集在训练结束后评价选中的模型一次。完整参数说明可分别查看：`python data.py --help`、`python centralized.py --help`、`python local.py --help`、`python fedavg.py --help`、`python split_fixed.py --help`、`python split_dynamic.py --help`。各训练脚本已固定对应方式，运行时无需 `--method` 参数。

## 6. 最简单的完整数据中心式实验

`resnet18_full.py` 使用一个完整 ResNet18。先生成完整数据清单：省略 `--train-samples` 和 `--samples-per-uav`，即可保留全部可用训练区域。`--uavs 1` 只用于生成清单，训练脚本直接读取训练集。

```powershell
python data.py --data-root "D:/Wangshiyu/Work1SFL/DataSet/CBCL StreetScenes" --uavs 1 --partition iid --seed 42 --image-size 64 --val-fraction 0.15 --test-fraction 0.15 --output output/data_full

python resnet18_full.py --data output/data_full/data.json --epochs 100 --batch-size 32 --lr 0.01 --output output/resnet18_full_100
```

默认 `holdout` 模式按原始街景划分约 70% 训练、15% 验证、15% 测试，同一街景的区域不会跨集合；裁剪后的样本数比例可以不同。每个 epoch 遍历全部训练样本，验证准确率最高的模型保存为 `best.pt`，准确率相同时选择验证损失较低的模型。训练结束后用这个模型评价测试集一次。

结果保存到 `output/resnet18_full_100`，`summary.json` 的 `test_accuracy` 和 `test_macro_f1` 是最终测试指标。`history.csv` 记录每个 epoch 的指标，`last.pt` 保存最后一轮模型。输出目录须不存在或为空。

若还要查看模型对所有已知样本的拟合能力，可显式使用 `all` 模式合并 `train + val + test`：

```powershell
python resnet18_full.py --data output/data_full/data.json --mode all --epochs 100 --batch-size 32 --lr 0.01 --output output/resnet18_all_100
```

此时 `summary.json` 的 `all_data_accuracy` 和 `all_data_macro_f1` 是在参与训练的所有样本上的指标，只反映拟合能力，不能作为独立测试准确率。

## 7. 独立配置、trade-off 切点和资源分配接口

本阶段新增三个生产模块。参数来自 `experiment_config.py` 的 `CONFIG`，直接编辑该字典即可；没有另一份必填配置文件。

| 模块 | 当前作用 |
| --- | --- |
| `experiment_config.py` | 配置训练超参、UAV、固定并行服务器、无线环境、参考资源和可选预算；检查和打印参数。 |
| `split_selection.py` | 集中管理旧时延选点算法，并提供固定参考资源下的系统能耗 trade-off 坐标扫描。 |
| `resource_allocation.py` | 建立逐 epoch 工作量，评价资源、时延、能耗和约束；资源求解函数明确保留空接口。 |

```powershell
# 默认只检查/打印配置，不需要读取默认数据路径。
python experiment_config.py
# 检查你已生成的清单与物理配置中的 UAV 数是否一致。
python experiment_config.py --data output/data_load_7500/data.json
# 固定 P4、固定参考资源的物理评价；默认 100 轮。
python resource_allocation.py --data output/data_load_7500/data.json --split P4
# trade-off 选择切点，资源仍使用同一组固定参考值；默认 100 轮。
python split_selection.py --data output/data_load_7500/data.json
# 资源求解器尚未实现：此命令会明确报错，画像和输出创建前就退出。
python resource_allocation.py --mode optimize
# 单元测试只验证公式和接口，不做正式训练。
python -B -m unittest discover -s tests -v
```

物理配置的 UAV 数量必须与清单一致；样本数以 `data.json` 的真实索引为准，不在配置中重复填写，不将尾批补齐到 batch size。普通训练只应用 `training` 超参和旧 `latency` 默认值，不受物理配置中的 UAV 数限制。真实 `split_dynamic.py` 调用迁入新模块的原 GFLOPS/Mbps 时延算法；当前尚未把新的 trade-off 或资源分配联动到真实训练。

### 参数怎样填写

默认物理参数均为可修改的仿真示例，不是实测设备参数。每架 UAV 都有自己的参数字典，可分别改频率上下界、参考频率、功率与位置。

| 配置组 | 示例值与含义 |
| --- | --- |
| `uavs` | 5 架；频率 0.2～1.5 GHz、参考 1 GHz；`kappa=1e-28`，`cycles_per_flop=0.5`；上行功率 0.01～1 W、参考 0.5 W；悬停功率 100 W。 |
| `server` | 5 个独立并行分支，每分支 3 GHz、`cycles_per_flop=0.03`、`kappa=1e-28`；每条下行链路发射功率 2 W；聚合时间 0.01 s、聚合功率 10 W。 |
| `radio` | 上下行总带宽分别 20 MHz，每链路最小 0.1 MHz；参考带宽按 UAV 数均分。 |
| `environment` | 载频 2.4 GHz、噪声谱密度 `3.981e-21 W/Hz`；LoS 参数 9.61/0.16，LoS/NLoS 附加损耗 1/20 dB；LoS 莱斯因子为线性值 10，NLoS 为瑞利衰落。 |
| `budgets` | 单轮时限、每 UAV 单轮电量、任务总时限和每 UAV 总电量，单位 s/J；默认 `None` 不启用。电量可给一个共同上限或每 UAV 一个上限。 |
| `selection` | P0～P8，初始比较点 P4，默认一次坐标扫描；能耗打平保留原切点。 |

固定巡逻点的服务器位置为 `(0,0,0)`，UAV 默认高度 100 m，水平距离依次约 100/120/140/160/180 m。LoS 状态在该点内固定，小尺度衰落每个完整 epoch 更新一次，上下行互易。选择切点只读取当前观测，未来信道不用于评分。服务器分支数必须不少于 UAV 数；每分支频率代表独立处理能力，不把服务器的共享总算力重复分配。

物理模型用 `计算量 FLOPs × cycles_per_flop` 得到计算周期，通信 `bytes × 8` 转成 bits。`cycles_per_flop` 是需标定的等效换算，不应将 GFLOPS 直接当作 GHz。UAV 计算功率是 `kappa × f³`，计算能耗是 `kappa × f² × cycles`；频率与计算功率不是两个独立优化变量。新模型的带宽 Hz 经 Shannon 公式得到速率 bits/s，与旧 `latency.bandwidth_mbps` 的直接速率假设分开。

### 评分、计费和后续算法接入

trade-off 参考 [Notion 动态UAV-SFL 的式（29）—（30）](https://app.notion.com/p/3b5427f94a0b8040a213d5d9b277c8df)，采用[完整 epoch 版本的负载定义](https://app.notion.com/p/3eb427f94a0b81eda002f114118cafaf)。固定资源时，逐架 UAV 比较候选，重新计算整个系统的能耗差；同步悬停由全体 UAV 的功率和系统时间决定。一次扫描不是离散全局最优求解。

评分针对拟执行周期：包含该周期开始的候选前缀下发、K 个 epoch、周期末候选前缀上传和聚合。上一周期当前前缀上传是固定成本，不随候选变化。首周期预加载免初始下发。实际记录按每轮发生的事件计费：聚合轮上传当前前缀，下发下一周期前缀，最终轮只上传聚合、不下发。选下一切点还会检查所在过渡轮的预算；候选下发不再占用下一周期首轮的预算。未来周期末的下一次下发，在下一次选点时检查。模型项不会再重复记作迁移通信。

SL 分支内部先累计计算/激活/服务器/梯度四项时间再取最大；模型上传和下发阶段各自取最大，再加入聚合时间。所有 UAV 在整个系统执行时间内悬停。单轮预算逐次检查；任务总预算对完整记录汇总检查，不通过缩短轮数或放宽资源来使结果显示为可行。

未来将论文资源算法填入 `allocate_resources(problem, warm_start=None)`：输入由 `build_round_problem()` 提供的完整 epoch 工作量、当前/下一切点、信道、资源上下界及预算，变量为 `f_hz`、`p_up_w`、`bandwidth_up_hz`、`bandwidth_down_hz` 和三个阶段辅助时延。`evaluate_resources()` 与 `validate_resources()` 是统一的原式评价和验收接口。当前调用求解器会抛出 `NotImplementedError`，不会自动回退到均分资源；无需安装 MOSEK/CVXPY/SciPy。

### 输出与证据范围

独立运行保存到新的 `output/resource_reference/<时间戳>` 目录，也可指定新的空 `--output`。其中 `config.json` 保存配置快照，`data_manifest.json` 仅保存实际样本量及数据指纹，`split_profile.json` 保存当前 ResNet18 画像，`split_schedule.json` 记录决策，`resource_history.json` 保存每轮参考资源、信道、载荷、分项时间/能耗和约束检查，`summary.json` 汇总模型估算值与可行性。

这些记录明确包含 `optimized: false`、`energy_measured: false`、`actual_training: false`；新规划与旧训练通信统计保持各自口径。画像包含前后向计数器覆盖的 FLOPs 和真实张量大小（包括标签及前缀 BN 状态），未完整覆盖优化器、部分 BN 算子、排队和硬件迁移开销。频率/无线功率只驱动仿真模型，不控制真实 CPU/GPU 或无线硬件。
