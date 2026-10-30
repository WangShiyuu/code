# 动态 UAV-SFL · code_v2

所有参数、数据准备、环境、优化、ResNet18、训练和检查均在 **`uav_sfl.py` 一个代码文件**。没有项目内部跨文件依赖，也不导入 v0/v1。实现依据为 [Notion 动态UAV-SFL](https://app.notion.com/p/3b5427f94a0b8040a213d5d9b277c8df)，源页面编辑时间为北京时间 **2026-10-30 02:45:37**，完整读取快照保存在 `notion_source.md`。

## 先运行

本机已检查 `code_v1/.venv` 具备所需依赖，可以直接复用解释器：

```powershell
cd E:\Work1SFL\code\code_v2
& '..\code_v1\.venv\Scripts\python.exe' -X utf8 -B uav_sfl.py config
& '..\code_v1\.venv\Scripts\python.exe' -X utf8 -B uav_sfl.py check
& '..\code_v1\.venv\Scripts\python.exe' -X utf8 -B uav_sfl.py smoke
```

`check` 验证数学/训练协议；`smoke` 使用两机、4 epoch、32×32 合成图像执行真实反向传播、资源优化和两次完整聚合。它是诊断，不能作为 CBCL 准确率实验。

在新电脑中安装：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -X utf8 -B uav_sfl.py smoke
```

Python 3.10+；CPU 可运行，GPU 环境需安装与设备匹配的 PyTorch。无需 torchvision。

## 正式使用

```powershell
# 按场景准备数据清单、等量客户端序列和完整释放时间表
python -X utf8 -B uav_sfl.py prepare --config example_config.json

# 生成九个切分点画像
python -X utf8 -B uav_sfl.py profile --config example_config.json

# 仅仿真资源/能耗，不更新神经网络参数
python -X utf8 -B uav_sfl.py simulate --config example_config.json

# 同时执行资源决策与真实训练，读取本目录已准备的官方预训练 checkpoint
python -X utf8 -B uav_sfl.py train --checkpoint .\resnet18-f37072fd.pth

# 显式启用参考资源重建扩展，4 epoch 物理成本诊断
python -X utf8 -B uav_sfl.py simulate --config reference_rebuild_example.json
```

上述 `python` 应指向已安装依赖的解释器。本机可替换为 `& '..\code_v1\.venv\Scripts\python.exe'`。

`default_config()` 是唯一默认参数表；`example_config.json` 只是局部覆盖示例。相对数据路径和 checkpoint 路径相对配置文件，默认输出锚定本目录。`--output` 指定本次实验的新目录，已有实验不覆盖。

正式 `train` 默认要求提供预训练模型，不会静默下载、随机初始化或冒称已有预训练权重。支持标准 ResNet18 的 `state_dict`、`{"state_dict": ...}` 或本程序保存的 `{"params": ..., "buffers": ...}`；ImageNet 分类头维数不符时，保留预训练骨干并用统一种子重建九分类头，记录 `head_reinitialized`。v1 自定义 `residual_blocks.block*` 格式未做自动转换。只有显式 `--allow-random-init` 才允许随机初始化实验。

本目录现已准备 `resnet18-f37072fd.pth`，来自 [PyTorch 官方 ResNet18 IMAGENET1K_V1](https://download.pytorch.org/models/resnet18-f37072fd.pth)，文件大小 46,830,571 字节（约 44.66 MiB）。SHA-256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`，已匹配官方文件名中的哈希前缀。它在 ImageNet 上预训练，CBCL 九分类头仍需本研究训练。该官方文件采用旧 BN 格式，加载器按 PyTorch 兼容规则为缺失的 `num_batches_tracked` 补零，保留所有预训练权重和 BN 统计量，并记录到初始化来源中。

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
| 画像 | `profile_splits(model, config)` | 每个字段长度为 9 的 `Profile` |
| 原始成本 | `evaluate_resources(problem, resources, profile, config)` | 阶段时延、每机能耗、服务器能耗及组成 |
| 约束检查 | `audit_resources(...)` | `ok, normalized_violation, residuals` |
| 切分决策 | `select_splits(...)` | 候选向量列表、评分矩阵 `[N,9]`、枚举接受/拒绝日志 |
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

## 与 Notion 的对应

- 式(1)：每个 epoch 用 `D(t_k)` 的索引前缀。释放表从物理时间 0 开始；初始数据达到 `initial_fraction` 时确定 `t_1`。没有轮内追加训练样本。
- 式(2)—(5)：标准 ResNet18，stem 后及八个完整残差块后共九个切分点。每架 UAV 保留独立完整参数字典，其前缀/后缀由函数互补执行。每两轮进行完整模型平均，切换切点保留参数。
- 式(6)—(8)：预定偏心圆巡逻轨迹随物理时间变化，使用概率加权的**线性功率增益**；无旧版随机小尺度衰落。轨迹是可替换的场景输入。
- 式(9)—(17)：UAV/服务器前后向计算、激活+int64标签上传、梯度下传、旧前缀上传、新前缀下发、聚合开销均计入。三个阶段分别取最大时延后相加；最终聚合不下发。
- 式(19)：普通 SGD，实际批次步长为 `eta * batch_size_actual / D_k`，尾批同样计入。BN 统计量固定、仿射参数可训练。不使用动量、衰减、随机增强或混合精度。
- 式(23)：默认 `K_t=100` 为经验预算。`A_bar=3.85,C_bar=0.0115,epsilon=0.05` 得到条件性 100 epoch，但代码不会据此宣布理论保证。`budget.mode=theoretical` 要求显式设置有效界已经验证。
- 式(25)—(26)：窗口只含已完成轮次信道；参考预测覆盖未执行聚合轮和下一两轮；下一聚合时是否继续下发依据剩余轮数决定。逐机枚举九点，严格降分才接受，同分保持原切分。首个可行组合按参考评分从低到高搜索。
- 式(28)—(31)：CVXPY 用相对熵表示 Shannon perspective、按阶段引入上图变量；逆二次变换更新上行能量上界。频率/带宽使用比例变量，速率使用 Mbit/s，目标和电池约束做数值缩放。求解后必须重算原始约束及真实能耗，失败不会静默执行参考资源。

若第一次固定参考资源不可行，初始化在若干固定发射功率下求凸 Phase-I。未找到可行点的状态是 `feasible_initialization_not_found`，不是全局不可行证明。离散初始化达到 `max_initial_candidates` 或坐标扫描上限会明确报告。逆二次迭代达到上限但原约束可行时返回 `iteration_limit_feasible`，不冒称已收敛或全局最优。

## 已发现的参考筛选边界与可选扩展

严格版本使用最近一次可行资源作为下一次 P1 的固定参考。真实数据试算中，第一轮以 444 样本/机训练、耗时约 600 秒；下一轮累计样本增长后，上一轮低频参考资源无法满足时限，参考 P1 无可行切分。这说明**参考筛选失败**，并不能证明当前联合资源问题 P2 无解。程序通过固定参考下的时延下界提前识别，避免无意义枚举所有 `9^N` 组合。

默认 `optimization.reference_recovery="none"` 保留文稿行为并返回 `incomplete`。如需研究这一边界，可显式设为 `"rebuild"`：先在窗口平均信道下为保留原切分重建可行参考资源，再重新运行 P1，最终仍按当前真实信道解 P2。此选项是**文稿之外的工程扩展**，日志记录原失败状态和重建过程，应在实验方法中单独说明。它也不保证剩余任务全部可行。

## 数据、基线及参数标定

默认读取 `E:/Work1SFL/DataSet/CBCL StreetScenes/Images.zip` 和 `Annotations.zip`。每个 polygon 的外接矩形是一个九分类区域。先按原始街景隔离训练/验证/测试，再将训练区域 IID 等量、不放回分给无人机，余数明确放入 `unused_train`。因此 80/10/10 是**独立场景比例**，区域数比例不会严格相等。

本机实际读取到 3,547 个场景、27,662 个有效区域，跳过 5 个删除/空标注区域。默认五机划分为每机 4,442 个训练区域，共 22,210 个，验证 2,751 个、测试 2,700 个，剩余 1 个训练区域不用。初始每机释放 444 个。数据不会被复制为 Notion 中尚待核对的 27,000 样本示例。

`data.partition="dirichlet"` 支持类别异质性，并强制等客户端容量；容量约束会改变最终分布，实际类别计数保存在清单。更换独立划分单位时需调整 `prepare_data`，不能直接对同场景区域随机切分 holdout。

| strategy | 切分 | 连续资源 |
|---|---|---|
| `fixed_fixed` | 固定 `fixed_split` | 固定参考资源，仍检验时限/电池 |
| `fixed_opt` | 固定 `fixed_split` | 逐轮优化 |
| `dynamic_opt` | 最近 W 个已完成 epoch 平均信道 | 逐轮优化 |
| `no_window` | 最近 1 个已完成 epoch | 逐轮优化 |

多种子、带宽/电池/采集率等敏感性可用配置循环调用 `run_experiment`。不同策略必须共用数据划分、释放表、模型 checkpoint 和种子。正常模式允许不同物理执行时间导致不同 `D_k`，结果同时统计唯一已用样本量与累计处理次数。`data.fixed_epoch_counts=[...]` 可提供共同的非减轮次样本数，属于**统一快照消融**，此时明确绕过物理释放条件，不能混作主实验。

图像尺寸默认 64；本地批量大小、SGD 步长需在验证集调参。硬件 `kappa`、`cycles_per_flop`、服务器每分支频率、带宽、LoS 参数、轨迹、电池和 600 秒单轮时限都是场景假设。画像通过真实前后向 FLOP 计数乘配置的周期换算系数获得，**不是设备实测周期/能耗**；未被计数器支持的算子、SGD 更新和内存/I/O开销未计入。实际测量画像可通过 `run_experiment(..., profile=measured_profile)` 传入同结构数据并标明元数据。

`prefix_bits` 计可训练参数；固定且各分支相同的 BN buffers 已预部署，不重复上传。服务器频率和下行功率均按每条并行分支解释。正资源下界 `positive_floor=1e-5` 是数值近似，带宽/频率/功率的可行域因此比严格开区间略小。功耗仅为通信/计算模型预测，本机串行执行时间、真实硬件功耗不混入系统模型。

## 输出与验证

每次输出到 `output/<模式>_<北京时间>_seed.../`：

- `config.json`、`data_manifest.json`、`split_profiles.json`、`initialization.json`：全部输入、样本释放和来源。
- `decisions.jsonl`：每轮样本索引、切分候选/接受拒绝日志、资源、信道、载荷、能耗组成、原约束残差、优化目标轨迹、求解耗时。
- `history.csv`：便于后续绘图的逐轮指标。
- `summary.json`：完成状态、能耗、物理完成时刻、电池、累计样本处理数和失败原因；真实训练另有验证与测试指标。
- `last_aggregated_model.pt`、`best_validation_model.pt`：可选保存，验证集选择最佳模型，测试集仅在任务末评价。未完成时 `final_model=None`，最近聚合模型与未聚合分支不混淆；必要时另存 `partial_branches.pt`。

命令退出码：0=成功，1=输入/执行错误，2=任务未完成。未完成任务的低能耗不能作为节能收益。

已执行检查包括九点梯度及单步更新一致性、输入参数不变、尾批步长、加权聚合、快照冻结、数据互斥、同步载荷、最终零下发、原始约束、能量单调性和不可行拒绝。另执行真实 CBCL 三样本训练诊断和四轮合成数据端到端训练。完整 100 epoch 预训练模型实验没有在本次代码编写中执行。

API 依据：[CVXPY 原子函数](https://www.cvxpy.org/api_reference/cvxpy.atoms.elementwise.html)、[PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)、[标准 ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。
