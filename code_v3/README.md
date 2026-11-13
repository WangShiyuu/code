# UAV-SFL 模块版

配置、优化和训练各有独立的 Python 接口。优化与训练都从配置模块读取参数，也可以由 `main.py` 串成完整实验。

| 文件 | 负责的工作 | 常用函数 |
|---|---|---|
| `config.py` | 默认参数、JSON 覆盖、参数校验、训练预算及共享的切点定义 | `default_config`、`load_config`、`training_budget` |
| `optimization.py` | 轨迹与信道、切点候选、资源分配、时延与能耗 | `channel_at`、`make_problem`、`select_splits`、`solve_resources` |
| `training.py` | CBCL 和合成数据、ResNet18、分段反向传播、模型聚合与画像 | `prepare_data`、`initialize_model`、`train_client_epoch`、`aggregate_models`、`profile_splits` |
| `main.py` | 命令行入口、逐轮调度、实验记录及源码归档 | `main`、`run_experiment` |
| `verification.py` | 检查梯度、尾批、样本释放、资源约束与切点选择 | `verify_implementation` |

## 运行

在 PowerShell 中进入本目录，使用已安装依赖的 Python 环境。本机可以直接执行以下命令。

```powershell
cd 'E:\Work1SFL\code\code_v3'
$sflPython = 'E:\Work1SFL\artifacts\archive_verification_env\Scripts\python.exe'
& $sflPython -X utf8 -B main.py config
& $sflPython -X utf8 -B main.py check
& $sflPython -X utf8 -B main.py smoke
```

`config` 打印生效参数。`check` 执行 21 项协议检查。`smoke` 使用合成数据完成四机四轮训练及两次模型聚合。各命令可单独执行。`-B` 使 Python 跳过字节码缓存写入。

| 命令 | 用途 |
|---|---|
| `prepare` | 读取数据并保存清单、客户端划分及样本释放表 |
| `profile` | 测量五个切点的计算量和通信载荷，生成 `split_profiles.json` |
| `simulate` | 按轮求解切点和资源，计算任务时延、能耗 |
| `train` | 按轮求解后执行真实的分段训练、聚合和评估 |

默认 CBCL 数据位于 `E:\Work1SFL\DataSet\CBCL StreetScenes`。正式训练可以直接引用已有权重。

```powershell
& $sflPython -X utf8 -B main.py train --checkpoint '..\code_v2\resnet18-f37072fd.pth'
```

修改默认参数可编辑 `config.py` 中的 `default_config()`。单次实验也可以用 `--config <JSON 路径>` 覆盖部分参数。新 Python 环境可安装 `numpy>=1.26`、`Pillow>=10`、`torch>=2.5`、`cvxpy>=1.6` 和 `clarabel>=0.10`。

## 独立调用

从 `E:\Work1SFL\code` 启动 Python 后，可按需导入三个模块。以下示例让第一架 UAV 在切点 4 上执行一轮本地训练。

```python
from code_v3.config import load_config, smoke_config
from code_v3.training import epoch_snapshot, initialize_model, prepare_data, train_client_epoch

cfg = smoke_config(load_config())
data = prepare_data(cfg)
model, _ = initialize_model(cfg)
snapshot = epoch_snapshot(data, data["start_s"], 1, cfg)
updated_model, stats = train_client_epoch(model, data, snapshot["indices"][0], 4, 1, 0, cfg)
```

独立计算资源分配时，使用与当前配置相匹配的模型画像。下例使用默认配置，读取同一配置的 `profile` 命令生成的文件，将 `profile_path` 换成实际路径即可。返回值中的 `ok` 表示该资源方案是否通过原始约束检查。

```python
import json
from pathlib import Path
from code_v3.config import load_config, uav_parameter
from code_v3.optimization import channel_at, grouped_profile, make_problem, solve_resources

cfg = load_config()
profile_path = Path(r"E:\实际运行目录\split_profiles.json")
profile = grouped_profile(json.loads(profile_path.read_text(encoding="utf-8")))
gain = channel_at(0.0, cfg)["gain"]
cuts = [4] * cfg["num_uavs"]
problem = make_problem(1, 2, gain, cuts, None, uav_parameter(cfg, "energy_cap_j"), cfg)
solution = solve_resources(problem, profile, cfg)
print(solution["status"], solution["ok"])
```

## 结果位置

实验结果默认保存到 `E:\Work1SFL\outputs\code_v3` 下的独立运行目录。使用 `--output <新目录>` 可指定当次位置。`summary.json` 记录训练或仿真的完成状态，`history.csv` 保存逐轮指标，`source_snapshot.zip` 保存当次使用的五个 Python 模块。
