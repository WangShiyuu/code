# UAV-SFL 对照实验

四组基线共用 v2 的 CBCL StreetScenes 数据划分、样本释放、巡逻信道、ResNet18 训练和资源审计。默认加载 [`results1122_config.json`](results1122_config.json)，沿用 `results1122` 的四机硬件和训练配置。默认种子为 42，每机从 500 个样本开始，训练 120 轮，每两轮聚合一次，单轮时限 400 秒。四机的训练电量预算分别为 120、95、110、85 kJ。

| 入口 | 切分点 | 资源 |
| --- | --- | --- |
| `fixed_split.py` | 四机全程使用同一个切点，默认 P4 | 逐轮优化频率、发射功率和上下行带宽 |
| `fixed_resources.py` | 每个聚合周期按历史信道窗口和固定资源下的预测能耗选择 | 各机使用参考频率和上行发射功率，上下行分别等分带宽 |
| `random_split_resources.py` | 每个周期独立随机提议各机切点 | 每轮接受首个通过可行性审计的随机方案，最多提议 2048 次 |
| `fixed_split_resources.py` | 四机全程使用同一个切点，默认 P4 | 同固定参考资源 |

固定切点可用 `--split 0/2/4/6/8` 选择，数字表示 UAV 侧保留的残差块数。默认参考频率和上行功率取各机上限。中间聚合轮计入下个周期的前缀下发成本。

## 在另一台 Windows 电脑运行

将更新后的 `code_v2` 和 `code_v4` 同步到另一台电脑。准备 Python 3.13、支持 CUDA 12.6 的 NVIDIA 驱动、仓库内的 `code_v2/resnet18-f37072fd.pth`，以及包含 `Images.zip` 和 `Annotations.zip` 的 CBCL StreetScenes 目录。将下列项目路径替换为本机位置。

```powershell
$sflRoot = 'D:\Wangshiyu\Work1SFL'
Set-Location (Join-Path $sflRoot 'code')
$sflData = Join-Path $sflRoot 'DataSet\CBCL StreetScenes'
$sflCheckpoint = (Resolve-Path '.\code_v2\resnet18-f37072fd.pth').Path
python -m venv ..\.venv-gpu
$sflPython = (Resolve-Path '..\.venv-gpu\Scripts\python.exe').Path
& $sflPython -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu126
& $sflPython -m pip install 'numpy==2.5.5' 'Pillow==13.0.0' 'cvxpy==1.9.3' 'clarabel==0.11.1'
& $sflPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

验证通过后运行四组正式训练。下面两个固定切点入口显式使用同一个 P0。本机以配套配置做成本仿真时，固定 P0 加逐轮资源优化完成了 120 轮，默认固定 P4 在第 64 轮未找到可行初值。省略 `train` 时也执行训练。

```powershell
$sflInputs = @('--data-root', $sflData, '--checkpoint', $sflCheckpoint)
& $sflPython -X utf8 -B .\code_v4\fixed_split.py train --device cuda:0 --split 0 @sflInputs
& $sflPython -X utf8 -B .\code_v4\fixed_resources.py train --device cuda:0 @sflInputs
& $sflPython -X utf8 -B .\code_v4\random_split_resources.py train --device cuda:0 @sflInputs
& $sflPython -X utf8 -B .\code_v4\fixed_split_resources.py train --device cuda:0 --split 0 @sflInputs
```

可先扫描五个固定切点的可行性与模型成本。`simulate` 默认使用 CPU，执行切分、资源决策和约束审计，跳过模型训练与评价。

```powershell
foreach ($sflCut in 0, 2, 4, 6, 8) {
    & $sflPython -X utf8 -B .\code_v4\fixed_split.py simulate --split $sflCut @sflInputs
}
```

`--config` 接受 JSON 覆盖文件。`--seed` 和 `--epochs` 可覆盖种子和轮数，轮数须为正偶数。迁移已有归档配置时，请用 `--data-root` 和 `--checkpoint` 指向本机文件。用 `--output` 可指定新目录或空目录。

## 结果与对照口径

默认输出到 `code_v4/output/`，目录名格式为 `output_<北京时间到微秒>_<Git提交或源码标识>_<train|simulate>_<基线>[_pN]_seed<种子>`。每次运行独立存档，可将完整运行目录提交到 Git。

- `config.json`、`data_manifest.json`、`initialization.json`、`split_profiles.json` 记录生效配置、数据划分与释放表、初始化及计算通信画像。
- `history.csv`、`decisions.jsonl` 记录逐轮样本量、切点、实际资源、模型时延、能耗分项和原约束审计。`summary.json` 汇总完成状态、累计样本处理次数、能耗、电量和失败原因。
- `run.json`、`runtime.json`、`uav_sfl_snapshot.py` 和 `source_snapshot.zip` 记录运行环境、版本、源码哈希和使用的实现。完成训练后保存测试指标，以及按验证准确率选择的 checkpoint 和最后一次聚合的 checkpoint。

若某轮无法通过时限或电量审计，程序在该轮训练前停止，保留已完成轮次及失败详情，返回退出码 `2`。比较系统能耗时应同时报告完成轮次和样本处理次数。轨迹和样本释放按模型时钟推进，同一轮在不同策略下可能有不同的样本量。若需要统一负载消融，可用 `data.fixed_epoch_counts` 固定各轮样本数。跨机器核对时，以 `data_manifest.json` 中的样本划分和释放表为准，数据指纹也包含文件路径和修改时间。程序墙钟耗时单独记录，模型时延和能耗按系统成本公式计算。
