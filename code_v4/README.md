# UAV-SFL 对照实验

每个入口都执行完整的 CUDA 训练、聚合、验证和测试。四组实验复用 v3 的 CBCL StreetScenes 数据划分、样本释放、巡逻轨迹和 ResNet18 训练。默认每机从 500 个样本开始，计划训练 120 轮，每两轮聚合一次。

| 入口 | 切分点 | 资源 |
|---|---|---|
| `fixed_split.py` | 全程固定，默认所有 UAV 取 4 | 每轮用 v3 连续求解器优化频率、功率、上行和下行带宽 |
| `fixed_resources.py` | 每个聚合周期按历史信道窗口选择，按固定资源下的预测总能耗排序 | 全程使用固定参考值 |
| `random_split_resources.py` | 每个聚合周期按 UAV 独立均匀提议组边界 | 每轮随机提议资源，从通过可行性检查的方案中接受首次命中 |
| `fixed_split_resources.py` | 全程固定，默认所有 UAV 取 4 | 全程使用固定参考值 |

允许的切分点为 `0、2、4、6、8`，表示客户端持有的残差块数。非最终聚合轮先确定下一周期切分点，当前轮仍按当前切分训练，并计入聚合后新前缀的下发成本。固定资源默认取各机频率上限的 80%、上行功率上限的 50%，上下行分别等分各自的带宽预算。

随机基线使用独立的种子派生随机数流。每次提议按各机上限的 50%–100% 抽取频率，按 10%–100% 抽取上行功率，分别在上下行带宽份额单纯形上均匀采样。周期边界还使用当轮样本数和信道预检新切分点的下一周期成本。每轮最多提议 2048 次，耗尽后将本次实验记为未完成。

## 运行

从 `code_v4` 目录使用已安装 CUDA 版 PyTorch 的 Python。环境配置见 [v2 的 CUDA 安装步骤](../code_v2/README.md#在-rtx-3080-上运行)。

```powershell
$sflPython = (Resolve-Path '..\..\.venv-gpu\Scripts\python.exe').Path
& $sflPython -X utf8 -B .\fixed_split.py --device cuda:0
& $sflPython -X utf8 -B .\fixed_resources.py --device cuda:0
& $sflPython -X utf8 -B .\random_split_resources.py --device cuda:0
& $sflPython -X utf8 -B .\fixed_split_resources.py --device cuda:0
```

默认使用 `Work1SFL/DataSet/CBCL StreetScenes` 下的 `Images.zip`、`Annotations.zip` 和仓库内的 `code_v2/resnet18-f37072fd.pth`。可通过 `--data-root`、`--checkpoint`、`--seed`、`--epochs` 和 `--config` 覆盖实验参数。固定切分点的两个入口还支持 `--split`。入口文件决定实验策略。跨策略比较时使用同一配置和种子。物理时延不同的实验可能在同一轮拥有不同数量的样本；如需统一样本数，可在共用的 v3 配置中给出 `data.fixed_epoch_counts`。

## 输出

每次通过 CUDA 环境检查的训练在 `output/` 下创建独立目录，名称为 `output_<基线类型>_<北京时间 YYYYMMDD_HHMMSS_ffffff>_<12 位源码 SHA-256>`。源码哈希覆盖入口、共用训练代码和实际调用的 v3 模块。

- `history.csv` 记录逐轮时延、能耗、训练与验证指标，以及逐机切分点、资源、信道和电池余量。
- `summary.csv` 记录有效参数、权重与数据指纹、CUDA 环境、运行状态及汇总指标。完成全部轮次后记录测试集指标。
- 至少成功聚合一次时，默认保存 `last_aggregated_model.pt` 和按验证准确率选出的 `best_validation_model.pt`。

若某轮无法满足时限或电池约束，程序在该轮训练前停止，写明已完成轮次及失败原因，并返回退出码 `2`。固定切分点 4 随着样本释放可能触发这类停止。比较能耗时应结合完成轮次及处理样本数。
