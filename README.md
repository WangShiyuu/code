# UAV-SFL 代码

- `code_v0/`：旧版代码的完整本地归档；Git中按其原有忽略规则保存代码和训练记录。
- [`code_v1/`](code_v1/README.md)：统一参数、CBCL数据划分、连续采集快照、固定巡逻轨迹、动态信道及SFL训练诊断。

## code_v1 使用

修改 [`code_v1/config.py`](code_v1/config.py) 中的参数后运行：

```powershell
cd code_v1
python -m pip install -r requirements.txt
python -X utf8 -B prepare_scenario.py --stage all
python -X utf8 -B example_read.py --with-data
```

数据准备需要本地 CBCL StreetScenes 的 `Images.zip` 和 `Annotations.zip`，可通过配置或 `--data-root` 指定。程序不下载原始数据。

每次训练自动保存到 `code_v1/output/output_时间戳/`，其中报告、参数、环境副本及可选模型允许Git提交。可复用的图片缓存、默认共享数据/环境输入目录和虚拟环境留在本机。

首次训练另安装 `code_v1/requirements-training.txt`，运行 `python -X utf8 -B train_sfl.py`。默认完成2个epoch和1次聚合，不检查电量或正式任务时限；固定切点和参考资源，后续再接入优化与独立比较代码。每次结果含 `summary.json`、`report.md`、`key_metrics.csv`。已有记录可用 `python -X utf8 -B report_results.py --report 记录路径` 汇总，详细步骤及输出解释见 [`code_v1/README.md`](code_v1/README.md)。
