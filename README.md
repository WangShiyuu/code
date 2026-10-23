# UAV-SFL 代码

- `code_v0/`：旧代码完整归档。
- [`code_v1/`](code_v1/README.md)：统一数据与环境，以及周期切分、块级资源优化和实际 SFL 训练。

优化入口 `optimize_sfl.py` 只求解资源与切分，训练入口通过 `--optimization-results` 读取结果，不重复优化。输出目录后缀含当前 Git 的12位短哈希，元数据另记完整提交与未提交代码状态。

默认动态运行 **100 epoch、每 2 epoch 聚合**。先用独立校准信道预演成本，再按参考时长的 1.2 倍确定固定逐轮预算和总期限，生成完整分片长环境。正式块求解联合分配计算频率、上传功率及双向带宽，资源通过可行性和 KKT 审计后才提交模型更新。电量暂只统计；比较算法后续单独实现。

```powershell
cd code_v1
python -m pip install -r requirements-training.txt
python -X utf8 -B prepare_scenario.py --stage all
python -X utf8 -B example_read.py --with-data
python -X utf8 -B train_sfl.py --prepare-only
# 将目录替换为上一条命令打印的实际准备包。
python -X utf8 -B optimize_sfl.py --prepared-run output/output_时间戳_git哈希
python -X utf8 -B train_sfl.py --optimization-results output/output_时间戳_git哈希 --save-checkpoint
```

原始数据为本地 CBCL StreetScenes 的 `Images.zip` 和 `Annotations.zip`。图片缓存可复用，长环境保持原采集速率并在训练前完整生成，单个 CSV 分片不超过 20 MB。

每次运行保存到 `code_v1/output/output_时间戳_git哈希/`。其中 `report.md`、`key_metrics.csv`、`summary.json`、逐块决策日志、参数、长环境及可选模型允许 Git 提交。默认共享数据/环境缓存和虚拟环境留在本机。原固定诊断仍可通过 `--mode diagnostic --epochs 2 --split P4` 运行，历史结果保持原样。

另一台电脑的 Conda、D 盘完整 PowerShell 命令、各代码作用、参数和报告说明见 [`code_v1/README.md`](code_v1/README.md)。
