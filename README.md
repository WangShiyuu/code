# UAV-SFL 代码

当前版本位于 [code_v2](code_v2/README.md)。新电脑以 `D:\Wangshiyu\Work1SFL\code` 为仓库目录，将 `Images.zip` 和 `Annotations.zip` 放入 `D:\Wangshiyu\Work1SFL\DataSet\CBCL StreetScenes`。在 PowerShell 中首次运行时创建项目独立的 Python 环境。

```powershell
cd 'D:\Wangshiyu\Work1SFL\code\code_v2'
python -m venv .venv
$sflPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $sflPython -m pip install -r .\requirements.txt

# 查看解析后的数据及输出位置，检查求解器和四机小规模训练。
& $sflPython -X utf8 -B .\uav_sfl.py config
& $sflPython -X utf8 -B .\uav_sfl.py check
& $sflPython -X utf8 -B .\uav_sfl.py smoke

# 正式训练：使用已包含的官方 ImageNet 预训练权重。
& $sflPython -X utf8 -B .\uav_sfl.py train --checkpoint .\resnet18-f37072fd.pth
```

正式训练和仿真的结果默认写入 `code_v2/output/` 下的独立目录，实际生效的参数保存为 `config.json`。数据准备、纯成本仿真、异构硬件配置和 `--output` 覆盖方法见 [v2 运行说明](code_v2/README.md)。

- `code_v0/`：旧代码完整归档。
- [`code_v1/`](code_v1/README.md)：统一数据与环境，以及周期切分、块级资源优化和实际 SFL 训练。
- [`code_v2/`](code_v2/README.md)：四机异构资源实验，切分点穷举、连续资源分配、独立电池约束和 ResNet18 训练集中在 `uav_sfl.py`。下文命令和配置说明针对 v1。

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
