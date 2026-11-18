# UAV-SFL 小函数拆分版

入口为同目录的 `uav-sfl_1.py`，可独立运行，不导入 `uav_sfl.py`。它保留更细的小函数调用结构，与当前主入口同步使用动态切分＋资源优化、四机等周期闭合蛇形轨迹，以及每机500个初始样本。轨迹和采样参数说明见本目录README；旧圆轨迹配置仍可重现历史结果。

当前 `check` 包含26项检查：原21项协议检查，加上等周期闭合、旋转与错相、圆角运动连续性、500样本释放边界及小规模数据兼容检查。

## 主要拆分

| 原主要函数 | 新拆出的步骤 |
|---|---|
| `validate_config` | 训练、数据、硬件、轨迹、优化、样本快照六组校验；共用 `require_positive` |
| `read_cbcl_records` | `index_cbcl_images` 建图片索引，`cbcl_object_record` 解析单对象 |
| `prepare_data` | `split_cbcl_scenes` 划分真实场景，`synthetic_data_indices` 生成合成清单，`assemble_data_manifest` 组装清单 |
| `load_batch` | 真实/合成样本读取与 `batch_tensors` 标准化、张量转换 |
| `initialize_model` | `initialize_resnet_tensors` 创建初值，`restore_model_checkpoint` 恢复本地权重 |
| `train_client_epoch` | 循环调用 `train_client_batch`，完成读取、梯度与按样本比例更新 |
| `profile_splits` | `profile_split_point` 测量单个切点，`assemble_split_profile` 组装画像 |
| `convex_subproblem` | 比例变量、速率表达式、能量表达式、约束与目标、求解、物理资源恢复六步 |
| `execute_experiment` | 输入准备、切分选择与参考恢复、候选求解、分支训练、训练结果保存、CSV 保存 |
| `verify_implementation` | 梯度、训练聚合、采样划分、异构资源、求解、电池、参考恢复及预算信道的分类检查 |
| `main` | 参数解析、配置覆盖、smoke 配置、实验命令和准备命令分派 |

## 调用顺序

```text
main
  ├─ build_argument_parser / load_cli_config
  ├─ run_preparation_command                 # check / prepare / profile
  └─ run_experiment_command                  # simulate / train / smoke
       └─ run_experiment
            └─ execute_experiment
                 ├─ prepare_experiment_inputs
                 ├─ 每轮冻结样本和信道
                 ├─ select_epoch_splits
                 ├─ solve_epoch_candidates → solve_resources → convex_subproblem
                 ├─ train_epoch_branches → train_client_epoch → train_client_batch
                 ├─ 聚合与验证；本轮成功后提交时间、电量和下一切点
                 └─ save_training_results / save_history_csv
```

## 运行

在 PowerShell 中使用原代码已安装依赖的 Python 环境。本次验证使用的解释器如下；也可将 `$sflPython` 换为你自己的环境路径。

```powershell
cd 'E:\Work1SFL\code\code_v2'
$sflPython = 'E:\Work1SFL\artifacts\archive_verification_env\Scripts\python.exe'
& $sflPython -X utf8 -B .\uav-sfl_1.py config
& $sflPython -X utf8 -B .\uav-sfl_1.py check
& $sflPython -X utf8 -B .\uav-sfl_1.py smoke
```

其他命令仍为 `prepare`、`profile`、`simulate` 和 `train`，沿用原 JSON 配置及 `--config`、`--output`、`--checkpoint`、`--allow-random-init` 参数。默认结果目录仍为同目录 `output`，每次运行新建子目录。

文件名按要求包含连字符，适合以上命令行启动；它不能直接写成 Python 的 `import uav-sfl_1`。需要从其他 Python 代码加载时可使用 `importlib.util.spec_from_file_location`；Windows 并行搜索建议使用命令行入口。

## 验证结果

以下为初版函数拆分时的回归验证记录；当前新轨迹与初始样本规则另由上述26项检查覆盖。

- 新版 `check` 的 21 项协议检查全部通过。
- 新旧版本四机四轮 `smoke` 均完成；逐轮决策、CSV、能耗、训练与验证指标一致，比较时排除了墙钟时间和源码版本字段。
- 数据清单、IID/Dirichlet 划分、CBCL 裁剪、随机初始化、官方 checkpoint、五个切点画像和尾批训练通过对照。
- 奇数轮、中间聚合轮、最终轮各自的正常优化与 Phase-I，共六组资源结果在 `1e-9` 容差内一致。
- 完整训练与注入中断的部分训练，对应模型张量和 checkpoint 保存结果一致。
- 单机两轮、两个工作进程的 `optimized_window` 命令行仿真完成，验证 Windows 子进程入口。

以上为重构回归验证，未运行正式 120 轮 CBCL 训练。验证脚本和输出位于 `E:\Work1SFL\tmp\uav_sfl_refactor`。
