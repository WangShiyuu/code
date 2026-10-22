# UAV-SFL 代码

- `code_v0/`：旧版代码的完整本地归档；Git中按其原有忽略规则保存代码和训练记录。
- [`code_v1/`](code_v1/README.md)：统一参数、CBCL数据划分、连续采集快照、固定巡逻轨迹和动态信道环境。

## code_v1 使用

修改 [`code_v1/config.py`](code_v1/config.py) 中的参数后运行：

```powershell
cd code_v1
python -m pip install -r requirements.txt
python -X utf8 -B prepare_scenario.py --stage all
python -X utf8 -B example_read.py --with-data
```

数据准备需要本地 CBCL StreetScenes 的 `Images.zip` 和 `Annotations.zip`，可通过配置或 `--data-root` 指定。程序不下载原始数据。

图片缓存和 `code_v1/output/` 不纳入Git；运行脚本可在本地生成数据清单、环境CSV、参数快照及巡逻轨迹图。详细配置与接口见 [`code_v1/README.md`](code_v1/README.md)。完整训练循环尚未接入 `code_v1`。
