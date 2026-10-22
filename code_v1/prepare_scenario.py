"""作用：统一执行数据准备、容量解析和飞行环境生成，是日常使用的主入口。

运行指令（先将工作目录切换到 code_v1）：
    python -X utf8 -B prepare_scenario.py --stage all
    python -X utf8 -B prepare_scenario.py --stage data
    python -X utf8 -B prepare_scenario.py --stage environment
    python -X utf8 -B prepare_scenario.py --config settings.json --stage all
    python -X utf8 -B prepare_scenario.py --help

输入：config.py 默认参数、可选 JSON 配置和命令行覆盖项；数据阶段还需要
CBCL 的 Images.zip 与 Annotations.zip。environment 阶段使用已准备的 data.json。
输出：data 阶段生成数据清单与 crops/；environment 阶段生成 environment.csv、
scenario.json 和 patrol_map.svg。all 阶段按上述顺序执行两部分。

本文件不训练神经网络，不执行切分点优化或资源优化。实际数据容量以清单为准，
不会分别在数据模块和环境模块中重复估计。-X utf8 统一控制台编码，-B 避免字节码缓存。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config import load_config, resolve_collection, theory_budget, validate_config


def read_manifest_metadata(path):
    """只使用标准库读取并审计 code_v1 数据清单。

    path 可指向 data.json 或其所在目录。返回解析后的绝对路径和清单字典。
    核验协议标识与原有指纹，防止误用旧版不等量划分或被改写的采集顺序；
    不打开 PNG，也不引入 NumPy、Pillow 或 PyTorch，因此纯环境阶段可独立运行。
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "data.json"
    if not path.is_file():
        raise ValueError(f"缺少数据清单：{path}；先运行 --stage data 或 --stage all")
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    keys = ("schema", "settings", "classes", "samples", "clients")
    if not isinstance(manifest, dict) or any(key not in manifest for key in keys):
        raise ValueError("数据清单字段不完整")
    if (not isinstance(manifest["settings"], dict)
            or manifest["settings"].get("protocol") != "code_v1_scene_holdout_iid_equal_v1"):
        raise ValueError("需使用code_v1生成的IID等量采集清单；旧缓存不会自动改写")
    # 与 data.py 使用同一指纹规则：只对固定核心字段做稳定排序后计算SHA-256。
    # 原清单记录的指纹不参与自己的计算，避免形成循环依赖。
    content = {key: manifest[key] for key in keys}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if manifest.get("fingerprint") != digest:
        raise ValueError("数据清单指纹不一致")
    return path, manifest


def build_parser():
    """定义统一命令行；参数未显式传入时，继续使用 config.py/JSON 中的值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "data", "environment"), default="all")
    parser.add_argument("--config", help="可选JSON配置；覆盖config.py中相应字段")
    parser.add_argument("--num-uavs", type=int, help="无人机数量，统一作用于数据与环境")
    parser.add_argument("--samples-per-uav", type=int, help="每架无人机共同的样本上限")
    parser.add_argument("--initial-samples-per-uav", type=int, help="每机初始缓存，包含在样本上限中")
    parser.add_argument("--data-root", help="原始Images.zip、Annotations.zip所在目录")
    parser.add_argument("--data-output", help="数据清单及crops缓存目录")
    parser.add_argument("--output", help="环境CSV、场景快照和轨迹图目录")
    parser.add_argument("--manifest", help="仅environment阶段：指定现有code_v1数据清单")
    return parser


def configuration_from_args(args):
    """合并参数并检查一致性，优先级为命令行 > JSON配置 > CONFIG默认值。

    JSON路径的解析由 load_config 负责；此处的显式命令行路径相对当前工作目录。
    --manifest 只供环境阶段读取既有数据，不能替代 all/data 阶段的数据准备输出。
    """
    config = load_config(args.config)
    # 命令行名称与配置分组逐一映射；使用 is not None，避免将“未传入”误作覆盖值。
    for argument, section, key in (
        ("num_uavs", "mission", "num_uavs"),
        ("samples_per_uav", "data", "samples_per_uav"),
        ("initial_samples_per_uav", "data", "initial_samples_per_uav"),
        ("data_root", "data", "root"), ("data_output", "data", "output"),
        ("output", "output", "scenario_dir"),
    ):
        value = getattr(args, argument)
        if value is not None:
            if argument in ("data_root", "data_output", "output"):
                value = str(Path(value).expanduser().resolve())
            config[section][key] = value
    validate_config(config)
    if args.manifest and args.stage != "environment":
        raise ValueError("--manifest仅适用于environment阶段；all/data使用统一配置准备数据")
    return config


def run(args):
    """按 stage 顺序组织模块，打印概要并返回本次使用/生成的文件路径。

    数据准备结束后先解析真实单机容量、初始缓存和采集速率，再生成环境。
    data 阶段到此返回；all/environment 继续生成并读取环境，核对场景可正常回放。
    """
    config = configuration_from_args(args)
    if args.stage in ("all", "data"):
        # 仅在确实准备图片时导入数据模块；--help和纯环境命令不加载图片依赖。
        from data import prepare_data
        print("准备/核验CBCL数据缓存……", flush=True)
        manifest_path = prepare_data(config)
    else:
        manifest_path = Path(args.manifest) if args.manifest else Path(config["data"]["output"]) / "data.json"
    # 两种入口都经过同一元数据审计，保证生成环境时使用的UAV编号和容量一致。
    manifest_path, manifest = read_manifest_metadata(manifest_path)
    collection = resolve_collection(config, manifest)
    print(f"数据清单：{manifest_path}")
    print(f"训练/验证/测试：{manifest['subset_counts']}")
    print(f"无人机：{collection['num_uavs']}；每机上限：{collection['max_samples_per_uav']}；"
          f"初始缓存：{collection['initial_samples_per_uav']}；采集速率：{collection['rate_samples_per_s']:.8g} 样本/s")
    excluded = manifest.get("allocation_report", {}).get("excluded_train_samples", 0)
    print(f"未纳入本次训练流的训练池样本：{excluded}")
    if args.stage == "data":
        return dict(data_json=str(manifest_path))
    # 先写出/复用场景，再用正式读取器回读；打印的是固定航迹概要，不是训练结果。
    from environment import generate_environment, EnvironmentTrace
    paths = generate_environment(config, manifest_path)
    trace = EnvironmentTrace(paths["environment_csv"], paths["scenario_json"])
    for route in trace.scenario["trajectories"]:
        period = route["loop_length_m"] / route["speed_m_s"]
        print(f"UAV {route['uav_id']}：单圈 {route['loop_length_m']:.3f} m / {period:.3f} s；"
              f"任务内完整巡逻 {int(config['mission']['duration_s'] // period)} 圈")
    for key, path in paths.items():
        print(f"{key}：{path}")
    print(f"环境：{len(trace.times)} 个共同时间点，{len(trace.times) * len(trace.uav_ids)} 行；"
          "按模拟物理时间读取，未启动训练。")
    # 理论预算只做条件性参数计算；不满足理论前提不妨碍轨迹和信道文件生成。
    try:
        budget = theory_budget(config)
        print(f"条件性充分训练预算：{budget['sufficient_epochs']} epoch；"
              f"配置执行 {budget['configured_epochs']} epoch；该参数界尚未针对实际网络验证。")
    except ValueError as error:
        print(f"当前参数不提供理论充分预算：{error}")
    return dict(data_json=str(manifest_path), **paths)


def main(argv=None):
    """脚本入口：argv=None时读取命令行，常见输入错误以退出码2和简明提示报告。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"准备失败：{error}\n")


if __name__ == "__main__":
    # 被其他代码 import 时不会自动准备数据或写入场景文件。
    main()
