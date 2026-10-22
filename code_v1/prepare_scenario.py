"""统一入口：先准备数据，再按实际容量生成固定飞行环境；不启动训练。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config import load_config, resolve_collection, theory_budget, validate_config


def read_manifest_metadata(path):
    """标准库审计清单，不导入数据准备依赖，也不读取PNG像素。"""
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
    content = {key: manifest[key] for key in keys}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if manifest.get("fingerprint") != digest:
        raise ValueError("数据清单指纹不一致")
    return path, manifest


def build_parser():
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
    config = load_config(args.config)
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
    config = configuration_from_args(args)
    if args.stage in ("all", "data"):
        # 环境命令和--help均不依赖NumPy/Pillow/PyTorch。
        from data import prepare_data
        print("准备/核验CBCL数据缓存……", flush=True)
        manifest_path = prepare_data(config)
    else:
        manifest_path = Path(args.manifest) if args.manifest else Path(config["data"]["output"]) / "data.json"
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
    try:
        budget = theory_budget(config)
        print(f"条件性充分训练预算：{budget['sufficient_epochs']} epoch；"
              f"配置执行 {budget['configured_epochs']} epoch；该参数界尚未针对实际网络验证。")
    except ValueError as error:
        print(f"当前参数不提供理论充分预算：{error}")
    return dict(data_json=str(manifest_path), **paths)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"准备失败：{error}\n")


if __name__ == "__main__":
    main()
