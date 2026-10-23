"""作用：创建带本地时间和当前Git哈希的独立输出目录，记录源代码版本。
运行指令：由 train_sfl.py、optimize_sfl.py 调用，不单独启动实验。
目录格式为 output_YYYYMMDD_HHMMSS_微秒_12位提交哈希；没有Git时明确标记nogit。
"""
from datetime import datetime
from pathlib import Path
import subprocess
from config import ROOT

def create_run_directory(parent):
    """在输出父目录下原子创建独立运行目录，不覆盖同名文件或已有实验。

    时间戳使用运行机器的本地时间并精确到微秒；极少数重名时附加序号。
    mkdir(exist_ok=False) 使同一时刻启动的两个进程也不会共用结果目录。
    """
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    revision = source_revision()["short_hash"]
    for suffix in range(10000):
        name = f"output_{stamp}_{revision}" + (f"_{suffix:02d}" if suffix else "")
        candidate = parent / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("无法分配唯一运行目录，请检查输出父目录")

def source_revision():
    """读取运行时提交及代码脏状态；没有 Git 的复制件明确标注 nogit。"""
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal",
                                "--", "*.py", "*.txt", "*.md"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=commit, short_hash=commit[:12], code_modified=bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, short_hash="nogit", code_modified=None)
