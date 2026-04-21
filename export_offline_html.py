"""
将 Streamlit 应用导出为可离线使用的 HTML（基于 stlite）。

执行方式：
    python export_offline_html.py

输出文件：
    offline_streamlit\index.html
"""

import subprocess
import sys
from pathlib import Path


APP_FILE = Path("step3_streamlit_app.py")
OUT_DIR = Path("offline_streamlit")


def run_cmd(cmd):
    print("执行命令:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    if not APP_FILE.exists():
        raise FileNotFoundError(f"未找到应用文件: {APP_FILE}")

    # 安装 stlite（若已安装则跳过）
    run_cmd([sys.executable, "-m", "pip", "install", "stlite"])

    # 导出为离线可交互页面目录（包含 index.html 与运行资源）
    if OUT_DIR.exists():
        for p in OUT_DIR.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil
                shutil.rmtree(p)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            "-m",
            "stlite",
            "export",
            str(APP_FILE),
            "--output",
            str(OUT_DIR),
        ]
    )

    print("\n导出完成！")
    print(f"离线页面入口：{OUT_DIR / 'index.html'}")
    print("说明：直接双击 index.html 或用本地静态服务器打开即可。")


if __name__ == "__main__":
    main()

