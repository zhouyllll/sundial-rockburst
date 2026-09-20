"""Explicit opt-in model download. This file is never called by build/predict."""

import argparse
import re


def main():
    parser = argparse.ArgumentParser(description="下载固定提交的 Sundial 权重和自定义代码")
    parser.add_argument("--revision", required=True, help="Hugging Face 模型仓库的完整40位提交SHA")
    parser.add_argument("--output", default="models/sundial-base-128m")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("revision 必须是模型仓库完整提交SHA，不能使用 main 或 GitHub 示例仓库的SHA")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="thuml/sundial-base-128m", revision=args.revision,
                             local_dir=args.output,
                             allow_patterns=["*.json", "*.py", "*.safetensors", "*.bin",
                                             "README.md", "LICENSE*"])
    print(f"已下载到 {path}。运行时仅从此本地目录加载，不自动联网。")


if __name__ == "__main__":
    main()
