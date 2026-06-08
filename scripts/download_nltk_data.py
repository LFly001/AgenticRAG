#!/usr/bin/env python3
"""NLTK 数据预下载脚本 — 容器构建前在宿主机执行一次即可"""
import nltk
import os

TARGET_DIR = os.environ.get("NLTK_DATA_DIR", "./nltk_data")

PACKAGES = ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"]

print(f"下载 NLTK 数据到: {os.path.abspath(TARGET_DIR)}")
for pkg in PACKAGES:
    print(f"  [{pkg}] ...", end=" ", flush=True)
    nltk.download(pkg, download_dir=TARGET_DIR, quiet=True)
    print("✓")

print("✅ 完成")
