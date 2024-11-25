#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Update Date:   2024/11/25 修改和中文化 By 陳會安
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# 取得目前檔案的絕對路徑
file_path = Path(__file__).resolve()

# 取得目前檔案的父路徑
root_path = file_path.parent

# 如果根路徑不存在 sys.path 串列, 在 sys.path路徑加上根路徑
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 取得根目錄和目前工作目錄的相對目錄
ROOT = root_path.relative_to(Path.cwd())

# 資料來源
SOURCES_LIST = ["圖片", "影片", "網路攝影機Webcam", "網路監控攝影機IPcam"]

# YOLO 物體偵測模型路徑設置
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
# YOLO 圖片分割模型路徑設置
SEGMENT_MODEL_DIR = ROOT / 'weights' / 'segment'
# YOLO 姿態評估模型路徑設置
POSE_MODEL_DIR = ROOT / 'weights' / 'pos'
# YOLO模型選項
DETECTION_MODEL_LIST = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "best.pt"]

SEGMENT_MODEL_LIST = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
    "best.pt"]

POSE_MODEL_LIST = [
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
    "best.pt"]
