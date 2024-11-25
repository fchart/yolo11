#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Update Date:   2024/11/25 修改和中文化 By 陳會安                   
   @Description:   已經新增"物體追蹤","圖片分割"和"姿態評估"功能
                   更清楚和完整顯示圖片偵測結果
                   在資料來源新增IP Camera網路監控攝影機
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
import streamlit as st
import os

import config
from utils import load_model, load_model_with_cache, infer_uploaded_image, infer_uploaded_video
from utils import infer_uploaded_webcam, infer_uploaded_IPcam

# 設定頁面佈局
st.set_page_config(
    page_title="Ultralytics YOLO11 模型任務的互動介面",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# 主頁面標題, 顯示主標題
st.title("Ultralytics YOLO11 模型的互動介面")

# 側邊欄
st.sidebar.header("YOLO模型設置")

# 作業選項
pre_task_type = None
task_type = st.sidebar.selectbox(
    "選擇任務",
    ["物體偵測",
     "物體追蹤",
     "圖片分割",     
     "姿態評估"]
)
# 顯示次標題
st.subheader(task_type)
# 新增一條水平線
st.markdown("---")
# 模型選項
model_type = None
if task_type == "物體偵測" or task_type == "物體追蹤":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        config.DETECTION_MODEL_LIST
    )
elif task_type == "圖片分割":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        config.SEGMENT_MODEL_LIST
    )
elif task_type == "姿態評估":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        config.POSE_MODEL_LIST
    )
else:
    st.error("目前已經實作 '物體偵測' ,'物體追蹤', '圖片分割'和 '姿態評估' 功能")
# 信心指數選項
confidence = float(st.sidebar.slider(
    "選擇信心指數", 30, 100, 50)) / 100
# 取得模型儲存的路徑
model_path = ""
if model_type:
    if task_type == "物體偵測" or task_type == "物體追蹤":
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    elif task_type == "圖片分割":
        model_path = Path(config.SEGMENT_MODEL_DIR, str(model_type))        
    elif task_type == "姿態評估":
        model_path = Path(config.POSE_MODEL_DIR, str(model_type))
else:
    st.error("請在側邊欄選擇模型種類")

# 載入YOLO預訓練模型或客製化模型best.pt
try:
    # 判斷模型檔案是否存在
    if os.path.exists(model_path):
        model = load_model(model_path)             # 存在, 直接載入模型檔
    else:
        model = load_model_with_cache(model_path)  # 不存在, 用快取下載和載入模型檔
except Exception as e:
    st.error(f"無法載入模型. 請確認模型路徑: {model_path}")

# 圖片/影片設置選項
st.sidebar.header("圖片/影片設置")
# 篩選資料來源, 因為物體追蹤的資料來源沒有'圖片'
items_list = config.SOURCES_LIST[1:] if task_type == "物體追蹤" else config.SOURCES_LIST
source_selectbox = st.sidebar.selectbox(
    "選擇來源",
    items_list
)
# 針對資料來源來上傳圖片/影片和依據task_type來執行相關任務
source_img = None
if source_selectbox == config.SOURCES_LIST[0]:   # Image
    infer_uploaded_image(confidence, model, task_type)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model, task_type)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model, task_type)
elif source_selectbox == config.SOURCES_LIST[3]: # IPcam
    infer_uploaded_IPcam(confidence, model, task_type)    
else:
    if task_type == "物體追蹤":
        st.error("目前資料來源支援'影片' ,'網路攝影機' 和 '網路監控攝影機'")
    else:
        st.error("目前資料來源支援'圖片' ,'影片' ,'網路攝影機' 和 '網路監控攝影機'")
