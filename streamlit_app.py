#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Update Date:   2024/11/22 修改和中文化 By 陳會安
   @Description:
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
import streamlit as st
import sys
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile

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

# YOLO 物體偵測模型設置
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
# YOLO 姿態評估模型設置
POSE_MODEL_DIR = ROOT / 'weights' / 'pos'

DETECTION_MODEL_LIST = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "best.pt"]

POSE_MODEL_LIST = [
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
    "best.pt"]

def _display_detected_frames(conf, model, st_frame, image, task_type):
    """
    使用 YOLO11 模型在影片幀上標記出偵測到的物體。
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param st_frame (Streamlit 物件): 用於顯示偵測影片結果的 Streamlit 物件。
    :param image (numpy 陣列): 影片幀的 numpy 陣列。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
       
    # Predict the objects in the image using YOLO11 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='偵測結果影片',
                   channels="BGR",
                   use_container_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    在指定 model_path 目錄下載 YOLO 模型。

    參數：

    model_path (str): YOLO 模型檔案的路徑。

    返回： YOLO 模型。
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model, task_type):
    """
    執行上傳圖片的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    source_img = st.sidebar.file_uploader(
        label="選擇圖片檔案...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="上傳圖片",
                use_container_width=True
            )

    if source_img:
        if st.button("執行"):
            with st.spinner("執行中..."):
                if task_type == "物體偵測": 
                    res = model.predict(uploaded_image,
                                        conf=conf)                    
                    res_plotted = res[0].plot()[:, :, ::-1]
                    with col2:
                        st.image(res_plotted,
                                 caption="偵測結果圖片",
                                 use_container_width=True)
                        try:
                            with st.expander("偵測結果"):
                                #boxes = res[0].boxes
                                #for box in boxes:
                                #    st.write(box.xywh)
                                for box in res[0].boxes:
                                    cords = box.xyxy[0].tolist()
                                    cords = [round(x) for x in cords]
                                    class_id = int(box.cls[0].item())
                                    conf = box.conf[0].item()
                                    conf = round(conf*100, 2)
                                    st.write("分類編號:", class_id)
                                    st.write("分類名稱:", model.names[class_id])
                                    st.write("方框座標:", cords)
                                    st.write("信心指數:", conf, "%")
                                    st.write("------------------------")    
                        except Exception as ex:
                            st.write("尚未有圖檔上傳!")
                            st.write(ex)
                elif task_type == "姿態評估":
                    res = model.predict(uploaded_image,
                                        conf=conf)
                    res_plotted = res[0].plot()[:, :, ::-1]
                    with col2:
                        st.image(res_plotted,
                                 caption="偵測結果圖片",
                                 use_container_width=True)
                        try:
                            with st.expander("偵測結果"):
                                for r in res:
                                    boxes = r.boxes
                                    kps = r.keypoints
                                    for idx, p in enumerate(kps):
                                        list_p = p.data.tolist()
                                        st.write("姿態評估偵測的17個關鍵點:", idx)
                                        for i, point in enumerate(list_p[0]):
                                            st.write(f"關鍵點: {i}: ({int(point[0])}, {int(point[1])})")
                                        st.write("------------------------")    
                                            
                        except Exception as ex:
                            st.write("尚未上傳圖檔!")
                            st.write(ex)                    

def infer_uploaded_video(conf, model, task_type):
    """
    執行上傳影片的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    source_video = st.sidebar.file_uploader(
        label="選擇上傳影片檔..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("執行"):
            with st.spinner("執行中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image,
                                                     task_type
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"載入影片錯誤: {e}")


def infer_uploaded_webcam(conf, model, task_type):
    """
    執行網路攝影機Webcam的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    try:
        flag = st.button(
            label="停止執行"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image,
                    task_type
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"載入網路攝影機錯誤: {str(e)}")
        
def infer_uploaded_IPcam(conf, model, task_type):
    """
    執行網路監控攝影機IP Camera的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    # tw live可搜尋台灣即時影像：https://trafficvideo2.tainan.gov.tw/b596d902
    # 在側邊欄中輸入 URL
    url = st.sidebar.text_input("輸入URL")
    if url:
        st.write(f"您輸入的URL是: {url}")
        try:
            flag = st.button(
                label="停止執行"
            )
            vid_cap = cv2.VideoCapture(url)  # IP camera
            st_frame = st.empty()
            while not flag:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        task_type
                    )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.error(f"載入網路攝影機錯誤: {str(e)}")        
    else:
        st.error(f"請在側邊欄[輸入URL]欄位輸入網路監控攝影機IP Camera的URL網址後, 按Enter鍵")


# 設定頁面佈局
st.set_page_config(
    page_title="Ultralytics YOLO11 模型任務的互動介面",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# 主頁面標題
st.title("Ultralytics YOLO11 模型任務的互動介面")

# 側邊欄
st.sidebar.header("YOLO模型設置")

# 作業選項
task_type = st.sidebar.selectbox(
    "選擇任務",
    ["物體偵測",
     "姿態評估"]
)
# 模型選項
model_type = None
if task_type == "物體偵測":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        DETECTION_MODEL_LIST
    )
elif task_type == "姿態評估":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        POSE_MODEL_LIST
    )
else:
    st.error("目前已經實作 '物體偵測' 和 '姿態評估' 功能")
# 信心指數選項
confidence = float(st.sidebar.slider(
    "選擇信心指數", 30, 100, 50)) / 100
# 取得模型儲存的路徑
model_path = ""
if model_type:
    if task_type == "物體偵測":
        model_path = Path(DETECTION_MODEL_DIR, str(model_type))
    elif task_type == "姿態評估":
        model_path = Path(POSE_MODEL_DIR, str(model_type))
else:
    st.error("請在側邊欄選擇模型種類")

# 載入YOLO預訓練模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"無法載入模型. 請確認模型路徑: {model_path}")

# 圖片/影片選項
st.sidebar.header("圖片/影片設置")
source_selectbox = st.sidebar.selectbox(
    "選擇來源",
    SOURCES_LIST
)

source_img = None
if source_selectbox == SOURCES_LIST[0]:   # Image
    infer_uploaded_image(confidence, model, task_type)
elif source_selectbox == SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model, task_type)
elif source_selectbox == SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model, task_type)
elif source_selectbox == SOURCES_LIST[3]: # IPcam
    infer_uploaded_IPcam(confidence, model, task_type)    
else:
    st.error("目前資料來源支援'圖片' ,'影片' ,'網路攝影機' 和 '網路監控攝影機'")