#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Update Date:   2024/11/25 修改和中文化 By 陳會安
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np

def _display_detected_frames(conf, model, st_frame, image, task_type):
    """
    使用 YOLO11 模型在影片幀上標記出偵測或追蹤到的物體, 包含圖片分割和姿態評估。
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param st_frame (Streamlit 物件): 用於顯示偵測影片結果的 Streamlit 物件。
    :param image (numpy 陣列): 影片幀的 numpy 陣列。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    # 調整圖片尺寸成為標準尺寸
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    if task_type == "物體追蹤":
        # 用YOLO11模型追蹤影片幀中的物體
        res = model.track(image, persist=True, conf=conf)
    else:
        # 用YOLO11模型偵測影片幀中的物體
        res = model.predict(image, conf=conf)

    # 在影片幀中繪出偵測結果的物體
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='偵測結果影片',
                   channels="BGR",
                   use_container_width=True
                   )


@st.cache_resource
def load_model_with_cache(model_path):
    """
    模型檔不存在, 使用Cache Resource在指定 model_path 目錄下載 YOLO 模型。
    :param model_path (str): YOLO 模型檔案的路徑。
    :return： YOLO 模型。
    """
    model = YOLO(model_path)
    
    return model

def load_model(model_path):
    """
    模型檔已經存在, 不用快取直接載入指定 model_path 目錄下的 YOLO 模型。
    :param model_path (str): YOLO 模型檔案的路徑。
    :return: YOLO 模型。
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
    if task_type == "物體追蹤":
        st.error("物體追蹤任務不支援圖片, 請在側邊欄的來源選擇影片或攝影機")
        return
    
    source_img = st.sidebar.file_uploader(
        label="選擇圖片檔案...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # 新增上傳圖片至網頁和加上說明文字
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
                elif task_type == "圖片分割":
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
                                    for idx,polygon in enumerate(r.masks.xy):
                                        polygon = polygon.astype(np.int32)
                                        x1,y1,x2,y2 = r.boxes.xyxy[idx].cpu().numpy().astype(int)
                                        st.write("圖片分割區域:", polygon)
                                        st.write("圖片方框:", x1, y1, x2, y2)
                                        st.write("------------------------")
                        except Exception as ex:
                            st.write("尚未上傳圖檔!")
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
                    while vid_cap.isOpened():
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
        vid_cap = cv2.VideoCapture(0)  # 本機的Webcam網路攝影機
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
    # 請在tw live: https://tw.live/搜尋台灣即時影像
    # 例如： https://trafficvideo2.tainan.gov.tw/b596d902
    # 例如： https://trafficvideo.tainan.gov.tw/6a0feb9b
    # 在側邊欄中輸入 URL
    url = st.sidebar.text_input("輸入URL")
    if url:
        st.write(f"您輸入的URL是: {url}")
        try:
            flag = st.button(
                label="停止執行"
            )
            vid_cap = cv2.VideoCapture(url)  # IP camera網路監控攝影機
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
