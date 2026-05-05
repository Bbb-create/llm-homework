import streamlit as st
from PIL import Image
from src.face_utils import detect_faces, draw_face_boxes, recognize_faces, load_known_faces
import os
import cv2
import numpy as np

# 页面配置
st.set_page_config(page_title="人脸检测与识别", layout="wide")
st.title("🔍 基于face_recognition的人脸检测与识别")
st.markdown("---")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 功能设置")
    task_mode = st.radio("选择功能模式", ["仅人脸检测", "人脸检测+识别"])

    if task_mode == "人脸检测+识别":
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
        known_encodings, known_names = load_known_faces("known_faces")
        if len(known_names) == 0:
            st.warning("known_faces 文件夹还没有图片，无法识别")
        else:
            st.success(f"已加载 {len(known_names)} 个已知人脸：{known_names}")

# 图片上传
uploaded_file = st.file_uploader("📤 上传图片（JPG/PNG）", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", use_column_width=True)
    st.markdown("---")

    if st.button("🚀 开始处理", use_container_width=True):
        with st.spinner("正在处理..."):
            if task_mode == "仅人脸检测":
                face_locations = detect_faces(image)
                if len(face_locations) == 0:
                    st.error("❌ 未检测到人脸")
                else:
                    st.success(f"✅ 检测到 {len(face_locations)} 张人脸")
                    result_img = draw_face_boxes(image, face_locations)
                    st.image(result_img, caption="检测结果", use_column_width=True)
            else:
                if len(known_names) == 0:
                    st.error("❌ 请先在 known_faces 文件夹放入已知人脸图片")
                else:
                    face_locations, face_names = recognize_faces(image, known_encodings, known_names)
                    if len(face_locations) == 0:
                        st.error("❌ 未检测到人脸")
                    else:
                        st.success(f"✅ 检测到 {len(face_locations)} 张人脸")
                        img_array = np.array(image)
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(img_array, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        result_img = Image.fromarray(img_array)
                        st.image(result_img, caption="识别结果", use_column_width=True)