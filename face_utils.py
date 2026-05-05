import face_recognition
import cv2
import numpy as np
from PIL import Image
import os

def detect_faces(image):
    """检测图片中的人脸位置"""
    img_array = np.array(image)
    rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    return face_locations

def draw_face_boxes(image, face_locations):
    """在图片上画人脸框"""
    img_array = np.array(image)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
    return Image.fromarray(img_array)

def get_face_encodings(image, face_locations):
    """获取人脸128维特征编码"""
    img_array = np.array(image)
    rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return encodings

def load_known_faces(known_faces_dir="known_faces"):
    """加载已知人脸库"""
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(filename)[0]
            path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            known_names.append(name)
    return known_encodings, known_names

def recognize_faces(image, known_encodings, known_names, tolerance=0.6):
    """识别人脸"""
    img_array = np.array(image)
    rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        face_names.append(name)
    return face_locations, face_names