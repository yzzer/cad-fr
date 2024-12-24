import cv2
import numpy as np
from utils.face_extract import face_extract, cut_faces, resize_faces

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    face_objs = face_extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    face_h = 128
    face_w = 98
    face_imgs = resize_faces(cut_faces(frame, face_objs), h=face_h, w=face_w)
    
    for face in face_objs:
        facial_area = face
        cv2.rectangle(frame, (facial_area.x, facial_area.y), (facial_area.x + facial_area.w, facial_area.y + facial_area.h), (0, 255, 0), 2)    
    
    # 合并帧
    total_width = frame.shape[1]
    total_height = face_h
    bottom_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    x_offset = 0
    for face_img in face_imgs:
        bottom_frame[:, x_offset:x_offset+face_w] = face_img
        x_offset += face_w + face_w // 2
    
    upper_frame = frame[:-total_height, :]
    final_frame = np.vstack((upper_frame, bottom_frame))
    
    cv2.imshow('frame', final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()