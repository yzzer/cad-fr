from deepface import DeepFace
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将OpenCV的BGR格式转换为RGB格式
    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_objs = DeepFace.extract_faces(
            img_path = frame_np, 
            detector_backend = 'yolov11s',
            enforce_detection=False,
        )

    for face in face_objs:
        facial_area = face['facial_area']
        cv2.rectangle(frame, (facial_area['x'], facial_area['y']), (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), (0, 255, 0), 2)

    # 展示处理后的画面
    cv2.imshow('YOLOv8 Face Detection', frame)
    if cv2.waitKey(1) == 27:  # 按Esc键退出
        break

cap.release()
cv2.destroyAllWindows()