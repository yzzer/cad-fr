from ultralytics import YOLO
import cv2

# 加载YOLOv8人脸检测模型，这里以官方预训练模型为例
model = YOLO('yolov11s-face.pt')

# 打开摄像头（0表示默认摄像头，你也可以替换成视频文件路径来检测视频中的人脸）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型进行推理
    results = model(frame)

    # 遍历检测结果，绘制人脸的检测框
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 展示处理后的画面
    cv2.imshow('YOLOv8 Face Detection', frame)
    if cv2.waitKey(1) == 27:  # 按Esc键退出
        break

cap.release()
cv2.destroyAllWindows()