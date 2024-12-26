import cv2
import numpy as np
from deepface import DeepFace
from pydantic import BaseModel
from hashlib import sha256
    

def hash_face(rep: dict) -> str:
    '''
    从人脸的特征向量中生成哈希值
    representation: np.ndarray, 人脸的特征向量
    return: str, 哈希值
    '''
    return sha256((f"{rep['hash']}_{rep['target_x']}_{rep['target_y']}_{rep['target_w']}_{rep['target_h']}").
            encode('utf-8')).hexdigest()

 
def face_extract(frame: np.ndarray, detector_be: str = 'yolov11s', confidence_threshold: float = 0.7, expand: int = 10) -> list[dict[str, any]]:
    '''
    从视频帧中提取人脸
    frame: np.ndarray, 视频帧 numpy array (BGR)
    detector_be: str, 人脸检测模型, 默认 yolov11s
    return: list[FaceObj], 视频帧中人脸的列表
    '''
    
    face_objs = DeepFace.extract_faces(
            img_path = frame, 
            detector_backend = detector_be,
            enforce_detection=False,
            grayscale=False,
            expand_percentage=expand,
        )
    
    results = []
    for face_obj in face_objs:
        
        if face_obj['confidence'] < confidence_threshold:
            continue
        results.append({
            "confidence": face_obj['confidence'],
            "x": face_obj['facial_area']['x'],
            "y": face_obj['facial_area']['y'],
            "w": face_obj['facial_area']['w'],
            "h": face_obj['facial_area']['h'],
        })
    del face_objs
    return results

def cut_faces(frame: np.ndarray, face_objs: list[dict[str, any]]) -> list[np.ndarray]:
    '''
    从视频帧中提取人脸
    frame: np.ndarray, 视频帧 numpy array (BGR)
    face_objs: list[FaceObj], 视频帧中人脸的列表
    return: list[np.ndarray], 视频帧中人脸的列表
    '''
    results = []
    for face_obj in face_objs:
        results.append(frame[face_obj['y']:face_obj['y']+face_obj['h'], face_obj['x']:face_obj['x']+face_obj['w']])
    return results


def resize_faces(face_objs: list[np.ndarray], w: int = 98, h: int = 128) -> list[np.ndarray]:
    '''
    从视频帧中提取人脸
    face_objs: list[np.ndarray], 视频帧中人脸的列表
    size: int, 人脸的大小
    return: list[np.ndarray], 视频帧中人脸的列表
    '''
    results = []
    for face_obj in face_objs:
        width = int(face_obj.shape[0] / h * w)
        results.append(cv2.resize(face_obj, (width, h),
            interpolation=cv2.INTER_AREA))
    face_objs = [cv2.resize(face_obj, (w, h), 
            interpolation=cv2.INTER_AREA) for face_obj in face_objs]
    return face_objs
        




if __name__ == '__main__':
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_objs = face_extract(frame)
        for face in face_objs:
            facial_area = face
            cv2.rectangle(frame, (facial_area["x"], facial_area["y"]), (facial_area["x"] + facial_area["w"], facial_area["y"] + facial_area["h"]), (0, 255, 0), 2)    
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()