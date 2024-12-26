import cv2
import time
import numpy as np
from cad_fr.config import settings
from utils.face_extract import face_extract, cut_faces, resize_faces
from utils.search import FaceDedupBuffer, FaceSearchService

from multiprocessing import Process, Queue
from utils.database import get_db_conn, checkin


def search_service(queue: Queue):
    service = FaceSearchService(db_path=settings.pkl_file, workers=settings.workers, warm_up_time_ms=settings.worker_warm_up_time_ms)
    conn = get_db_conn(settings.db_file)
    cursur = conn.cursor()
    
    while True:
        face: np.ndarray = queue.get()
        if face is None:
            continue
        
        tik = time.time()
        find_idxes = service.find(face, model_name=settings.presentation_model_name, threshold=settings.presentation_sim_threshold, detector_backend="skip")
        checkin(find_idxes, cursur)
        print(find_idxes)
        print(f"search time: {time.time() - tik}s")
        
        

def main():
    cap = cv2.VideoCapture(0)
    buffer = FaceDedupBuffer(threshold=settings.dedup_similarity_threshold, max_windows_size=100)
    queue = Queue()
    
    search_process = Process(target=search_service, args=(queue,))
    search_process.start()
    
    warm_up_frame_cnt = 10
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_objs = face_extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                detector_be=settings.detector_backend,
                                confidence_threshold=settings.face_detection_threshold,
                                expand=settings.presentation_expand_percentage)
        
        raw_frame = frame.copy()
        for obj in face_objs:
            obj["raw_face"] = cut_faces(raw_frame, [obj])[0]
            obj["face"] = cv2.cvtColor(obj["raw_face"], cv2.COLOR_BGR2RGB)
            if buffer.add_face(obj, model_name=settings.presentation_model_name):
                queue.put(obj["face"])
        
        face_h = 128
        face_w = 98
        
        face_imgs = resize_faces(buffer.get_faces(), h=face_h, w=face_w)
        
        if len(face_objs) > 0 and warm_up_frame_cnt > 0:
            warm_up_frame_cnt -= 1
            continue
        
        for face in face_objs:
            facial_area = face
            cv2.rectangle(frame, (facial_area["x"], facial_area["y"]), (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), (0, 255, 0), 2)    
        
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
    search_process.terminate()
    queue.close()
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()