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
    
    # from utils.face_extract import hash_face 
    # import pickle
    # with open(settings.pkl_file, "rb") as f:
    #     reps = pickle.load(f)
    # new_reps = []
    # for rep in reps:
    #     if hash_face(rep) not in ['382d5c79157b139e31db4001a700a45e080571d60d039b25a8401f50a5f3b72e', '9b6b8f486800424ff0fe2a85ae20cdc0c2f23099168cdcdc8801144dd27608b9', 'a86a80dae3e5a6f9c887553292d4ac24c21d8f0c946b1f43590f5302380d46ec']:            
    #         new_reps.append(rep)
    # with open(settings.pkl_file, "wb") as f:
    #     pickle.dump(new_reps, f)
    # reps = new_reps
    
    while True:
        face: np.ndarray = queue.get()
        if face is None:
            continue
        
        tik = time.time()
        find_idxes = service.find(face, model_name=settings.presentation_model_name, threshold=settings.presentation_sim_threshold
                                        , detector_backend="skip", method=settings.presentation_distance_metric)
        checkin(find_idxes.keys(), cursur)
        print(find_idxes)
        print(f"search time: {time.time() - tik}s")
        
        # if len(find_idxes) > 0:
        #     new_reps = []
        #     for rep in reps:
        #         if hash_face(rep) not in find_idxes:
        #             new_reps.append(rep)
        #     with open(settings.pkl_file, "wb") as f:
        #         pickle.dump(new_reps, f)
        #     reps = new_reps
        
        

def main():
    cap = cv2.VideoCapture(0)
    buffer = FaceDedupBuffer(threshold=settings.dedup_similarity_threshold, max_windows_size=100)
    queue = Queue()
    
    search_process = Process(target=search_service, args=(queue,))
    search_process.start()
    
    # warm_up_frame_cnt = 30
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_objs = face_extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                detector_be=settings.detector_backend,
                                confidence_threshold=settings.face_detection_threshold,
                                expand=5)
        
        raw_frame = frame.copy()
        for obj in face_objs:
            obj["raw_face"] = cut_faces(raw_frame, [obj])[0]
            obj["face"] = cv2.cvtColor(obj["raw_face"], cv2.COLOR_BGR2RGB)
            if buffer.add_face(obj, model_name="Facenet512"):
                queue.put(obj["face"])
        
        face_h = 128
        face_w = 98
        
        face_imgs = resize_faces(buffer.get_faces(), h=face_h, w=face_w)
        
        # if len(face_objs) > 0 and warm_up_frame_cnt > 0:
        #     warm_up_frame_cnt -= 1
        #     continue
        
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
    
    cap.release()
    cv2.destroyAllWindows()
    queue.close()
    search_process.terminate()
    
    
if __name__ == "__main__":
    main()