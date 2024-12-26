
from deepface import DeepFace
from deepface.DeepFace import representation, verification, detection
from .face_extract import hash_face
import numpy as np
from typing import List, Dict, Any
from multiprocessing import Pool

def query(source_img, db_path, model_name, detector_backend="centerface", threshold: float = 0.3) -> list[str]:
    recall_results = DeepFace.find(
        img_path=source_img,
        db_path=db_path,
        enforce_detection=False,  # 让source里没有人脸，也不报错
        threshold=threshold,  # 过滤掉不相似图片的阈值
        detector_backend=detector_backend,
        model_name=model_name
    )   
    results = []
    for face_obj in recall_results:
        results.append(hash_face(face_obj))
    return results

_local_hash_index = []
_local_emebddings = []
def subprocess_init(hash_index: list[str], embeddings: list[any], warm_up_time_ms: int = 300):
    global _local_hash_index
    global _local_emebddings
    _local_hash_index = hash_index
    _local_emebddings = embeddings
    
    import time
    time.sleep(warm_up_time_ms / 1000)


def find_local(i, face_embedding: np.ndarray, threshold: float = 0.3, workers: int = 5) -> List[str]:
    global _local_hash_index
    global _local_emebddings
    
    if i == -1:
        start = 0
        end = len(_local_hash_index)
    else:
        start = i * len(_local_hash_index) // workers
        end = (i + 1) * len(_local_hash_index) // workers if i < workers - 1 else len(_local_hash_index)

    results = []
    for idx in range(start, end):
        if _local_emebddings[idx].shape != face_embedding.shape:
            continue
        distance = verification.find_distance(
            face_embedding, _local_emebddings[idx], "cosine"
        )
        if distance <= threshold:
            results.append(_local_hash_index[idx])
    return results 


class FaceSearchService:
    '''
    扩展deepface的find, 使用内存加速检索, 支持灵活的检索能力变更
    '''
    def __init__(self, db_path: str = None, workers: int = 5, warm_up_time_ms: int = 300):
        self.hash_index = []
        self.embeddings = []
        self.load_db(db_path)
        
        # init process pool
        self.workers = Pool(workers)
        self.workers_num = workers
        self.__init_pool(workers, warm_up_time_ms)
        
        
    def __del__(self):
        if hasattr(self, "workers"):
            try:
                self.workers.terminate()
                self.workers.close()
            except Exception as e:
                pass
            
    def load_db(self, db_path: str):
        if db_path is None:
            return
        import pickle
        representations = pickle.load(open(db_path, "rb"))
        for rep in representations:
            if "embedding" not in rep:
                print(f"face {rep['identity']} has no embedding")
                continue
            self.hash_index.append(hash_face(rep))
            self.embeddings.append(np.array(rep["embedding"]))
            
    def __init_pool(self, workers: int = 5, warm_up_time_ms: int = 300):
        for i in range(workers):
            self.workers.apply_async(subprocess_init, args=(self.hash_index, self.embeddings, warm_up_time_ms))
               
    
    def find(self, face: np.ndarray, model_name: str = "Facenet512", threshold: float = 0.3, detector_backend="yolov11s", expand: int = 10) -> List[str]:
        source_objs = None
        if detector_backend == "skip":
            source_objs = [{
                "face": face,
            }]
        else:
            source_objs = self.extract_face(face, detector_backend=detector_backend, expand=expand)
            
        results = set()
        for source_obj in source_objs:
            face_embedding = self.emebedding_face(source_obj["face"], model_name=model_name)
            params = [(i, face_embedding, threshold, self.workers_num) for i in range(self.workers_num)]
            local_results = self.workers.starmap(find_local, params)
            for local_result in local_results:
                if len(local_result) > 0:
                    results.update(local_result)
        return list(results)

    
    def extract_face(self, face: np.ndarray, detector_backend="yolov11s", expand: int = 15) -> List[Dict[str, Any]]:
        face_objs = detection.extract_faces(face, detector_backend=detector_backend, expand_percentage=expand, enforce_detection=False, grayscale=False)
        processed_objs = []
        for face_obj in face_objs:
            processed_objs.append({
                "face": face_obj["face"],
            })
        return processed_objs
    
    def emebedding_face(self, face: np.ndarray, model_name: str = "Facenet512") -> np.ndarray:
        source_embedding_obj = representation.represent(
            img_path=face,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",
        )
        return np.array(source_embedding_obj[0]["embedding"])
    
    

class FaceDedupBuffer(FaceSearchService):
    
    def __init__(self, threshold: float = 0.8, max_windows_size: int = 100):
        super().__init__(workers=1)
        self.threshold = threshold
        self.max_windows_size = max_windows_size
        subprocess_init(self.hash_index, self.embeddings)
        self.faces = []
        
        
    def find(self, face_embedding: np.ndarray) -> list[str]:
        results = find_local(-1, face_embedding, self.threshold)
        return results
        
        
    def add_face(self, face: dict, model_name: str = "Facenet512") -> bool:
        embedding = self.emebedding_face(face["face"], model_name=model_name)
        hash = id(embedding)
        if len(self.find(embedding)) > 0:
            return False
        self.hash_index.append(hash)
        self.embeddings.append(embedding)
        self.faces.append(face["raw_face"])
        if len(self.hash_index) > self.max_windows_size:
            self.hash_index.pop(0)
            self.embeddings.pop(0)
            self.face.pop(0)
        return True
            
    def get_faces(self, limit: int = 10):
        return list(reversed(self.faces[-limit:]))
    

    