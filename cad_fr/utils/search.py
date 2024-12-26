
from deepface import DeepFace
from deepface.DeepFace import representation, verification, detection
from .face_extract import hash_face
import numpy as np
from typing import List, Dict, Any
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity

_local_hash_index = []
_local_emebddings = []
_checkin_status = []
def subprocess_init(hash_index: list[str], embeddings: list[any], warm_up_time_ms: int = 300):
    global _local_hash_index
    global _local_emebddings
    global _checkin_status
    _local_hash_index = []
    _local_emebddings = []
    _checkin_status = []
    for i in range(len(hash_index)):
        if embeddings[i] is not None and len(embeddings[i].shape) == 1:
            _local_hash_index.append(hash_index[i])
            _local_emebddings.append(embeddings[i])
            _checkin_status.append(False)
    
    import time
    time.sleep(warm_up_time_ms / 1000)


def find_local(i, face_embedding: np.ndarray, threshold: float = 0.3, workers: int = 5, method: str = "cosine") -> List[str]:
    global _local_hash_index
    global _local_emebddings
    global _checkin_status
    
    
    if workers == 1:
        start = 0
        end = len(_local_hash_index)
    else:
        start = i * len(_local_hash_index) // workers
        end = (i + 1) * len(_local_hash_index) // workers if i < workers - 1 else len(_local_hash_index)

    match_dict = {}
    for i in range(start, end):
        if _checkin_status[i]:
            continue
        distance = verification.find_cosine_distance(face_embedding, _local_emebddings[i])
        if distance <= threshold:
            match_dict[_local_hash_index[i]] = 1 - distance
            _checkin_status[i] = True
    return match_dict


class FaceSearchService:
    '''
    扩展deepface的find, 使用内存加速检索, 支持灵活的检索能力变更
    '''
    def __init__(self, db_path: str = None, workers: int = 5, warm_up_time_ms: int = 300):
        self.hash_index = []
        self.embeddings = []
        self.load_db(db_path)
        
        # init process pool
        # self.workers = Pool(workers)
        # self.workers_num = workers
        # self.__init_pool(workers, warm_up_time_ms)
        if db_path is not None:
            subprocess_init(self.hash_index, self.embeddings, warm_up_time_ms)
        
        
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
            self.workers.apply(subprocess_init, args=(self.hash_index, self.embeddings, warm_up_time_ms))
               
    
    def find(self, face: np.ndarray, model_name: str = "Facenet512", method: str = "cosine", threshold: float = 0.3, detector_backend="yolov11s", expand: int = 10) -> List[str]:
        source_objs = None
        if detector_backend == "skip":
            source_objs = [{
                "face": face,
            }]
        else:
            source_objs = self.extract_face(face, detector_backend=detector_backend, expand=expand)
            
        match_sim = {}
        for source_obj in source_objs:
            face_embedding = self.emebedding_face(source_obj["face"], model_name=model_name)
            local_result = find_local(-1, face_embedding, threshold, 1, method)
            if len(local_result) > 0:
                match_sim.update(local_result)
        return match_sim

    
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
        # subprocess_init(self.hash_index, self.embeddings)
        self.faces = []
        
        
    def find(self, face_embedding: np.ndarray) -> list[str]:
        for i in range(len(self.hash_index) - 1, -1, -1):
            hash, embedding = self.hash_index[i], self.embeddings[i]
            if verification.find_cosine_distance(face_embedding, embedding) <= self.threshold:
                return [hash]
        return []
        
        
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
            self.faces.pop(0)
        return True
            
    def get_faces(self, limit: int = 10):
        return list(reversed(self.faces[-limit:]))
    

    