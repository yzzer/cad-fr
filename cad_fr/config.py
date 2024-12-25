from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_file: str = 'config/db.sqlite3'
    pkl_file: str = 'config/ds_model.pkl'
    photo_db_dir: str = 'cad-db'
    dedup_similarity_threshold: float = 0.4
    detector_backend: str = 'yolov11s'
    face_detection_threshold: float = 0.8
    presentation_model_name: str = 'Facenet512'
    # presentation_face_detector: str = 'centerface'
    presentation_face_detector: str = 'yolov11m'
    presentation_expand_percentage: int = 15
    presentation_sim_threshold: float = 0.25
    workers: int = 5
    worker_warm_up_time_ms: int = 300
    
    


settings = Settings()