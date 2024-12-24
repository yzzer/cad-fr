from pydantic import BaseSettings

class Settings(BaseSettings):
    db_file: str = 'config/db.sqlite3'
    pkl_file: str = 'config/df_model.pkl'
    photo_db_dir: str = 'cad-db'
    dedup_similarity_threshold: float = 0.8
    detector_backend: str = 'yolov11s'
    face_detection_threshold: float = 0.8
    

settings = Settings()