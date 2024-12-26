from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_path: str = 'cad_db'
    demo_img_path: str = 'config/2014.jpg'
    db_file: str = 'config/db.sqlite3'
    pkl_file: str = 'config/ds_model.pkl'
    photo_db_dir: str = 'cad-db'
    dedup_similarity_threshold: float = 0.42
    detector_backend: str = 'yolov11s'
    face_detection_threshold: float = 0.8
    presentation_model_name: str = 'Facenet512'
    # presentation_face_detector: str = 'centerface'
    presentation_face_detector: str = 'yolov11m'
    presentation_expand_percentage: int = 10
    presentation_sim_threshold: float = 0.22
    workers: int = 3
    worker_warm_up_time_ms: int = 300
    
def read_yaml_config():
    import os
    workdir = os.environ["WORKDIR"]
    config_file = os.path.join(workdir, "config", "config.yaml")
    with open(config_file, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
        return Settings(**config)
    
settings = read_yaml_config()