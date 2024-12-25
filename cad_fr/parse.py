from hashlib import sha256
import pickle
from cad_fr.config import settings
import sqlite3
from sqlite3 import Cursor
from cad_fr.utils.face_extract import hash_face

def insert_face(id: str, file_id: str, file_name: str, target_x: int, target_y: int, target_w: int, target_h: int, flag: int, cursor: Cursor) -> None:
    cursor.execute('''
    INSERT INTO face (id, file_id, file_name, target_x, target_y, target_w, target_h, flag)
                   VALUES (?,?,?,?,?,?,?,?)
    ''', (id, file_id, file_name, target_x, target_y, target_w, target_h, flag))
    if target_x == 0 and target_y == 0 and target_w == 0 and target_h == 0:
        raise Exception('target_x, target_y, target_w, target_h cannot be 0')
    

def clear_flag(cursor: Cursor) -> None:
    cursor.execute('''
    UPDATE face SET flag = 0
    ''')


def parse():
    pkl_file = settings.pkl_file
    db_file = settings.db_file
    
    import os
    if os.path.exists(db_file):
       os.remove(db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face (
        id TEXT PRIMARY KEY,
        file_id TEXT,
        file_name TEXT,
        target_x INTEGER,
        target_y INTEGER,
        target_w INTEGER,
        target_h INTEGER,
        flag INTEGER,
        checkin_time BIGINT DEFAULT 0
    )
    ''')

    with open(pkl_file, 'rb') as f:
        representations = pickle.load(f)
    for rep in representations:
        id: str = hash_face(rep)
        file_id: str = rep['hash']
        file_name: str = rep['identity']
        flag: int = 0  # 0: 不展示, 1: 展示
        insert_face(id, file_id, file_name, rep['target_x'], rep['target_y'], rep['target_w'], rep['target_h'], flag, cursor)

    conn.commit()
    conn.close()

def clear():
    db_file = settings.db_file
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    clear_flag(cursor)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    import sys
    param = sys.argv
    if 'parse' in param:
        parse()
        print('parse done')
    elif 'clear' in param:
        clear()
        print('clear done')

