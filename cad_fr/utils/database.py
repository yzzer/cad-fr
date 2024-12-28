import sqlite3
import time

def get_db_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    return conn

def checkin(ids: list[str], cursor: sqlite3.Cursor):
    print(ids)
    for id in ids:
        cursor.execute('''
            UPDATE face SET checkin_time = ?, flag = 1  WHERE id = ?
            ''', (int(time.time() * 1000), id))
    cursor.connection.commit()

if __name__ == "__main__":
    conn = get_db_conn("../../config/db.sqlite3")
    cursor = conn.cursor()
    checkin(['791d8e9e43924617449b866431742ee59b8cd1aa6141140b22e02dac6a2c1f82'], cursor)
    # conn.commit()
    conn.close()