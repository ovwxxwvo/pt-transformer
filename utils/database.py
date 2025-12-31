import os
import sqlite3


class TrainDB:
    def __init__(self, db_path="data/pt_train.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.cursor().execute('''
            CREATE TABLE IF NOT EXISTS train_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER,
                loss REAL,
                bleu REAL,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_log(self, epoch, loss, bleu):
        self.conn.cursor().execute(
            "INSERT INTO train_logs (epoch, loss, bleu) VALUES (?, ?, ?)",
            (epoch, loss, bleu)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


db = TrainDB()


