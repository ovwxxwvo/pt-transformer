import os, pathlib, sqlite3
from .version import get_version_str


proj_root = pathlib.Path(__file__).parent.parent
db_dir = os.path.join(proj_root, "database")
db_file = os.path.join(db_dir, "metric.db")
os.makedirs(db_dir, exist_ok=True)

class MetricDB:
    def __init__(self):
        self.conn = None
        self._epoch_states = {
            "train": {"last_global": 0},
            "eval": {"last_global": 0},
            "test": {"last_global": 0}
        }
        self._connect_db()
        self._create_table()
        self._init_last_global()

    def _connect_db(self):
        try:
            self.conn = sqlite3.connect(db_file)
            self.conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as e:
            raise e

    def _create_table(self):
        create_sql = '''
        CREATE TABLE IF NOT EXISTS metric (
            step_type TEXT NOT NULL,
            version TEXT NOT NULL,
            global_epoch INTEGER NOT NULL,
            current_epoch INTEGER NOT NULL,
            total_epoch INTEGER NOT NULL,
            loss REAL,
            bleu REAL,
            PRIMARY KEY (step_type, global_epoch)
        )
        '''
        try:
            self.conn.execute(create_sql)
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def _init_last_global(self):
        for step_type in self._epoch_states.keys():
            sql = "SELECT MAX(global_epoch) FROM metric WHERE step_type = ?"
            cursor = self.conn.execute(sql, (step_type,))
            max_val = cursor.fetchone()[0]
            if max_val is not None:
                self._epoch_states[step_type]["last_global"] = max_val

    def insert_metric(self, step_type, current_epoch, total_epoch, loss, bleu=None):
        if step_type not in ["test", "train", "eval"]:
            raise ValueError(f"step_type must be test|train|eval")

        state = self._epoch_states[step_type]
        if step_type == "eval":
            global_epoch = self._epoch_states["train"]["last_global"]
        elif step_type == "train":
            global_epoch = state["last_global"] + 1
        else:
            global_epoch = state["last_global"] + 1

        version = get_version_str()
        insert_sql = '''
        INSERT INTO metric
        (step_type, version, global_epoch, current_epoch, total_epoch, loss, bleu)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        try:
            self.conn.execute(insert_sql, (
                step_type, version, global_epoch,
                current_epoch, total_epoch, loss, bleu
            ))
            self.conn.commit()
            if step_type in ["train", "test"]:
                state["last_global"] = global_epoch
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def query_metrics(self, step_type):
        if step_type not in ["test", "train", "eval"]:
            raise ValueError(f"step_type must be test|train|eval")
        query_sql = '''
        SELECT version, global_epoch, current_epoch, total_epoch, loss, bleu
        FROM metric
        WHERE step_type = ?
        ORDER BY global_epoch ASC
        '''
        try:
            cursor = self.conn.execute(query_sql, (step_type,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            return []

    def delete_metric(self, step_type):
        if step_type not in ["test", "train", "eval"]:
            raise ValueError(f"step_type must be test|train|eval")
        delete_sql = '''
        DELETE FROM metric WHERE step_type = ?
        '''
        try:
            self.conn.execute(delete_sql, (step_type,))
            self.conn.commit()
            self._epoch_states[step_type]["last_global"] = 0
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            return False

    def close(self):
        if self.conn:
            self.conn.close()

_global_db_instance = None
def get_metric_db():
    global _global_db_instance
    if _global_db_instance is None:
        _global_db_instance = MetricDB()
    return _global_db_instance


