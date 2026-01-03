import os, pathlib, sqlite3
from datetime import datetime
from .version import get_version_str


proj_root = pathlib.Path(__file__).parent.parent
db_dirt = os.path.join(proj_root, "database")
db_file = os.path.join(db_dirt, "metric.db")
os.makedirs(db_dirt, exist_ok=True)



