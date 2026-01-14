import os, pathlib


class Paths():
    proj_root = pathlib.Path(__file__).parent.parent.resolve()

    conf_dir  = proj_root/"config"
    log_dir   = proj_root/"logs"
    db_dir    = proj_root/"database"
    data_dir  = proj_root/"data"

    ver_file  = proj_root/"CHANGELOG.md"
    conf_file = proj_root/"config.toml"
    db_file   = db_dir/"metric.db"

    def __init__(self):
        # print()
        os.makedirs(self.conf_dir, exist_ok=True)
        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.db_dir,   exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

_global_paths_instance = None
def get_paths():
    global _global_paths_instance
    if _global_paths_instance is None:
        _global_paths_instance = Paths()
    return _global_paths_instance


