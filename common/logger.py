import os, logging, logging.handlers
from .paths import get_paths
from .version import get_version_str


class VersionFilter(logging.Filter):
    def filter(self, record):
        full_version = get_version_str()
        record.version = ".".join(full_version.split(".")[:3])
        return True

def init_logger(log_id):
    p = get_paths()
    log_dir = p.log_dir

    log_file = os.path.join(log_dir, log_id)
    logger = logging.getLogger(log_id)
    if logger.hasHandlers(): return logger

    logger.addFilter(VersionFilter())  # Inject dynamic version
    logger.setLevel(logging.DEBUG)     # Global log level (capture all levels)
    logger.propagate = False           # Avoid duplicate output

    # Log format: Time | Version | Module:LineNo | Level | Message (traceable)
    formatter = logging.Formatter(
        "%(asctime)s | %(version)s | %(module)s:%(lineno)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console Handler: Output INFO+ (for development, filter redundant debug logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler: Daily rotation, output DEBUG+ (persistent storage, keep 7 days)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="D",          # Rotate by day
        interval=1,        # x file per day
        backupCount=3,     # Keep x days of logs
        encoding="utf-8",  # Avoid Chinese garbled
        )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.suffix="%Y-%m-%d.log"
    file_handler.namer = lambda name: name.replace(".", "_", 1)
    logger.addHandler(file_handler)

    return logger


