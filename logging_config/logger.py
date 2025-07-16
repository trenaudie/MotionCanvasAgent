import logging
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

class NYCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("America/New_York"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

    def format(self, record):
        try:
            cwd = os.getcwd()
            record.pathname = os.path.relpath(record.pathname, start=cwd)
        except Exception:
            pass  # fallback to absolute if relpath fails
        return super().format(record)

def setup_logging(
    name="MotionCanvasAgent",
    level=logging.INFO,
    logfile=True,
    rewrite_logfile=True
):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers on multiple imports

    logger.setLevel(level)

    formatter = NYCFormatter(
        fmt="%(asctime)s %(levelname)-8s : %(pathname)s(%(lineno)d) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # StreamHandler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logfile:
        log_mode = "w" if rewrite_logfile else "a"
        file_handler = logging.FileHandler(f"{name}.log", mode=log_mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Export singleton logger
LOG = setup_logging(level=logging.DEBUG)
