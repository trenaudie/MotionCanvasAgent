# logging_config.py
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo  # Requires Python 3.9+
import logging
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

class NYCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("America/New_York"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

    def format(self, record):
        # Replace pathname with relative path
        try:
            cwd = os.getcwd()
            record.pathname = os.path.relpath(record.pathname, start=cwd)
        except Exception:
            pass  # Fallback to absolute if relpath fails
        return super().format(record)

def setup_logging(
    name=None,
    level=logging.INFO,
    logfile: bool = True,
    debug_project: bool = False,
    rewrite_logfile: bool = True  # NEW argument
):
    if name is None:
        name = "logger" if not debug_project else "logger_debug"
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    fmt = NYCFormatter(
        "%(asctime)s %(levelname)-8s : %(pathname)s(%(lineno)d) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        # Choose mode: "w" = rewrite, "a" = append
        mode = "w" if rewrite_logfile else "a"
        fh = logging.FileHandler(f"{name}.log", mode=mode, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
LOG = setup_logging(
    level=logging.DEBUG,
    debug_project=True
)
