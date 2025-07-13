# logging_config.py
import logging, sys
from datetime import datetime
from pathlib import Path
def setup_logging(name=None, level=logging.INFO, logfile="log_debug"):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s [%(pathname)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

Path('logs').mkdir(exist_ok=True)
logfile = f"logs/debug_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG = setup_logging(
    name="debug",
    level=logging.DEBUG,
    logfile=logfile
)
