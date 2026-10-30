import logging
from datetime import datetime
from pathlib import Path

COMMON_LOG_PATH = Path(
    "/Users/dixitpatel/Documents/Personal-Dev-Workspace/Meta_hackathon/workspace/logs/alphagenome_pipeline_debug.log"
)

logger = logging.getLogger("alphagenome-sidecar")


def log_debug(phase: str, message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    formatted = f"[{now}] [ALPHAGENOME-SIDECAR] [{phase}] {message}"
    logger.info(f"[{phase}] {message}")
    print(formatted, flush=True)
    try:
        COMMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(COMMON_LOG_PATH, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
