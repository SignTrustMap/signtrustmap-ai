import os
import sys
import logging
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logger():
    logger.remove()
    
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    json_logging = os.environ.get("LOG_TO_JSON", "FALSE").upper() == "TRUE"
    
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(
        sys.stderr,
        level=log_level,
        format=console_format,
        colorize=True
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "app.log")
    
    if json_logging:
        logger.add(
            log_file_path,
            level="DEBUG",
            serialize=True,
            rotation="50 MB",
            compression="zip",
            retention="10 days"
        )
    else:
        file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        logger.add(
            log_file_path,
            level="DEBUG",
            format=file_format,
            rotation="50 MB",
            compression="zip",
            retention="10 days"
        )
        
    # Redirect standard logging to Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Intercept FastAPI, Uvicorn, and other library loggers
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
