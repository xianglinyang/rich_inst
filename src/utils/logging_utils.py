import logging
import os
import sys
import time

def setup_logging(
    task_name, # train, eval, etc.
    log_level=logging.INFO,
    log_dir="logs",
    run_id=None,
):

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(log_dir, task_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path with timestamp
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{time_stamp}_{run_id}.log") if run_id else os.path.join(log_dir, f"{time_stamp}.log")

    # Setup logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file