import logging
import os
# --- 日志配置 ---
def set_logger(logger_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)

    log_path = os.path.join("../logs/" + logger_name)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)
    # 减少 transformers 的日志噪音
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logger
