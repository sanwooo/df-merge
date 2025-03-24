import logging

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

formatter = logging.Formatter('%(asctime)s:%(module)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S')

# Handler that prints logs above INFO level (included) in the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_info_file(file_path: str):
    # Handler that prints logs above DEBUG level (included) in the file
    file_debug_handler = logging.FileHandler(file_path)
    file_debug_handler.setLevel(logging.DEBUG)
    file_debug_handler.setFormatter(formatter)
    logger.addHandler(file_debug_handler)