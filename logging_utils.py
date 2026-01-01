import logging

def get_logger(name=None):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    if not logger.handlers:
        
        logger.setLevel(logging.INFO)
    return logger