
from mmengine import MMLogger

logger = MMLogger.get_instance('mmengine')


def log(*args):
    args = [str(args) for i in args]
    logger.info(''.join(args))
