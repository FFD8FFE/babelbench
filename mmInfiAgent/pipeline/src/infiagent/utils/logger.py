
# Custom Logger Adapter to include X-Tt-Logid
import logging
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

log_id_var: ContextVar[str] = ContextVar("log_id", default="")


class ContextualLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, context):
        super().__init__(logger, {})
        self.context = context

    def process(self, msg, kwargs):
        return '[%s] %s' % (self.context, msg), kwargs


def init_logging():
    """
    Initialize logging configuration.
    """
    # Basic logging configuration with your specified settings
    # config_default()
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=r'[%(levelname)s %(asctime)s %(filename)s:[line:%(lineno)d]] %(message)s',
    )
    return

def get_logger() -> logging.LoggerAdapter:
    """
    Retrieve a logger instance configured with X-Tt-Logid.
    """
    logger = logging.getLogger("infiagent_logger")
    logger.setLevel(logging.INFO)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     datefmt=r'%Y/%m/%d %H:%M:%S',
    #     format=r'[%(levelname)s %(asctime)s %(filename)s:[line:%(lineno)d]] %(message)s',
    #     # filename='test.log',
    #     # filemode='a'
    # )

    # 清除现有的处理器
    logger.handlers = []

    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(filename)s:[line:%(lineno)d]] %(message)s')

    # terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file handler
    log_file_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = TimedRotatingFileHandler(log_file_name, when='midnight', backupCount=7)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return ContextualLoggerAdapter(logger, {})

