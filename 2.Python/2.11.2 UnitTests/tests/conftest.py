import sys

from loguru import logger # логирование

# Здесь определяется новый формат логирования с использованием библиотеки loguru
NEW_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    "- <level>{message}</level>"
)

def configure_stdout_log():
    '''
    функция настраивает логирование в stdout с использованием нового формата
    '''
    logger.remove()
    logger.configure(
        handlers=[dict(sink=sys.stdout, format=NEW_FORMAT, diagnose=False, level="DEBUG")]
    )


def pytest_configure(config):
    """Функция настраивает логирование для pytest, включая уровень логирования и цвета"""
    logger.remove()
    logger.add(sys.stdout, filter=__name__, format="<level>{message}</level>")
    logger.level("DEBUG", color="<cyan>")
    logger.level("INFO", color="<light-blue>")
    logger.debug("Running pytest pre-configuration")
    logger.critical(f"Current level is {logger._core.min_level}")  # pylint:disable=protected-access
    logger.info(config.inicfg["env"].replace("\n", " ") + "\n")

    configure_stdout_log()


def pytest_collection_modifyitems(items):
    """Функция добавляет метки к тестовым элементам 
    в зависимости от их расположения

    Используется с командой 'pytest -m unit'
    """
    #for item in items:
    #    if "/unit/" in str(item.module):
    #        item.add_marker("unit")
    #    elif "/mock/" in str(item.module):
    #        item.add_marker("mock")
