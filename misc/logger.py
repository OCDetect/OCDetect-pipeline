import logging
import inspect
import pathlib

# Create a logger object, which can be imported from the other files.
# Set the logging settings:

logger = logging.getLogger('OCDetect_logger')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
fh = logging.FileHandler("debug.log")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
sh.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(sh)
logger.addHandler(fh)

debug = logger.debug
warning = logger.warning
error = logger.error
info = logger.info


def logwrapper(log_message):
    """
    Function wraps all original logger functions, adds the name of the calling module.
    :param log_message: The message to be logged. The additional text will be added to it.
    :return:
    """
    frame = inspect.stack()[2]  # gets the module from which this function was called (wont work inside logger.py)
    filename = pathlib.Path(frame[0].f_code.co_filename).name
    return f"[{filename}] {log_message}"


logger.debug = lambda x: debug(logwrapper(x))
logger.warning = lambda x: warning(logwrapper(x))
logger.error = lambda x: error(logwrapper(x))
logger.info = lambda x: info(logwrapper(x))

