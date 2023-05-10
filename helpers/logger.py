import logging

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
