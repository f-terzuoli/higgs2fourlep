import logging, sys
from ..higgs2fourlep import checkRequirements

logging.basicConfig(filename='reqlog.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
screen_handler = logging.StreamHandler(stream=sys.stdout)
logging.getLogger().addHandler(screen_handler)
checkRequirements()
