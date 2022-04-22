import os
current_dir = os.path.dirname(os.path.realpath(__file__)))
working_dir = os.path.join(current_dir , "..")

import sys
sys.path.append(working_dir)

import logging, sys
from higgs2fourlep import checkRequirements

logging.basicConfig(filename='reqlog.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
screen_handler = logging.StreamHandler(stream=sys.stdout)
logging.getLogger().addHandler(screen_handler)
checkRequirements()
