#checkRequirements.py
import sys
import os, errno
import logging

logger = logging.getLogger(__name__)

def checkRequirements():
    """
    Check if required modules are correctly installed and available.
    """
    try:
        import ROOT
        logger.info("'PyROOT' installed and correcly set")
    except ModuleNotFoundError:
        logger.fatal("'PyROOT' not correcly configured. Check if ROOT is installed and PyROOT libraries env paths correctly set.")
    
    try:
        if not ROOT.gROOT.GetVersionInt()>62400:
            raise Exception
        else:
            logger.info("ROOT version is {} >= 6.24/00 minimum required.".format(ROOT.gROOT.GetVersion()))
    except:
        logger.fatal("Upgrade your ROOT version to at least 6.24/00.")
    
    try:
        import sklearn
        logger.info("'Sklearn' installed")
    except ModuleNotFoundError:
        logger.fatal("'Sklearn' not found. Install via 'pip3 install sklearn'.")
    
    try:
        import xgboost
        logger.info("'XGBoost' installed")
    except ModuleNotFoundError:
        logger.fatal("'XGBoost' not found. Install via 'pip3 install xgboost'. Requires OpenMP for multicore.")
    
    try:
        import yaml
        logger.info("'PyYaml' installed")
    except ModuleNotFoundError:
        logger.fatal("'PyYaml' not found. Install via 'pip3 install pyyaml'.")
    
    try:
        import munch
        logger.info("'Munch' installed")
    except ModuleNotFoundError:
        logger.fatal("'Munch' not found. Install via 'pip3 install munch'.")
    
    return

if __name__ == "__main__":
    logging.basicConfig(filename='reqlog.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(screen_handler)
    checkRequirements()
