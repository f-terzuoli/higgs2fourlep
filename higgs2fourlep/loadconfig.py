#loadconfig.py
"""
This module manages the storage and the checks for loading the correct settings which will be deployed in the Higgs analysis.
"""

import yaml
import sys
import os, errno
import logging
import munch
import urllib.request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

def read_yaml(config_file):
    """
    Return the the parsed .yaml configuration file as a Munch dictionary where items can be accessed via "." dot syntax.
    
    :param config_file: the .yaml file path to be parsed.
    :type config_file: str
    :return: Parsed configuration file as Munch dictionary.
    :rtype: Munch(dict)
    
    """
    try:
        with open(config_file) as file:
            logger.info("Loading analysis settings from {} ...".format(file.name))
            conf = yaml.load(file, Loader=yaml.FullLoader)
            logger.info("Current YAML file contents: \n{}".format(yaml.dump(conf)))
            return munch.munchify(conf)
    except IOError as error:
        logger.error(error)
        logger.fatal("Exiting program due to {}".format(error))
        logging.shutdown()
        sys.exit(1)
    
class AnalysisCfg(object):
    """
    Class for loading the parsed .yaml file and cheking if the settings have acceptable values.
    
    :param loaded_conf: the Munch dictionary containing the settings.
    :type loaded_conf: Munch(dict)
    """
    def __init__(self, loaded_conf):
        """
        Contructor method.
        """
        self.InputAnalysisJSONFile    = loaded_conf.InputAnalysisJSONFile
        self.InputOpenDataURL         = loaded_conf.InputOpenDataURL
        self.OutputPlotsStructure     = loaded_conf.OutputPlotsStructure
        self.OutputYelds              = loaded_conf.OutputYelds
        self.OutputHistosPath         = loaded_conf.OutputHistosPath
        self.SaveSkimmedNTuples       = loaded_conf.SaveSkimmedNTuples
        self.PrepareMLTraining        = loaded_conf.PrepareMLTraining
        self.SkimmedNtuplesPath       = loaded_conf.SkimmedNtuplesPath
        self.NThreads                 = loaded_conf.NThreads
        self.Luminosity               = loaded_conf.Luminosity
        
    @property
    def InputAnalysisJSONFile(self):
        """
        Get or set the path where the path of the JSON file containg all the .root file to be analysed. If the file is not found, it will revert back to the previous value, if already set, otherwise the program will shutdown.
        
        :type: str
        """
        return self._InputAnalysisJSONFile
        
    @InputAnalysisJSONFile.setter
    def InputAnalysisJSONFile(self, in_path):
        try:
            if not os.path.isfile(in_path):
                raise FileNotFoundError("Cannot find a JSON input file at {}".format(in_path))
            else:
                logger.info("Found JSON input file at {}".format(in_path))
                self._InputAnalysisJSONFile = in_path
        except FileNotFoundError as error:
            try:
                dummy = self.InputAnalysisJSONFile
                logger.warning("Invalid new JSON input path/file: {}. Reverting to: {}".format(in_path, self.InputAnalysisJSONFile))
            except AttributeError:
                logger.fatal("Exiting program due to {}".format(error))
                logging.shutdown()
                sys.exit(1)
                
    @property
    def InputOpenDataURL(self):
        """
        Get or set the URL where ATLAS OpenData are located. If an HTTP error occurs, it will be reverted back to the previous value, if already set, otherwise the program will shutdown.
        
        :type: str
        """
        return self._InputOpenDataURL
        
    @InputOpenDataURL.setter
    def InputOpenDataURL(self, in_url):
        try:
            urllib.request.urlopen(in_url)
            logger.info("HTTP Status OK for {}".format(in_url))
            self._InputOpenDataURL = in_url
        except (URLError, HTTPError) as error:
            try:
                dummy = self.InputOpenDataURL
                logger.warning("Invalid new OpendData URL: {}. Reverting to: {}".format(in_url, self.InputOpenDataURL))
            except AttributeError:
                logger.fatal("Exiting program due to {}".format(error))
                logging.shutdown()
                sys.exit(1)
            
    @property
    def OutputPlotsStructure(self):
        """
        Get or set the path where the path of the JSON file containg all the structure of the output histograms for all the variables of interest. If the file is not found, it will revert back to the previous value, if already set, otherwise the program will shutdown.
        
        :type: str
        """
        return self._OutputPlotsStructure
    
    @OutputPlotsStructure.setter
    def OutputPlotsStructure(self, out_path):
        try:
            if not os.path.isfile(out_path):
                raise FileNotFoundError("Cannot find a JSON file for output plots structure at {}".format(out_path))
            else:
                logger.info("Found JSON output plots file at {}".format(out_path))
                self._OutputPlotsStructure = out_path
        except FileNotFoundError as error:
            try:
                dummy = self.OutputPlotsStructure
                logger.warning("Invalid new JSON output plots structure path/file: {}. Reverting to: {}".format(out_path, self.OutputPlotsStructure))
            except AttributeError:
                logger.fatal("Exiting program due to {}".format(error))
                logging.shutdown()
                sys.exit(1)
        
    @property
    def OutputYelds(self):
        """
        Get or set the output folder where the yelds will be stored. If the path setting is empty or of wrong type, the default ./Yelds path will be imposed.
        
        :type: str
        """
        return self._OutputYelds
        
    @OutputYelds.setter
    def OutputYelds(self, yeld_path):
        try:
            os.makedirs(yeld_path)
            self._OutputYelds = yeld_path
            logger.info("Setting \"{}\" as yelds output directory".format(yeld_path))
            logger.info("Created \"{}\" as yelds output directory".format(yeld_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                logger.error(e)
            else:
                logger.warning("Yelds output directory already exists: files might be overwritten")
                self._OutputYelds = yeld_path
        except TypeError as error:
            logger.error("Yelds output directory missing or of wrong type! Setting to default \"./Yelds\"")
            self._OutputYelds = "./Yelds"
            
    @property
    def OutputHistosPath(self):
        """Get or set the output folder where the histograms will be stored. If the path setting is empty or of wrong type, the default ./Histos path will be imposed.
        
        :type: str
        """
        return self._OutputHistosPath
        
    @OutputHistosPath.setter
    def OutputHistosPath(self, histos_path):
        try:
            os.makedirs(histos_path)
            self._OutputHistosPath = histos_path
            logger.info("Setting \"{}\" as histos output directory".format(histos_path))
            logger.info("Created \"{}\" as histos output directory".format(histos_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                logger.error(e)
            else:
                logger.warning("Histos output directory already exists: files might be overwritten")
                self._OutputHistosPath = histos_path
        except TypeError as error:
            logger.warning("Histos output directory missing or of wrong type! Setting to default \"./Histos\"")
            self._OutputHistosPath = "./Histos"
        
    @property
    def SaveSkimmedNTuples(self):
        """Get or set the flag for saving or not the .root Ntuples after bosons reconstruction. If the flag setting is empty or of wrong type, the default 'False' will be imposed.
        
        :type: bool
        """
        return self._SaveSkimmedNTuples
        
    @SaveSkimmedNTuples.setter
    def SaveSkimmedNTuples(self, saveNTuples):
        if not type(saveNTuples) == bool:
            logger.error("SaveSkimmedNTuples flag not set properly! Setting to default \"false\"")
            self._SaveSkimmedNTuples = False
        else:
            self._SaveSkimmedNTuples = saveNTuples
            logger.info("SaveSkimmedNTuples flag correctly set")
            
    @property
    def PrepareMLTraining(self):
        """Get or set the flag for saving or not the .root Ntuples for ML training/testing. If the flag setting is empty or of wrong type, the default 'False' will be imposed.
        
        :type: bool
        """
        return self._PrepareMLTraining
        
    @PrepareMLTraining.setter
    def PrepareMLTraining(self, prepareMl):
        if not type(prepareMl) == bool:
            logger.error("PrepareMLTraining flag not set properly! Setting to default \"false\"")
            self._PrepareMLTraining = False
        else:
            self._PrepareMLTraining = prepareMl
            logger.info("PrepareMLTraining flag correctly set")
            
    @property
    def SkimmedNtuplesPath(self):
        """Get or set the output folder where the skimmed .root Ntuples will be stored. If the path setting is empty or of wrong type, the default ./Skimmed_Ntuples path will be imposed.
        
        :type: str
        """
        return self._SkimmedNtuplesPath
            
    @SkimmedNtuplesPath.setter
    def SkimmedNtuplesPath(self, ntuples_path):
        if self.SaveSkimmedNTuples == True or self.PrepareMLTraining == True:
            try:
                os.makedirs(ntuples_path)
                self._SkimmedNtuplesPath = ntuples_path
                logger.info("Setting \"{}\" as NTuples output directory".format(ntuples_path))
                logger.info("Created \"{}\" as NTuples output directory".format(ntuples_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    logger.error(e)
                else:
                    logger.warning("NTuples output directory already exists: files might be overwritten")
                    self._SkimmedNtuplesPath = ntuples_path
            except TypeError as error:
                logger.warning("NTuples output directory missing or of wrong type! Setting to default \"./Skimmed_NTuples\"")
                self._SkimmedNtuplesPath = "./Skimmed_NTuples"
        else:
            self._SkimmedNtuplesPath = None
            
    @property
    def NThreads(self):
        """
        Get or set the number of threads available for ROOT implicit multi-threading. If incorrectly set it will be reverted to default 0, thus disabling implicit multi-threading.
        
        :type: int
        """
        return self._NThreads
        
    @NThreads.setter
    def NThreads(self, nthread):
        if not (type(nthread) == int and nthread >= 0):
            logger.error("NThreads not set properly! Setting to default \"0\". Multithread not activated.")
            self._NThreads = 0
        else:
            self._NThreads = nthread
            logger.info("NThreads correctly set.")
            
    @property
    def Luminosity(self):
        """
        Get or set the integrated luminosity of all the dataset. If incorrectly set, it will revert back to the previous value, if previously set, otherwise the program will shutdown.
        
        :type: double
        """
        return self._Luminosity
        
    @Luminosity.setter
    def Luminosity(self, lumi):
        try:
            if not (type(lumi) == float and lumi >= 0):
                raise ValueError("Invalid value.")
            else:
                self._Luminosity = lumi
                logger.info("Luminosity correctly set.")
        except ValueError as error:
            try:
                dummy = self.Luminosity
                logger.warning("Invalid new luminosity value: {}. Reverting to: {}".format(lumi, dummy))
            except AttributeError:
                logger.fatal("Exiting program due to {}".format(error))
                logging.shutdown()
                sys.exit(1)
            

def loadSettings(config_file):
    """
    Parse the .yaml file and then load the settings in a :py:class:`AnalysisCfg` object.
    
    :param config_file: the .yaml file path to be parsed.
    :type config_file: str
    :return: The loaded configuration settings.
    :rtype: AnalysisCfg
    """
    cfg = read_yaml(config_file)
    settings = AnalysisCfg(cfg)
    return settings

            
if __name__ == "__main__":
    logging.basicConfig(filename='testlog.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(screen_handler)
    #cfg = read_yaml("conf.yaml")
    #AnaCfg = AnalysisCfg(cfg)
    AnaCfg = load_settings("conf.yaml")
    
    
