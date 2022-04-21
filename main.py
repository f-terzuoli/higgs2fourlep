import higgs2fourlep
import logging, sys
from datetime import datetime

start=datetime.now()

logging.basicConfig(filename='all.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
screen_handler = logging.StreamHandler(stream=sys.stdout)
logging.getLogger().addHandler(screen_handler)
higgs2fourlep.checkRequirements()
AnaCfg = higgs2fourlep.loadSettings("conf.yaml")
AnaVars= higgs2fourlep.HiggsProcessor(AnaCfg)
AnaVars.regions = ['4l', '4e', '4mu', '2e2mu', '2mu2e', '4l0j', '4l1j', '4l2j']
AnaVars.regionlabel = ['4l', '4e', '4\mu', '2e2\mu', '2\mu2e', '4l+0j', '4l+1j', '4l+2j']
AnaVars.filterRegion = ["kTRUE", "goodlep_sumtypes == 44", "goodlep_sumtypes == 52", "goodlep_sumtypes == 48 && goodlep_type[Z_idx[0][0]] == 11", "goodlep_sumtypes == 48 && goodlep_type[Z_idx[0][0]] == 13", "goodjet_n == 0", "goodjet_n == 1", "goodjet_n == 2"]
AnaVars.particles = ['H_boson', 'Z_boson', 'Lepton', 'Jets', 'Topology']
AnaVars.initializeData()
AnaVars.skimRecoHiggs()
AnaVars.recoCut()
AnaVars.tightCut()
AnaVars.defineHistos()
#AnaVars.prepareForML()
AnaVars.fitMass()
AnaVars.runNodes()
_ = AnaVars.plotHistos(True)

AnaVars.printStats(AnaCfg.OutputYelds)
AnaCfg_bdt = higgs2fourlep.loadSettings("conf_bdt.yaml")
boosted_tree = higgs2fourlep.traintest(AnaCfg_bdt)
higgs2fourlep.apply_bdt(AnaCfg_bdt, boosted_tree)

logging.info("Analysis and plot run in: {}".format(datetime.now()-start))

