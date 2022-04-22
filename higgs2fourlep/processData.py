#processData.py
"""
This module manages the workflow of the analysis, accessing the files to be analysed, filtering events and reconstructing the Higgs boson from its decay products, compute the yelds per signal category, plot the distribution of interest regarding the decay and performing a fit to measure the Higgs mass.
"""

import logging
import json
import sys
#sys.argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch()
import os
import errno

logger = logging.getLogger(__name__)
higgs_lib = os.path.join(os.path.dirname(__file__),"higgs4l.h")
class HiggsProcessor(object):
    """
    Class managing the Higgs Analysis flow. It relies on the "higgs.h" header file containing all the definition of the C++ functions for filtering and reconstructing the decay particles.
    
    :param conf: The loaded configuration settings.
    :type conf: AnalysisCfg
    
    :ivar cfg: initial value: :param conf:
    :vartype cfg: AnalysisCfg
    :ivar initialized: flag indicating if the HiggsProcessor has loaded the necessary setting for starting the analysis
    :vartype initialized: bool
    :ivar reconstructedHiggs: flag indicating if the the Higgs boson has already benn reconstructed and operation are available on its 4-vector
    :vartype reconstructedHiggs: bool
    :ivar lumi: the integrated luminosity of the dataset
    :vartype lumi: double
    :ivar path: the URL where ATLAS Open Data is located
    :vartype path: str
    :ivar files: dictionary cointaining the input files, structured based on their location folder, physical process and containing information as cross-section of the process and sum-of-weights of MC simulations
    :vartype files: dict
    :ivar processes: list containing the processes simulated via MC and the acquired data as 'Data_<A,B,C,D>'
    :vartype processes: list of str
    :ivar df: dictionary containing all the dataframe nodes for very input file to be analysed
    :vartype df: dict
    :ivar df_snapshot: dictionary containing all the dataframe nodes for very input file to saved in .root file as ROOT::Tree
    :vartype df_snapshot: dict
    :ivar df_training: dictionary containing all the dataframe nodes destined to be the training sample for the XGBoost
    :vartype df_training: dict
    :ivar df_testing: dictionary containing all the dataframe nodes destined to be the testing sample for the XGBoost
    :vartype df_testing: dict
    :ivar df_1jet: dictionary containing all the dataframe nodes with events having at least 1 jet
    :vartype df_1jet: dict
    :ivar df_2jet: dictionary containing all the dataframe nodes with events having at least 2 jets
    :vartype df_2jet: dict
    :ivar histos: nested dictionary containing the histograms of all variables of interested for every file analysed and for every region defined
    :vartype histos: dict
    :ivar xsecs: dictionary containing the cross-section for every MC analysed
    :vartype histos: dict
    :ivar sumws: dictionary containing the sum-of-weights for every MC analysed
    :vartype sumws: dict
    :ivar samples: list containing all the files analysed
    :vartype samples: list of str
    :ivar count: nested dictionary containing the yelds counter for every region defined, every category of data defined (different Higgs production modes, data, zz_bkg, other_bkg) and at different steps of the analysis
    :vartype count: dict
    :ivar count_w: the same as :ivar count: but MC events are weighted taking in consideration the integrated luminosity, the process cross-section and the sum-of-weights of the produced MC
    :vartype count_w: dict
    :ivar regions: list containing the defined regions' names
    :vartype regions: list of str
    :ivar regionlabel: list containing the defined regions' labels to appear in the output histograms
    :vartype regionlabel: list of str
    :ivar filterRegion: list of string containing the criteria defining the regions and to be passed to the :meth: `ROOT::RDataframe::Filter()` method
    :vartype filterRegion: list of str
    :ivar particles: list of the particles partecipating in the decay. It will be used to organize the OutputHisto folder where the histograms will be saved.
    :vartype particles: list of str
    :ivar nodes: list of RDataframe Nodes, to be fed to ROOT::RDF::RunGraph, in order to parellize the analysis on the different files.
    :vartype nodes: list of :class:`ROOT::RDF::RNode`
    
    """
    def __init__(self, conf):
        self.cfg = conf
        ROOT.gInterpreter.Declare(f" #include \"{higgs_lib}\" ")
        self.initialized = False
        self.reconstructedHiggs = False
        self.lumi = 0.
        self.path = ""
        self.files = {}
        self.processes = []
        self.df = {}
        self.df_snapshot = {}
        self.df_training = {}
        self.df_testing = {}
        self.df_1jet = {}
        self.df_2jet = {}
        self.histos = {}
        self.xsecs = {}
        self.sumws = {}
        self.samples = []
        self.count = {}
        self.count_w = {}
        self.regions = []
        self.regionlabel = []
        self.filterRegion = []
        self.particles = []
        self.nodes = []
        

    def initializeData(self):
        """
        Initialize instance attributes with configuration loaded from :class:`AnalysisCfg` object.
        For every regions defined an initial counter is set for a yeld of the "untouched" data sets.
        Once all is correctly loaded the flag :ivar initialized: is set to True.
        
        """
        #ROOT.ROOT.EnableImplicitMT(self.cfg.NThreads)
        ROOT.TH1.SetDefaultSumw2()
        
        for r in self.regions:
            self.count[r] = {}
            self.count_w[r] = {}
        
        self.lumi = self.cfg.Luminosity
        self.path = self.cfg.InputOpenDataURL
        with open(self.cfg.InputAnalysisJSONFile) as f:
            self.files = json.load(f)
        self.processes = self.files.keys()
        for r in self.regions:
            self.count[r]['initial'] = {}
            self.count_w[r]['initial'] = {}
        for p in self.processes:
            for d in self.files[p]:
                # Construct the dataframes
                folder = d[0] # Folder name
                nlep = d[1] # Lepton multiplicity
                sample = d[2] # Sample name
                self.xsecs[sample] = d[3] # Cross-section
                self.sumws[sample] = d[4] # Sum of weights
                self.samples.append(sample)
                self.df[sample] = ROOT.RDataFrame("mini", "{}/{}/{}/{}.{}.root".format(self.path, nlep, folder, sample, nlep))
                if 'data' in sample:
                    self.df[sample] = self.df[sample].Define("weight", "1.0")
                elif 'zz' in p:
                    self.df[sample] = self.df[sample].Define("weight", "1.4 * scaleFactor_ELE * scaleFactor_MUON * scaleFactor_LepTRIGGER * scaleFactor_PILEUP * mcWeight * {} / {} * {}".format(self.xsecs[sample], self.sumws[sample], self.lumi))
                else:
                    self.df[sample] = self.df[sample].Define("weight", "scaleFactor_ELE * scaleFactor_MUON * scaleFactor_LepTRIGGER * scaleFactor_PILEUP * mcWeight * {} / {} * {}".format(self.xsecs[sample], self.sumws[sample], self.lumi))
                for r in self.regions:
                    self.count[r]['initial'][sample] = self.df[sample].Count()
                    self.count_w[r]['initial'][sample] =  self.df[sample].Sum("weight")
        logger.info("HiggsProcessor initialized.")
        self.initialized = True
        return
    
    def saveNTuples(self, ml = False):
        """
        Prepare the snapshot nodes for saving the NTuples in .root files. The action is set to 'Lazy' so that the snapshot does not trigger the loop over the samples.
        
        :param ml: Flag for XGBoost training/test samples production. If set to 'True' every physical process sample is split in two different samples of equal population: one for training, the other for testing. If set to 'False' only one file will be produced per every physical process.
        :type ml: bool
        :raises: RuntimeError if Higgs has not been reconstructed before invoking this method and the program is shutdown.
        
        """
        try:
            if self.reconstructedHiggs:
                pass
            else:
                raise RuntimeError("Higgs has not been reconstructed yet. Please invoke skimRecoHiggs() before invoking saveNTuples(ml).")
        except RuntimeError as error:
            logger.fatal("Exiting program due to {}".format(error))
            logging.shutdown()
            sys.exit(1)
        
        colNames = [ "Z_vector", "H_vector", "Z1_CoM_H", "Z2_CoM_H", "goodlep_CoM_H", "goodlep1_CoM_Z1", "goodlep2_CoM_Z2", "m4l", "pt4l", "eta4l", "phi4l", "mZ1", "ptZ1", "etaZ1", "phiZ1", "mZ2", "ptZ2", "etaZ2", "phiZ2", "costhetastar", "costheta1_H", "costheta2_H", "costheta1_Z1", "costheta2_Z2", "phi", "phi1", "deltaphiZ1Z2", "deltaphiZ1Z2com", "goodlep_idx", "goodlep_sumtypes", "goodlep_pt", "goodlep_eta", "goodlep_phi", "goodlep_E", "goodlep_z0", "goodlep_charge", "goodlep_type", "goodlep_ptcone30", "goodlep_etcone20", "goodlep_d0" , "goodlep_sigd0", "good_jet", "goodjet_n", "goodjet_idx", "goodjet_pt", "goodjet_eta", "goodjet_phi", "goodjet_E", "weight", "runNumber", "eventNumber", "channelNumber", "mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PHOTON", "scaleFactor_TAU", "scaleFactor_BTAG", "scaleFactor_LepTRIGGER", "scaleFactor_PhotonTRIGGER", "lep_pt_syst", "met_et_syst", "jet_pt_syst", "photon_pt_syst", "tau_pt_syst", "XSection", "SumWeights" ]
        
        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        snapshotOptions.fLazy = True
        colList = ROOT.std.vector("string")()
        for colName in colNames:
            colList.push_back(colName)
        for s in self.samples:
            if ('data' in s or (not ml)):
                self.df_snapshot[s] = self.df[s].Snapshot("RecoTree", "{}/Reco_{}.root".format(self.cfg.SkimmedNtuplesPath, s), colList, snapshotOptions)
            else:
                self.df_training[s] = self.df[s].Range(1,0,2).Snapshot("RecoTree", "{}/Training_{}.root".format(self.cfg.SkimmedNtuplesPath, s), colList, snapshotOptions)
                self.df_testing[s] = self.df[s].Range(2,0,2).Snapshot("RecoTree", "{}/Testing_{}.root".format(self.cfg.SkimmedNtuplesPath, s), colList, snapshotOptions)
        return

    def skimRecoHiggs(self):
        """
        Method for reconstructing the Higgs boson and the two Z bosons.
        
        First is required a trigger on either electron or muon. Then 'GoodLeptons' are designed so that they satisfy the geometrical acceptance, good isolation and pt>5GeV.
        Then events with 4 'GoodLeptons', with neutral net charge and either 4e, 4#mu or 2e+2#mu, are selected.
        Good jets are defined as those inside the geometrical acceptance and with pt>30GeV.
        Additional selections are imposed on the leptons via GoodElectronsAndMuons function defined in Higgs4l.h, and only those with high transverse momentum are selected.
        Then the Higgs is reconstructed paring the leptons as follows: two leptons of the same flavour, opposite chaarge, and with invarinat mass closer to the Z mass are defined as the leading pair and reconstructed as the real Z boson. The remaning pair (subleading) is the virtual Z boson.
        
        Through the GenVect class of ROOT various observables are extracted from the reconstructed 4-vectors.
        An additional counter is then "positioned" at this point of the analysis for every defined region.
        The flag :ivar reconstructedHiggs: is set to True, and if the the :ivar cfg: is set to save the skimmed NTuples, .root files with the events hitherto survived, and with all the reconstructed variables, will be saved in the proper directory.
        
        :raises: RuntimeError if the HiggsProcessor is not initialised before invoking this method and the program is shutdown.
        
        """
        try:
            if self.initialized:
                pass
            else:
                raise RuntimeError("Data not loaded and initialized trough initializeData(configfile). Invoke initializeData(configfile) before invoking skimRecoHiggs(dataframe, configfile).")
        except RuntimeError as error:
            logger.fatal("Exiting program due to {}".format(error))
            logging.shutdown()
            sys.exit(1)
            
        for r in self.regions:
            self.count[r]['lepSel'] = {}
            self.count_w[r]['lepSel'] = {}
            

        for s in self.samples:
        
            self.df[s] = self.df[s].Filter("trigE || trigM")

            self.df[s] = self.df[s].Define("good_lep", "abs(lep_eta) < 2.7 && lep_pt > 5000 && (lep_ptcone30 + 0.4*lep_etcone20 )/ lep_pt < 0.16")\
                         .Define("goodlep_charge", "lep_charge[good_lep]")\
                         .Define("goodlep_type", "lep_type[good_lep]")\
                         .Define("good_e", "goodlep_type == 11")\
                         .Filter("Sum(good_lep) == 4")\
                         .Filter("Sum(lep_charge[good_lep]) == 0")\
                         .Filter("Sum(goodlep_charge[good_e]) == 0")\
                         .Define("goodlep_sumtypes", "Sum(lep_type[good_lep])")\
                         .Filter("goodlep_sumtypes == 44 || goodlep_sumtypes == 52 || goodlep_sumtypes == 48")

                         
            # Count the jets with pt > 30 GeV
            self.df[s] = self.df[s].Define("good_jet", "jet_pt > 30000 && jet_eta < 4.5")\
                         .Define("goodjet_n", "Sum(good_jet)")

            # Apply additional cuts depending on lepton flavour
            self.df[s] = self.df[s].Filter("SelectElectronsAndMuons(lep_type[good_lep], lep_pt[good_lep], lep_eta[good_lep], lep_phi[good_lep], lep_E[good_lep], lep_tracksigd0pvunbiased[good_lep], lep_z0[good_lep])")

            # Create new columns with the kinematics of good leptons
            self.df[s] = self.df[s].Define("goodlep_pt", "lep_pt[good_lep]")\
                         .Define("goodlep_eta", "lep_eta[good_lep]")\
                         .Define("goodlep_phi", "lep_phi[good_lep]")\
                         .Define("goodlep_E", "lep_E[good_lep]")\
                         .Define("goodlep_ptcone30", "lep_ptcone30[good_lep]")\
                         .Define("goodlep_etcone20", "lep_etcone20[good_lep]")\
                         .Define("goodlep_z0", "lep_z0[good_lep]")\
                         .Define("goodlep_d0", "lep_trackd0pvunbiased[good_lep]")\
                         .Define("goodlep_sigd0", "lep_tracksigd0pvunbiased[good_lep]")

            # Select leptons with high transverse momentum
            self.df[s] = self.df[s].Filter("goodlep_pt[0] > 20000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000")
            

            # Create new columns with the kinematics of good jets
            self.df[s] = self.df[s].Define("goodjet_pt", "jet_pt[good_jet]")\
                         .Define("goodjet_eta", "jet_eta[good_jet]")\
                         .Define("goodjet_phi", "jet_phi[good_jet]")\
                         .Define("goodjet_E", "jet_E[good_jet]")\
                         .Define("goodjet_idx", "order_jets(goodjet_pt, goodjet_n)")
                        
            self.df[s] = self.df[s].Define("Z_idx", "reco_zz(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E, goodlep_charge, goodlep_type, goodlep_sumtypes)")\
                         .Define("Z_vector", "compute_z_vectors(Z_idx, goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")\
                         .Define("H_vector", "compute_H_vector(Z_vector)")\
                         .Define("m4l", "H_vector.M() / 1000.")\
                         .Define("pt4l", "H_vector.Pt() / 1000.")\
                         .Define("eta4l", "H_vector.Eta()")\
                         .Define("phi4l", "H_vector.Phi()")\
                         .Define("mZ1", "Z_vector[0].M() / 1000.")\
                         .Define("ptZ1", "Z_vector[0].Pt() / 1000.")\
                         .Define("etaZ1", "Z_vector[0].Eta()")\
                         .Define("phiZ1", "Z_vector[0].Phi()")\
                         .Define("mZ2", "Z_vector[1].M() / 1000.")\
                         .Define("ptZ2", "Z_vector[1].Pt() / 1000.")\
                         .Define("etaZ2", "Z_vector[1].Eta()")\
                         .Define("phiZ2", "Z_vector[1].Phi()")\
                         .Define("goodlep_idx", "get_idxlep(Z_idx)")\
                         .Define("goodlep_CoM_H", "compute_lep_vectors_com_H(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E, H_vector)")\
                         .Define("Z1_CoM_H", "com_vector(Z_vector[0], H_vector)")\
                         .Define("Z2_CoM_H", "com_vector(Z_vector[1], H_vector)")\
                         .Define("goodlep1_CoM_Z1", "com_vector(goodlep_CoM_H[goodlep_idx[0]], Z1_CoM_H)")\
                         .Define("goodlep2_CoM_Z2", "com_vector(goodlep_CoM_H[goodlep_idx[2]], Z2_CoM_H)")\
                         .Define("costhetastar", "TMath::Cos(Z1_CoM_H.Theta())")\
                         .Define("costheta1_H", "compute_costheta12(goodlep_CoM_H[goodlep_idx[0]], Z1_CoM_H)")\
                         .Define("costheta2_H", "compute_costheta12(goodlep_CoM_H[goodlep_idx[2]], Z2_CoM_H)")\
                         .Define("costheta1_Z1", "TMath::Cos(goodlep1_CoM_Z1.Theta())")\
                         .Define("costheta2_Z2", "TMath::Cos(goodlep2_CoM_Z2.Theta())")\
                         .Define("phi","compute_phi(goodlep_CoM_H, goodlep_idx, Z1_CoM_H)")\
                         .Define("phi1","compute_phi1(goodlep_CoM_H, goodlep_idx, Z1_CoM_H)")\
                         .Define("deltaphiZ1Z2","ROOT::Math::VectorUtil::DeltaPhi(Z_vector[0], Z_vector[1])")\
                         .Define("deltaphiZ1Z2com","ROOT::Math::VectorUtil::DeltaPhi(Z1_CoM_H, Z2_CoM_H)")
            #Counter after lepton selection
            for r, filter in zip(self.regions, self.filterRegion):
                region_df = self.df[s].Filter(filter)
                self.count[r]['lepSel'][s] = region_df.Count()
                self.count_w[r]['lepSel'][s] = region_df.Sum("weight")
        #Higgs is reconstructed: save Ntuple (optional)
        self.reconstructedHiggs = True
        if self.cfg.SaveSkimmedNTuples == True:
            self.saveNTuples(False)
        return

    def recoCut(self):
        """
        Method for selecting events using reconstructed variables.
        Additional selection on the reconstructed 4 lepton invarian mass, the Z and Z* mass and the direction separation of the paired leptons.
        An ulterior counter is put at this step for every region.
        
        :raises: RuntimeError if the Higgs boson has not been recontructed before invoking this method and the program is shutdown.
        
        """
        try:
            if self.reconstructedHiggs:
                pass
            else:
                raise RuntimeError("Higgs has not been reconstructed yet. Please invoke skimRecoHiggs() before invoking recoCut().")
        except RuntimeError as error:
            logger.fatal("Exiting program due to {}".format(error))
            logging.shutdown()
            sys.exit(1)
        
        for r in self.regions:
            self.count[r]['RecoCut'] = {}
            self.count_w[r]['RecoCut'] = {}
        #Select on reconstructed masses
        for s in self.samples:
            self.df[s] = self.df[s].Filter("m4l>80. && m4l<170.")\
                         .Filter("mZ1 > 50. && mZ1 < 106.")\
                         .Filter("((mZ2 > (12. + 0.76 * (m4l - 140.)) && m4l > 140.) || (mZ2 > 12 && m4l <= 140.)) && mZ2 < 115.")\
                         .Filter("filter_z_dr(Z_idx, goodlep_eta, goodlep_phi)")
            #Counter after selections on reconstructed masses
            for r, filter in zip(self.regions, self.filterRegion):
                region_df = self.df[s].Filter(filter)
                self.count[r]['RecoCut'][s] = region_df.Count()
                self.count_w[r]['RecoCut'][s] = region_df.Sum("weight")
        return

    def tightCut(self):
        """
        Method for applying tighter cuts on reonstructed variables.
        Additional selection on the reconstructed 4 lepton invarian mass and jet multiplicity.
        An ulterior counter is put at this step for every region.
        
        :raises: RuntimeError if the Higgs boson has not been recontructed before invoking this method and the program is shutdown.
        
        """
        try:
            if self.reconstructedHiggs:
                pass
            else:
                raise RuntimeError("Higgs has not been reconstructed yet. Please invoke skimRecoHiggs() before invoking tightCut().")
        except RuntimeError as error:
            logger.fatal("Exiting program due to {}".format(error))
            logging.shutdown()
            sys.exit(1)
            
        for r in self.regions:
            self.count[r]['TightCut'] = {}
            self.count_w[r]['TightCut'] = {}
        #Tighter selectionon Higgs mass window and jet multiplicity
        for s in self.samples:
            self.df[s] = self.df[s].Filter("m4l>110. && m4l<132.5 && goodjet_n<3.")
            for r, filter in zip(self.regions, self.filterRegion):
                region_df = self.df[s].Filter(filter)
                self.count[r]['TightCut'][s] = region_df.Count()
                self.count_w[r]['TightCut'][s] = region_df.Sum("weight")
        return


    def runNodes(self):
        """
        Method for triggering the loop over the samples using implicit multithreading of ROOT and also parallelize the loop over different samples.
        
        """
        #Append all RDF::RNodes to the list self.nodes and feed it to RDF.RunGraphs
        for s in self.samples:
            for r in self.regions:
                dummycounter = self.count[r]
                counterlaststep = list(dummycounter.keys())[-1]
                self.nodes.append(self.count[r][counterlaststep][s])
                dummyhisto = self.histos[s][r]
                for o in dummyhisto.keys():
                    self.nodes.append(self.histos[s][r][o])
        for s in self.df_snapshot.keys():
            self.nodes.append(self.df_snapshot[s])
        for s in self.df_training.keys():
            self.nodes.append(self.df_training[s])
        for s in self.df_testing.keys():
            self.nodes.append(self.df_testing[s])
        ROOT.RDF.RunGraphs(self.nodes)
        return

    def merge_stat(self, label, counter):
        """
        Method for merging the counters' stats of the same process category (different Higgs production modes, real data, zz_bkg, other_bkg).
        It is invoked by :py:meth:`printStats()` method.
        
        :param label: The category under which the samples are to be merged.
        :type label: str
        :param counter: The dictionary of counters to be merged.
        :type counter: dict
        :return: the sum of the merged counters
        :rtype: double
        
        """
        h = 0
        for i, d in enumerate(self.files[label]):
            t = counter[d[2]].GetValue()
            h += t
        return h
        
    def printStats(self, file_path):
        """
        Method for printing the analysis yelds on file RetainRates_<region>.log .
        
        :param file_path: The path where the yeld file will be stored.
        :type file_path: str
        
        """
        for r in self.regions:
            dummycounter = self.count[r]
            file = f"{file_path}/RetainRates_{r}.log"
            with open(file, "w+") as f:
                f.write("########### RETENTION RATES #######################\n")
                for p in self.processes:
                    mergedstat = {}
                    mergedstat_w = {}
                    error = {}
                    for step in dummycounter.keys():
                        mergedstat[step] = self.merge_stat(p, self.count[r][step])
                        mergedstat_w[step] = self.merge_stat(p, self.count_w[r][step])
                        error[step] = mergedstat_w[step]/mergedstat[step]*ROOT.TMath.Sqrt(mergedstat[step])
                        f.write("Events selection for {}: {} -> {}\n".format(p, step, mergedstat[step]))
                        f.write("Events weights for {}: {} -> {:.5f}+-{:.5f}\n\n".format(p, step, mergedstat_w[step],error[step]))
                    f.write("-----------------------------------------------------\n")
                f.write("####################################################\n")
            logger.info("Retain rates written to {}".format(file))
        return
                
    def defineHistos(self):
            
        """
        Method for defining the histograms of the variables of interest for every file and region analysed.
        
        """
        pi_inf = -ROOT.TMath.Pi()-0.01
        pi_sup = ROOT.TMath.Pi()+0.01
        for s in self.samples:
            self.df_1jet[s] = self.df[s].Filter("goodjet_n > 0")
            self.df_1jet[s] = self.df_1jet[s].Define("HJ_vector", "add_jet(H_vector, goodjet_pt, goodjet_eta, goodjet_phi, goodjet_E, goodjet_idx[0])")
                                           
            self.df_2jet[s] = self.df_1jet[s].Filter("goodjet_n > 1")\
                                             .Define("HJJ_vector", "add_jet(HJ_vector, goodjet_pt, goodjet_eta, goodjet_phi, goodjet_E, goodjet_idx[1])")\
                                             .Define("JJ_vector", "HJJ_vector - H_vector")
            self.histos[s] = {}
            for r, filter in zip(self.regions, self.filterRegion):
                region_df = self.df[s].Filter(filter)
                region_df_1jet = self.df_1jet[s].Filter(filter)
                region_df_2jet = self.df_2jet[s].Filter(filter)
                self.histos[s][r] = {}
                
                #BOSONS
                self.histos[s][r]['m4l'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "m4l", 24, 80, 170), "m4l", "weight")
                self.histos[s][r]['pt4l'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "pt4l", 24, 0, 200), "pt4l", "weight")
                self.histos[s][r]['eta4l'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "eta4l", 12, -6, 6), "eta4l", "weight")
                self.histos[s][r]['phi4l'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "phi4l", 12, pi_inf, pi_sup), "phi4l", "weight")

                self.histos[s][r]['mZ1'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "mZ1", 24, 50, 106), "mZ1", "weight")
                self.histos[s][r]['ptZ1'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "pt4l", 24, 0, 200), "ptZ1", "weight")
                self.histos[s][r]['etaZ1'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "eta4l", 10, -5, 5), "etaZ1", "weight")
                self.histos[s][r]['phiZ1'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "phiZ1", 12, pi_inf, pi_sup), "phiZ1", "weight")

                self.histos[s][r]['mZ2'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "mZ2", 24, 12, 115), "mZ2", "weight")
                self.histos[s][r]['ptZ2'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "ptZ2", 24, 0, 200), "ptZ2", "weight")
                self.histos[s][r]['etaZ2'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "etaZ2", 10, -5, 5), "etaZ2", "weight")
                self.histos[s][r]['phiZ2'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "phiZ2", 12, pi_inf, pi_sup), "phiZ2", "weight")
                
                #JETS
                
                self.histos[s][r]['njet'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "njet", 6, 0, 6), "goodjet_n", "weight")
                
                
                self.histos[s][r]['ptjet1'] = region_df.Filter("goodjet_n > 0").Define("ptjet1","goodjet_pt[goodjet_idx[0]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "ptjet1", 17, 30, 370), "ptjet1", "weight")
                self.histos[s][r]['etajet1'] = region_df.Filter("goodjet_n > 0").Define("etajet1","goodjet_eta[goodjet_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "etajet1", 6, -3, 3), "etajet1", "weight")
                self.histos[s][r]['phijet1'] = region_df.Filter("goodjet_n > 0").Define("phijet1","goodjet_phi[goodjet_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "phijet1", 12, pi_inf, pi_sup), "phijet1", "weight")
                
                
                self.histos[s][r]['ptjet2'] = region_df.Filter("goodjet_n > 1").Define("ptjet2","goodjet_pt[goodjet_idx[1]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "ptjet2", 17, 30, 370), "ptjet2", "weight")
                self.histos[s][r]['etajet2'] = region_df.Filter("goodjet_n > 1").Define("etajet2","goodjet_eta[goodjet_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "etajet2", 6, -3, 3), "etajet2", "weight")
                self.histos[s][r]['phijet2'] = region_df.Filter("goodjet_n > 1").Define("phijet2","goodjet_phi[goodjet_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "phijet2", 12, pi_inf, pi_sup), "phijet2", "weight")
                
                
                self.histos[s][r]['ptjet3'] = region_df.Filter("goodjet_n > 2").Define("ptjet3","goodjet_pt[goodjet_idx[2]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "ptjet3", 17, 30, 370), "ptjet3", "weight")
                self.histos[s][r]['etajet3'] = region_df.Filter("goodjet_n > 2").Define("etajet3","goodjet_eta[goodjet_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "etajet3", 6, -3, 3), "etajet3", "weight")
                self.histos[s][r]['phijet3'] = region_df.Filter("goodjet_n > 2").Define("phijet3","goodjet_phi[goodjet_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "phijet3", 12, pi_inf, pi_sup), "phijet3", "weight")
                
                #TOPOLOGICAL
                
                self.histos[s][r]['costhetastar'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "costhetastar", 12, -1, 1), "costhetastar", "weight")
                
                self.histos[s][r]['costheta1_H'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "costheta1_H", 12, -1, 1), "costheta1_H", "weight")
                self.histos[s][r]['costheta2_H'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "costheta2_H", 12, -1, 1), "costheta2_H", "weight")
                
                self.histos[s][r]['costheta1_Z1'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "costheta1_Z1", 12, -1, 1), "costheta1_Z1", "weight")
                self.histos[s][r]['costheta2_Z2'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "costheta2_Z2", 12, -1, 1), "costheta2_Z2", "weight")
                
                self.histos[s][r]['phi'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "phi", 12, pi_inf, pi_sup), "phi", "weight")
                self.histos[s][r]['phi1'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "phi1", 12, pi_inf, pi_sup), "phi1", "weight")
                
                
                
                self.histos[s][r]['deltaphiZ1Z2'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "deltaphiZ1Z2", 12, pi_inf, pi_sup), "deltaphiZ1Z2", "weight")
                
                self.histos[s][r]['deltaphiZ1Z2com'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "deltaphiZ1Z2com", 12, pi_inf, pi_sup), "deltaphiZ1Z2com", "weight")
                
                self.histos[s][r]['mZ1mZ2'] = region_df.Histo2D(ROOT.RDF.TH2DModel(s, "mZ1mZ2", 24, 50, 106, 24, 12, 115), "mZ1", "mZ2", "weight")
                
                self.histos[s][r]['deltaphizzpt'] = region_df.Histo2D(ROOT.RDF.TH2DModel(s, "deltaphizzpt", 24, 0, 200, 12, pi_inf, pi_sup), "pt4l", "deltaphiZ1Z2", "weight")
                
                self.histos[s][r]['m4lJ'] = region_df_1jet.Define("m4lJ","HJ_vector.M()/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "m4lJ", 10, 110, 500), "m4lJ", "weight")
                self.histos[s][r]['pt4lJ'] = region_df_1jet.Define("pt4lJ","HJ_vector.Pt()/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "pt4lJ", 12, 0, 250), "pt4lJ", "weight")
                self.histos[s][r]['deltaeta4lJ'] = region_df_1jet.Define("deltaeta4lJ","TMath::Abs(goodjet_eta[goodjet_idx[0]] - H_vector.Eta())").Histo1D(ROOT.RDF.TH1DModel(s, "deltaeta4lJ", 12, 0, 6), "deltaeta4lJ", "weight")
                self.histos[s][r]['deltaphi4lJ'] = region_df_1jet.Define("deltaphi4lJ","compute_deltaphi(goodjet_phi[goodjet_idx[0]], H_vector.Phi())").Histo1D(ROOT.RDF.TH1DModel(s, "deltaphi4lJ", 12, pi_inf, pi_sup), "deltaphi4lJ", "weight")
                
                self.histos[s][r]['m4lJJ'] = region_df_2jet.Define("m4lJJ","HJJ_vector.M()/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "m4lJJ", 12, 150, 600), "m4lJJ", "weight")
                self.histos[s][r]['pt4lJJ'] = region_df_2jet.Define("pt4lJJ","HJJ_vector.Pt()/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "pt4lJJ", 12, 0, 250), "pt4lJJ", "weight")
                self.histos[s][r]['mJJ'] = region_df_2jet.Define("mJJ","JJ_vector.M()/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "mJJ", 12, 0, 500), "mJJ", "weight")
                self.histos[s][r]['ptJJ'] = region_df_2jet.Define("ptJJ","JJ_vector.Pt()/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "ptJJ", 12, 0, 250), "ptJJ", "weight")
                self.histos[s][r]['deltaetaJJ'] = region_df_2jet.Define("deltaetaJJ","TMath::Abs(goodjet_eta[goodjet_idx[0]] - goodjet_eta[goodjet_idx[1]])").Histo1D(ROOT.RDF.TH1DModel(s, "deltaetaJJ", 12, 0, 5), "deltaetaJJ", "weight")
                self.histos[s][r]['deltaphiJJ'] = region_df_2jet.Define("deltaphiJJ","compute_deltaphi(goodjet_phi[goodjet_idx[0]], goodjet_phi[goodjet_idx[1]])").Histo1D(ROOT.RDF.TH1DModel(s, "deltaphiJJ", 10, pi_inf, pi_sup), "deltaphiJJ", "weight")
                
                #LEPTONS
                self.histos[s][r]['pt1'] = region_df.Define("pt1", "goodlep_pt[goodlep_idx[0]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "pt1", 24, 5, 200), "pt1", "weight")
                self.histos[s][r]['pt2'] = region_df.Define("pt2", "goodlep_pt[goodlep_idx[1]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "pt2", 24, 5, 200), "pt2", "weight")
                self.histos[s][r]['pt3'] = region_df.Define("pt3", "goodlep_pt[goodlep_idx[2]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "pt3", 24, 5, 200), "pt3", "weight")
                self.histos[s][r]['pt4'] = region_df.Define("pt4", "goodlep_pt[goodlep_idx[3]]/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "pt4", 24, 5, 200), "pt4", "weight")
                self.histos[s][r]['ptall'] = region_df.Define("ptall", "goodlep_pt/1000.").Histo1D(ROOT.RDF.TH1DModel(s, "ptall", 24, 5, 200), "ptall", "weight")
                
                self.histos[s][r]['eta1'] = region_df.Define("eta1", "goodlep_eta[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "eta1", 5, -2.7, 2.7), "eta1", "weight")
                self.histos[s][r]['eta2'] = region_df.Define("eta2", "goodlep_eta[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "eta2", 5, -2.7, 2.7), "eta2", "weight")
                self.histos[s][r]['eta3'] = region_df.Define("eta3", "goodlep_eta[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "eta3", 5, -2.7, 2.7), "eta3", "weight")
                self.histos[s][r]['eta4'] = region_df.Define("eta4", "goodlep_eta[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "eta4", 5, -2.7, 2.7), "eta4", "weight")
                self.histos[s][r]['etaall'] = region_df.Define("etaall", "goodlep_eta").Histo1D(ROOT.RDF.TH1DModel(s, "etaall", 5, -2.5, 2.5), "etaall", "weight")
                
                self.histos[s][r]['phi1lep'] = region_df.Define("phi1lep", "goodlep_phi[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "phi1lep", 12, pi_inf, pi_sup), "phi1lep", "weight")
                self.histos[s][r]['phi2lep'] = region_df.Define("phi2lep", "goodlep_phi[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "phi2lep", 12, pi_inf, pi_sup), "phi2lep", "weight")
                self.histos[s][r]['phi3lep'] = region_df.Define("phi3lep", "goodlep_phi[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "phi3lep", 12, pi_inf, pi_sup), "phi3lep", "weight")
                self.histos[s][r]['phi4lep'] = region_df.Define("phi4lep", "goodlep_phi[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "phi4lep", 12, pi_inf, pi_sup), "phi4lep", "weight")
                self.histos[s][r]['phiall'] = region_df.Define("phiall", "goodlep_phi").Histo1D(ROOT.RDF.TH1DModel(s, "phiall", 12, pi_inf, pi_sup), "phiall", "weight")
                
                self.histos[s][r]['isotrack1'] = region_df.Define("isotrack1", "goodlep_ptcone30[goodlep_idx[0]]/goodlep_pt[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "isotrack1", 24, 0.001, 0.3), "isotrack1", "weight")
                self.histos[s][r]['isotrack2'] = region_df.Define("isotrack2", "goodlep_ptcone30[goodlep_idx[1]]/goodlep_pt[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "isotrack2", 24, 0.001, 0.3), "isotrack2", "weight")
                self.histos[s][r]['isotrack3'] = region_df.Define("isotrack3", "goodlep_ptcone30[goodlep_idx[2]]/goodlep_pt[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "isotrack3", 24, 0.001, 0.3), "isotrack3", "weight")
                self.histos[s][r]['isotrack4'] = region_df.Define("isotrack4", "goodlep_ptcone30[goodlep_idx[3]]/goodlep_pt[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "isotrack4", 24, 0.001, 0.3), "isotrack4", "weight")
                self.histos[s][r]['isotrackall'] = region_df.Define("isotrackall", "goodlep_ptcone30/goodlep_pt").Histo1D(ROOT.RDF.TH1DModel(s, "isotrackall", 24, 0.001, 0.3), "isotrackall", "weight")
                self.histos[s][r]['isotrackalllog'] = region_df.Define("isotrackall", "goodlep_ptcone30/goodlep_pt").Histo1D(ROOT.RDF.TH1DModel(s, "isotrackall", 24, 0., 0.3), "isotrackall", "weight")
                
                self.histos[s][r]['isocal1'] = region_df.Define("isocal1", "goodlep_etcone20[goodlep_idx[0]]/goodlep_pt[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "isocal1", 24, -0.1, 0.3), "isocal1", "weight")
                self.histos[s][r]['isocal2'] = region_df.Define("isocal2", "goodlep_etcone20[goodlep_idx[1]]/goodlep_pt[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "isocal2", 24, -0.1, 0.3), "isocal2", "weight")
                self.histos[s][r]['isocal3'] = region_df.Define("isocal3", "goodlep_etcone20[goodlep_idx[2]]/goodlep_pt[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "isocal3", 24, -0.1, 0.3), "isocal3", "weight")
                self.histos[s][r]['isocal4'] = region_df.Define("isocal4", "goodlep_etcone20[goodlep_idx[3]]/goodlep_pt[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "isocal4", 24, -0.1, 0.3), "isocal4", "weight")
                self.histos[s][r]['isocalall'] = region_df.Define("isocalall", "goodlep_etcone20/goodlep_pt").Histo1D(ROOT.RDF.TH1DModel(s, "isocalall", 24, -0.1, 0.3), "isocalall", "weight")
                
                self.histos[s][r]['isolep1'] = region_df.Define("isolep1", "(goodlep_ptcone30[goodlep_idx[0]]+0.4*goodlep_etcone20[goodlep_idx[0]])/goodlep_pt[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "isolep1", 24, -0.1, 0.16), "isolep1", "weight")
                self.histos[s][r]['isolep2'] = region_df.Define("isolep2", "(goodlep_ptcone30[goodlep_idx[1]]+0.4*goodlep_etcone20[goodlep_idx[1]])/goodlep_pt[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "isolep2", 24, -0.1, 0.16), "isolep2", "weight")
                self.histos[s][r]['isolep3'] = region_df.Define("isolep3", "(goodlep_ptcone30[goodlep_idx[2]]+0.4*goodlep_etcone20[goodlep_idx[2]])/goodlep_pt[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "isolep3", 24, -0.1, 0.16), "isolep3", "weight")
                self.histos[s][r]['isolep4'] = region_df.Define("isolep4", "(goodlep_ptcone30[goodlep_idx[3]]+0.4*goodlep_etcone20[goodlep_idx[3]])/goodlep_pt[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "isolep4", 24, -0.1, 0.16), "isolep4", "weight")
                self.histos[s][r]['isolepall'] = region_df.Define("isolepall", "(goodlep_ptcone30+0.4*goodlep_etcone20)/goodlep_pt").Histo1D(ROOT.RDF.TH1DModel(s, "isolepall", 24, -0.1, 0.16), "isolepall", "weight")
                
                self.histos[s][r]['z0sintheta1'] = region_df.Define("z0sintheta1", "compute_z0sintheta(goodlep_pt[goodlep_idx[0]], goodlep_eta[goodlep_idx[0]], goodlep_phi[goodlep_idx[0]], goodlep_E[goodlep_idx[0]], goodlep_z0[goodlep_idx[0]])").Histo1D(ROOT.RDF.TH1DModel(s, "z0sintheta1", 24, -0.3, 0.3), "z0sintheta1", "weight")
                self.histos[s][r]['z0sintheta2'] = region_df.Define("z0sintheta2", "compute_z0sintheta(goodlep_pt[goodlep_idx[1]], goodlep_eta[goodlep_idx[1]], goodlep_phi[goodlep_idx[1]], goodlep_E[goodlep_idx[1]], goodlep_z0[goodlep_idx[1]])").Histo1D(ROOT.RDF.TH1DModel(s, "z0sintheta2", 24, -0.3, 0.3), "z0sintheta2", "weight")
                self.histos[s][r]['z0sintheta3'] = region_df.Define("z0sintheta3", "compute_z0sintheta(goodlep_pt[goodlep_idx[2]], goodlep_eta[goodlep_idx[2]], goodlep_phi[goodlep_idx[2]], goodlep_E[goodlep_idx[2]], goodlep_z0[goodlep_idx[2]])").Histo1D(ROOT.RDF.TH1DModel(s, "z0sintheta3", 24, -0.3, 0.3), "z0sintheta3", "weight")
                self.histos[s][r]['z0sintheta4'] = region_df.Define("z0sintheta4", "compute_z0sintheta(goodlep_pt[goodlep_idx[3]], goodlep_eta[goodlep_idx[3]], goodlep_phi[goodlep_idx[3]], goodlep_E[goodlep_idx[3]], goodlep_z0[goodlep_idx[3]])").Histo1D(ROOT.RDF.TH1DModel(s, "z0sintheta4", 24, -0.3, 0.3), "z0sintheta4", "weight")
                self.histos[s][r]['z0sinthetaall'] = region_df.Define("z0sinthetaall", "compute_z0sinthetavec(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E, goodlep_z0)").Histo1D(ROOT.RDF.TH1DModel(s, "z0sinthetaall", 24, -0.3, 0.3), "z0sinthetaall", "weight")
                
                self.histos[s][r]['d0sd01'] = region_df.Define("d0sd01", "goodlep_sigd0[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "d0sd01", 24, 0, 5), "d0sd01", "weight")
                self.histos[s][r]['d0sd02'] = region_df.Define("d0sd02", "goodlep_sigd0[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "d0sd02", 24, 0, 5), "d0sd02", "weight")
                self.histos[s][r]['d0sd03'] = region_df.Define("d0sd03", "goodlep_sigd0[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "d0sd03", 24, 0, 5), "d0sd03", "weight")
                self.histos[s][r]['d0sd04'] = region_df.Define("d0sd04", "goodlep_sigd0[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "d0sd04", 24, 0, 5), "d0sd04", "weight")
                self.histos[s][r]['d0sd0all'] = region_df.Define("d0sd0all", "goodlep_sigd0").Histo1D(ROOT.RDF.TH1DModel(s, "d0sd0all", 24, 0, 5), "d0sd0all", "weight")
                
                
                self.histos[s][r]['charge1'] = region_df.Define("charge1", "goodlep_charge[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "charge1", 5, -2.5, 2.5), "charge1", "weight")
                self.histos[s][r]['charge2'] = region_df.Define("charge2", "goodlep_charge[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "charge2", 5, -2.5, 2.5), "charge2", "weight")
                self.histos[s][r]['charge3'] = region_df.Define("charge3", "goodlep_charge[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "charge3", 5, -2.5, 2.5), "charge3", "weight")
                self.histos[s][r]['charge4'] = region_df.Define("charge4", "goodlep_charge[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "charge4", 5, -2.5, 2.5), "charge4", "weight")
                self.histos[s][r]['chargeall'] = region_df.Define("chargeall", "goodlep_charge").Histo1D(ROOT.RDF.TH1DModel(s, "chargeall", 5, -2.5, 2.5), "chargeall", "weight")
                
                self.histos[s][r]['type1'] = region_df.Define("type1", "goodlep_type[goodlep_idx[0]]").Histo1D(ROOT.RDF.TH1DModel(s, "type1", 5, 9.5, 14.5), "type1", "weight")
                self.histos[s][r]['type2'] = region_df.Define("type2", "goodlep_type[goodlep_idx[1]]").Histo1D(ROOT.RDF.TH1DModel(s, "type2", 5, 9.5, 14.5), "type2", "weight")
                self.histos[s][r]['type3'] = region_df.Define("type3", "goodlep_type[goodlep_idx[2]]").Histo1D(ROOT.RDF.TH1DModel(s, "type3", 5, 9.5, 14.5), "type3", "weight")
                self.histos[s][r]['type4'] = region_df.Define("type4", "goodlep_type[goodlep_idx[3]]").Histo1D(ROOT.RDF.TH1DModel(s, "type4", 5, 9.5, 14.5), "type4", "weight")
                self.histos[s][r]['typeall'] = region_df.Define("typeall", "goodlep_type").Histo1D(ROOT.RDF.TH1DModel(s, "typeall", 5, 9.5, 14.5), "typeall", "weight")
        return
    
    def merge_histos(self, label, observable, region):
        """
        Method for merging the histograms of the same process category (different Higgs production modes, real data, zz_bkg, other_bkg).
        It is invoked by :py:meth:`plotHistos()` method.
        
        :param label: The category under which the samples are to be merged.
        :type label: str
        :param observable: The name of the observed physical variable.
        :type observable: str
        :param region: The name of region.
        :type region: str
        :return: The merged histogram.
        :rtype: ROOT::TH1D
        
        """
        h = None
        j = 0
        for i, d in enumerate(self.files[label]):
            try:
                t = self.histos[d[2]][region][observable].GetValue()
                j += 1
            except:
                print("No histo {} for {} in region {}".format(d[2], observable, region))
            if j==1: h = t.Clone()
            elif j != 0: h.Add(t)
        h.SetNameTitle("{}_{}".format(label, observable), "{}_{}".format(label, observable))
        return h
    
    def plotHistos(self, z1z2):
        """
        Method for plotting the merged histograms of the same process category (different Higgs production modes, real data, zz_bkg, other_bkg) as a THStack and superimposing the real data distributions.
        The histograms will be saved in the configured output directory and organized by particle name.
        If the 'z1z2' flag is True then it wll be plotted the 2-dimentional histogram of mZ1 vs mZ2 (only real data).
        
        :param z1z2: Flag for plotting mZ1 vs mZ2.
        :type z1z2: bool
        
        """
        
        data = {}
        ggf = {}
        vbf = {}
        vba = {}
        zz = {}
        other = {}
        stack = {}
        c = {}
        pad0 = {}
        pad1 = {}
        padX = {}
        pad2X = {}
        hline = {}
        h_ratio = {}
        h_ratio_out = {}
        err = {}
        histoweights_m4l = {}
        
        outputdirs = []
        for particle in self.particles:
            for region in self.regions:
                outputdirs.append('{}/{}/{}'.format(self.cfg.OutputHistosPath, particle, region))
                
        for dir in outputdirs:
            try:
                os.makedirs(dir)
                logger.info("Created folder {} for ouput histograms.".format(dir))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    logger.error("Could not create folder {} for output histograms.".format(dir))
        
        with open(self.cfg.OutputPlotsStructure) as g:
            structure = json.load(g)
        allobs = structure.keys()
        for r, rl in zip(self.regions, self.regionlabel):
            for out_folder in allobs:
                for graph in structure[out_folder]:
                    observable = graph[0]
                    xaxislabel = graph[1]
                    xaxisunit = graph[2]
                    
                    data[observable] = self.merge_histos("data", observable, r)
                    ggf[observable] = self.merge_histos("ggF", observable, r)
                    vbf[observable] = self.merge_histos("VBF", observable, r)
                    vba[observable] = self.merge_histos("VBA", observable, r)
                    zz[observable] = self.merge_histos("zz", observable, r)
                    other[observable] = self.merge_histos("other", observable, r)
                    
                    if ('4l' in r and 'm4l' in observable):
                        histoweights_m4l['data'] = [data[observable].GetSumOfWeights(), data[observable].GetEntries()]
                        histoweights_m4l['ggf']  = [ggf[observable].GetSumOfWeights(), ggf[observable].GetEntries()]
                        histoweights_m4l['vbf']  = [vbf[observable].GetSumOfWeights(), vbf[observable].GetEntries()]
                        histoweights_m4l['vba']  = [vba[observable].GetSumOfWeights(), vba[observable].GetEntries()]
                        histoweights_m4l['zz']   = [zz[observable].GetSumOfWeights(), zz[observable].GetEntries()]
                        histoweights_m4l['other']= [other[observable].GetSumOfWeights(), other[observable].GetEntries()]
                    
                    axis_label = xaxislabel + xaxisunit

                    
                    # Set styles
                    ROOT.gROOT.SetStyle("ATLAS")
                    ROOT.gStyle.SetHatchesLineWidth(1)

                    # Create canvas with pad
                    c[observable] = ROOT.TCanvas("c_{}{}".format(observable, r), "c_{}{}".format(observable, r), 700, 750)
                    c[observable].cd()
                    pad0[observable] = ROOT.TPad("pad0_{}{}".format(observable, r), "pad0_{}{}".format(observable, r), 0, 0.29, 1, 1, 0, 0, 0)
                    
                    pad0[observable].SetTickx(False)
                    pad0[observable].SetTicky(False)
                    pad0[observable].SetTopMargin(0.05)
                    pad0[observable].SetBottomMargin(0)
                    pad0[observable].SetLeftMargin(0.14)
                    pad0[observable].SetRightMargin(0.05)
                    pad0[observable].SetFrameBorderMode(0)
                    pad0[observable].SetTopMargin(0.06)
                    if "log" in observable: pad0[observable].SetLogy()
                    
                    pad1[observable] = ROOT.TPad("pad1_{}{}".format(observable, r), "pad1_{}{}".format(observable, r), 0, 0, 1, 0.29, 0, 0, 0)
                    pad1[observable].SetTickx(False)
                    pad1[observable].SetTicky(False)
                    pad1[observable].SetTopMargin(0.0)
                    pad1[observable].SetBottomMargin(0.5)
                    pad1[observable].SetLeftMargin(0.14)
                    pad1[observable].SetRightMargin(0.05)
                    pad1[observable].SetFrameBorderMode(0)
                    
                    pad1[observable].Draw()
                    pad0[observable].Draw()
                    
                    pad0[observable].cd()

                    # Draw stack with MC contributions
                    stack[observable] = ROOT.THStack()
                    for h, color in zip([other[observable], zz[observable], ggf[observable], vbf[observable], vba[observable]], [(155, 152, 204), (100, 192, 232), (255, 247, 77), (255, 133, 51), (51, 204, 51)]):
                        h.SetLineWidth(1)
                        h.SetLineColor(1)
                        h.SetFillColor(ROOT.TColor.GetColor(*color))
                        stack[observable].Add(h)
                    stack[observable].Draw("HIST")
                    stack[observable].GetXaxis().SetLabelSize(0.04)
                    stack[observable].GetXaxis().SetTitleSize(0.045)
                    stack[observable].GetXaxis().SetTitleOffset(1.3)
                    stack[observable].GetXaxis().SetTitle(axis_label)
                    stack[observable].GetYaxis().SetTitle("Events / bin")
                    stack[observable].GetYaxis().SetLabelSize(0.05)
                    stack[observable].GetYaxis().SetTitleSize(0.055)
                    stack[observable].GetYaxis().SetTitleOffset(1.1)
                    stack[observable].GetYaxis().SetLabelOffset(0.01)
                    stack[observable].SetMaximum(data[observable].GetBinContent(data[observable].GetMaximumBin())*1.6+2.45)
                    if "log" not in observable: stack[observable].SetMinimum(0.)
                    #if (observable == "z0sintheta1"): stack[observable].SetMaximum(data[observable].GetBinContent(data[observable].GetMaximumBin())*1.6+4.45)
                    stack[observable].GetYaxis().ChangeLabel(1, -1, 0)
                    
                    
                    stackHists = None
                    stackHists = stack[observable].GetHists()
                    htemp = None
                    htemp = stackHists.At(0).Clone()
                    for i in range(1, stackHists.GetSize()): htemp.Add(stackHists.At(i))
                    err[observable] = ROOT.TGraphAsymmErrors(htemp)
                    for bin in range(htemp.GetNbinsX()):
                        err[observable].SetPointEXhigh(bin, htemp.GetBinWidth(bin+1)/2)
                        err[observable].SetPointEXlow(bin, htemp.GetBinWidth(bin+1)/2)
                    err[observable].SetFillStyle(3254)
                    err[observable].SetFillColor(ROOT.kBlack)
                    err[observable].SetMarkerSize(0);
                    err[observable].SetLineColor(ROOT.kWhite)
                    err[observable].Draw("2 SAME")

                    # Draw data
                    data[observable].SetMarkerStyle(20)
                    data[observable].SetMarkerSize(1.2)
                    data[observable].SetLineWidth(2)
                    data[observable].SetLineColor(ROOT.kBlack)
                    data[observable].Draw("E SAME")

                    # Add legend
                    legend = None
                    legend = ROOT.TLegend(0.60, 0.65, 0.92, 0.92)
                    legend.SetTextFont(42)
                    legend.SetFillStyle(0)
                    legend.SetBorderSize(0)
                    legend.SetTextSize(0.035)
                    legend.SetTextAlign(32)
                    legend.AddEntry(data[observable], "Data" ,"ep")
                    legend.AddEntry(ggf[observable], "ggF", "f")
                    legend.AddEntry(vbf[observable], "VBF", "f")
                    legend.AddEntry(vba[observable], "ZH,WH", "f")
                    legend.AddEntry(zz[observable], "ZZ*", "f")
                    legend.AddEntry(other[observable], "Other", "f")
                    legend.AddEntry(err[observable], "Stat. Uncert.", "f")
                    legend.Draw("SAME")

                    # Add ATLAS label
                    text = None
                    text = ROOT.TLatex()
                    text.SetNDC()
                    text.SetTextFont(72)
                    text.SetTextSize(0.045)
                    text.DrawLatex(0.21, 0.86, "ATLAS")
                    text.SetTextFont(42)
                    text.DrawLatex(0.21 + 0.16, 0.86, "Open Data")
                    text.SetTextSize(0.04)
                    text.DrawLatex(0.21, 0.80, "#sqrt{s} = 13 TeV, 10 fb^{-1}")
                    text.SetTextSize(0.05)
                    text.DrawLatex(0.21, 0.72, "H #rightarrow ZZ* #rightarrow {}".format(rl))
                    
                    
                    pad1[observable].cd()
                    pad1[observable].GetFrame().SetY1(2)
                    pad1[observable].Draw()
                    #Ratio plots
                    h_ratio[observable] = data[observable].Clone("h_ratio_{}".format(observable))
                    h_ratio[observable].Divide(htemp)
                    h_ratio[observable].GetYaxis().SetTitle("Data / Pred   ")
                    
                    hline[observable] = ROOT.TLine(h_ratio[observable].GetXaxis().GetXmin(),1,h_ratio[observable].GetXaxis().GetXmax(),1)
                    hline[observable].SetLineColor(ROOT.kGray+2)
                    hline[observable].SetLineWidth(2)
                    hline[observable].SetLineStyle(1)
                    
                    ROOT.gStyle.SetEndErrorSize(4)
                    h_ratio[observable].GetYaxis().CenterTitle()
                    h_ratio[observable].GetYaxis().SetNdivisions(504, True)
                    if ("charge") in observable: h_ratio[observable].GetXaxis().SetNdivisions(5, True)
                    if ("type") in observable: h_ratio[observable].GetXaxis().SetNdivisions(4, True)

                    h_ratio[observable].Draw("0E1")
                    hline[observable].Draw()
                    
                    h_ratio[observable].SetMinimum(0)
                    h_ratio[observable].SetMaximum(3.5)
                    
                    
                    h_ratio[observable].GetXaxis().SetTitle(stack[observable].GetXaxis().GetTitle())
                    h_ratio[observable].GetXaxis().SetTitleSize(0.136)
                    h_ratio[observable].GetXaxis().SetLabelSize(0.12)
                    h_ratio[observable].GetXaxis().SetTitleOffset(1.3)
                    h_ratio[observable].GetYaxis().SetTitleSize(0.11)
                    h_ratio[observable].GetYaxis().SetTitleOffset(0.46)
                    h_ratio[observable].GetYaxis().SetLabelSize(0.112)
                    h_ratio[observable].GetYaxis().SetLabelOffset(0.01)
                    h_ratio[observable].GetXaxis().SetLabelOffset(0.035)
                    h_ratio[observable].SetMarkerSize(0.6)
                    h_ratio[observable].SetLineWidth(1)
                    

                    ROOT.gPad.RedrawAxis()
                    
                    pad1[observable].cd()
                    pad1[observable].cd().SetGridy()
                    
                    h_ratio[observable].Draw("SAME0E1")
                    h_ratio[observable].Draw("SAMEAXIS")
                    h_ratio[observable].GetYaxis().Draw()
                    h_ratio[observable].Draw("SAME0E1")
                    h_ratio_out[observable] = h_ratio[observable].Clone("h_ratio_out_{}".format(observable))
                    h_ratio_out[observable].SetMarkerSize(0)
                    h_ratio_out[observable].SetLineWidth(1)
                    h_ratio_out[observable].Draw("SAME0E0")
                    
                    pad0[observable].cd()
                    TAxis = None
                    Ay1 = stack[observable].GetXaxis()
                    Ay1.SetLabelSize(0)
                    Ay1.SetTitleSize(0)
                    
                    ROOT.gPad.RedrawAxis()
                    

                    # Save the plot
                    c[observable].SaveAs("{}/{}/{}/{}.pdf".format(self.cfg.OutputHistosPath, out_folder, r, observable))
            
            if z1z2 == True:
                cz = ROOT.TCanvas("cz_{}".format(r), "", 700, 750)
                cz.cd()
                cz.SetRightMargin(0.13)
                datamz1mz2 = self.merge_histos("data", "mZ1mZ2", r)
                datamz1mz2.GetXaxis().SetTitle("m_{Z1} [GeV]")
                datamz1mz2.GetYaxis().SetTitle("m_{Z2} [GeV]")
                datamz1mz2.Draw("COLZ")
                # Add ATLAS label
                text = None
                text = ROOT.TLatex()
                text.SetNDC()
                text.SetTextFont(72)
                text.SetTextSize(0.045)
                text.DrawLatex(0.21, 0.86, "ATLAS")
                text.SetTextFont(42)
                text.DrawLatex(0.21 + 0.16, 0.86, "Open Data")
                text.SetTextSize(0.04)
                text.DrawLatex(0.21, 0.80, "#sqrt{s} = 13 TeV, 10 fb^{-1}")
                text.SetTextSize(0.05)
                text.DrawLatex(0.21, 0.72, "H #rightarrow ZZ* #rightarrow {}".format(rl))
                cz.SaveAs("{}/Z_boson/{}/mZ1mZ2.pdf".format(self.cfg.OutputHistosPath, r))
        return histoweights_m4l
            
    def prepareForML(self):
        """
        Method for creating the training/test samples for XGBoost.
        
        """
        self.saveNTuples(self.cfg.PrepareMLTraining)
        return
        
    def fitMass(self, bdt = False):
        """
        Method for fitting the 4 lepton invariant mass using RooFit and extracting a measurement of the Higgs boson mass.
        
        It is an Unbinned Maximum Likelihood fit, where the background distribution is extracted via Kernel Density Estimation and the signal is modelled as a CrystalBall distribution. The parameters of the CrystalBall (except for the "mean") and the fraction signal/bkg are fixed on the values extracted from the MC samples. The only free parameter is then the Higgs mass. The 'bdt' flag is for correctly accessing the counters whether fitMass() is invoked before XGBoost has enhanced the signal purity of the sample or after, thus correcly fixing the fraction signal/bkg to yje correct value.
        
        :param bdt: Flag for plotting before or after XGBoost.
        :type bdt: bool
        
        """
        #Step tag accounting for halved MC simulation in case of testing/training and consequent reweighing
        if not bdt: step = list(self.count[self.regions[0]])[-1]
        else: step='bdt'
        # Create RooFit variables
        x = ROOT.RooRealVar("x", "m_{4l}", 110, 140, "GeV")
        w = ROOT.RooRealVar("w", "w", -1.0, 10.0)
        
        setperSamp =  {}
        for s in self.samples:
            if (self.df[s].HasColumn("bdt") and self.df[s].HasColumn("weight2")):
                dummyDataset = self.df[s].Filter("bdt==1").Book(ROOT.std.move(ROOT.RooDataSetHelper("dataset_"+s, "Four lepton mass distribution", ROOT.RooArgSet(x,w))), ("m4l", "weight2"))
            else:
                dummyDataset = self.df[s].Book(ROOT.std.move(ROOT.RooDataSetHelper("dataset_"+s, "Four lepton mass distribution", ROOT.RooArgSet(x,w))), ("m4l", "weight"))
            
            setperSamp[s] = ROOT.RooDataSet(dummyDataset.GetName(), dummyDataset.GetTitle(), dummyDataset.GetPtr(), dummyDataset.get(), "", w.GetName())
        #Creating the datasets
        bkg_inclusive=ROOT.RooDataSet("dataset_Data", "Four lepton mass distribution", ROOT.RooArgSet(x,w), w.GetName())
        sig_inclusive=ROOT.RooDataSet("dataset_Higgs", "Four lepton mass distribution", ROOT.RooArgSet(x,w), w.GetName())
        data_set=ROOT.RooDataSet("dataset_Bkg", "Four lepton mass distribution", ROOT.RooArgSet(x,w), w.GetName())
        
        #Adding datato the datasets
        for s in self.samples:
            print(s)
            setperSamp[s].Print()
            if 'data' in s:
                data_set.append(setperSamp[s])
            elif 'H125' in s:
                sig_inclusive.append(setperSamp[s])
            else:
                bkg_inclusive.append(setperSamp[s])
        #Calculate signal fraction for fit
        sig_frac_count = sig_inclusive.sumEntries()/(sig_inclusive.sumEntries() + bkg_inclusive.sumEntries())
        
        #KDE for background
        p2 = ROOT.RooKeysPdf("p2", "p2", x, bkg_inclusive, ROOT.RooKeysPdf.MirrorBoth)
        #Parameters and model for signal fit
        meanHiggs = ROOT.RooRealVar("meanHiggs", "The Higgs Mass CB", 123, 115, 135)
        sigmaHiggs = ROOT.RooRealVar("sigmaHiggs", "The width of Higgs mass CB", 5, 0., 20)
        alphaHiggs = ROOT.RooRealVar("alphaHiggs", "The alpha of Higgs mass CB", 1.5, -5, 5)
        nHiggs = ROOT.RooRealVar("nHiggs", "The n of Higgs mass CB", 1.5, 0, 10)
        CBHiggs = ROOT.RooCBShape("CBHiggs","The Higgs Crystall Ball",x,meanHiggs,sigmaHiggs,alphaHiggs,nHiggs)
        #Unbinned ML fit to signal
        fitHiggs = CBHiggs.fitTo(sig_inclusive, ROOT.RooFit.Save(True), ROOT.RooFit.AsymptoticError(True))
        
        fitHiggs.Print("v")
        #Parameters and model for data fit
        meanHiggs_data = ROOT.RooRealVar("m_{H}", "The Higgs Mass CB", 123, 115, 135, "GeV")
        sigmaHiggs_data = ROOT.RooRealVar("#Gamma_{H}", "The width of Higgs mass CB", sigmaHiggs.getValV(), "GeV")
        alphaHiggs_data = ROOT.RooRealVar("alphaHiggs_data", "The alpha of Higgs mass CB", alphaHiggs.getValV())
        nHiggs_data = ROOT.RooRealVar("nHiggs_data", "The n of Higgs mass CB", nHiggs.getValV())
        CBHiggs_data = ROOT.RooCBShape("CBHiggs_data","The Higgs Crystall Ball",x,meanHiggs_data,sigmaHiggs_data,alphaHiggs_data,nHiggs_data)
        #sigfrac= ROOT.RooRealVar("sigfrac", "fraction signal fraction", 0.8, 0., 1.)
        sigfrac= ROOT.RooRealVar("sigfrac", "fraction signal fraction", sig_frac_count)
        
        model = ROOT.RooAddPdf("model", "Higgs+irred. bkg", ROOT.RooArgList(CBHiggs_data, p2), sigfrac)
        #Unbinned ML fit to data
        fitdata = model.fitTo(data_set, ROOT.RooFit.Save(True), Minos=True, NumCPU=14)
        fitdata.Print("v")
        print("Fraction sig/bkg is:")
        print( sig_frac_count)
        
        # Plot weighted data and fit result
        x.setBins(12)
        # Construct plot frame
        frame = x.frame(ROOT.RooFit.Title("KDE of Background"))

        # Plot bkg accounting for weights
        bkg_inclusive.plotOn(frame, DataError="SumW2", Name="MCbkg")

        # Overlay KDE bkg
        p2.plotOn(frame, MoveToBack=True, DrawOption="F", FillColor="kOrange", Name="KDEbkg", LineWidth=0)
        
        
        # Construct plot frame
        frame2 = x.frame(ROOT.RooFit.Title("Unbinned ML fit to Signal"))

        # Plot signal accounting for weights
        sig_inclusive.plotOn(frame2, DataError="SumW2", Name="MCHiggs")

        # Overlay CrystalBall fit
        CBHiggs.plotOn(frame2, Name="CrystB")
        
        #Create canvas for MC mass ditributions
        canv2 = ROOT.TCanvas("Canvas2", "Canvas2", 800, 400)
        canv2.Divide(2)
        canv2.cd(2)
        ROOT.gPad.SetLeftMargin(0.18)
        frame.GetYaxis().SetTitleOffset(1.8)
        frame.Draw()
        #Legend
        legend1 = None
        legend1 = ROOT.TLegend(0.60, 0.74, 0.9, 0.9)
        legend1.SetTextFont(42)
        legend1.SetFillStyle(0)
        legend1.SetBorderSize(0)
        legend1.SetTextSize(0.03)
        legend1.SetTextAlign(32)
        legend1.AddEntry("MCbkg", "MC bkg." ,"ep")
        legend1.AddEntry("KDEbkg", "KDE bkg.", "f")
        legend1.Draw("SAME")
        
        # Add ATLAS label
        text1 = None
        text1 = ROOT.TLatex()
        text1.SetNDC()
        text1.SetTextFont(72)
        text1.SetTextSize(0.032)
        text1.DrawLatex(0.21, 0.86, "ATLAS")
        text1.SetTextFont(42)
        text1.DrawLatex(0.21 + 0.13, 0.86, "Open Data")
        text1.SetTextSize(0.028)
        text1.DrawLatex(0.21, 0.82, "#sqrt{s} = 13 TeV, 10 fb^{-1}")
        text1.SetTextSize(0.032)
        text1.DrawLatex(0.21, 0.77, "H #rightarrow ZZ* #rightarrow 4l")
        
        
        canv2.cd(1)
        ROOT.gPad.SetLeftMargin(0.18)
        frame2.GetYaxis().SetTitleOffset(1.8)
        frame2.Draw()
        #Legend
        legend2 = None
        legend2 = ROOT.TLegend(0.60, 0.74, 0.9, 0.9)
        legend2.SetTextFont(42)
        legend2.SetFillStyle(0)
        legend2.SetBorderSize(0)
        legend2.SetTextSize(0.03)
        legend2.SetTextAlign(32)
        legend2.AddEntry("MCHiggs", "MC Higgs" ,"ep")
        legend2.AddEntry("CrystB", "Cryst.Ball fit", "l")
        legend2.Draw("SAME")
        
        # Add ATLAS label
        text2 = None
        text2 = ROOT.TLatex()
        text2.SetNDC()
        text2.SetTextFont(72)
        text2.SetTextSize(0.032)
        text2.DrawLatex(0.21, 0.86, "ATLAS")
        text2.SetTextFont(42)
        text2.DrawLatex(0.21 + 0.13, 0.86, "Open Data")
        text2.SetTextSize(0.028)
        text2.DrawLatex(0.21, 0.82, "#sqrt{s} = 13 TeV, 10 fb^{-1}")
        text2.SetTextSize(0.032)
        text2.DrawLatex(0.21, 0.77, "H #rightarrow ZZ* #rightarrow 4l")
        canv2.SaveAs(f"{self.cfg.OutputHistosPath}/fit_{step}.pdf(")
        
        # Construct plot frame
        frame_data = x.frame(ROOT.RooFit.Title("Data"))

        # Plot data
        data_set.plotOn(frame_data, Name="data")
        #Signal+bkg model with a full line
        model.plotOn(frame_data, Name="model", LineColor="r")

        #Background component of model with filled orange
        model.plotOn(frame_data,Name="zz",Components={p2},MoveToBack=True, DrawOption="F", FillColor="kOrange", LineWidth=0)
        #Signal component of model with a dashed line
        model.plotOn(frame_data,Name="higgs",Components={CBHiggs_data},LineStyle="--")
        model.paramOn(frame_data, ROOT.RooFit.Format("NEU"),ROOT.RooFit.AutoPrecision(2), ROOT.RooFit.Layout(0.60,0.85, 0.95))
        frame_data.GetYaxis().SetTitleOffset(1.8)
        frame_data.Draw()
        
        #Legend
        legend = None
        legend = ROOT.TLegend(0.60, 0.60, 0.90, 0.82)
        legend.SetTextFont(42)
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)
        legend.SetTextSize(0.03)
        legend.SetTextAlign(32)
        legend.AddEntry("data", "Data" ,"ep")
        legend.AddEntry("model", "Higgs+bkg", "l")
        legend.AddEntry("higgs", "Higgs", "l")
        legend.AddEntry("zz", "ZZ bkg", "f")
        legend.Draw("SAME")
        
        # Add ATLAS label
        text = None
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextFont(72)
        text.SetTextSize(0.032)
        text.DrawLatex(0.21, 0.86, "ATLAS")
        text.SetTextFont(42)
        text.DrawLatex(0.21 + 0.1, 0.86, "Open Data")
        text.SetTextSize(0.028)
        text.DrawLatex(0.21, 0.82, "#sqrt{s} = 13 TeV, 10 fb^{-1}")
        text.SetTextSize(0.032)
        text.DrawLatex(0.21, 0.77, "H #rightarrow ZZ* #rightarrow 4l")
        
        canv.SaveAs(f"{self.cfg.OutputHistosPath}/fit_{step}.pdf)")
        
        #Export the fits results
        stdout = sys.stdout
        sys.stdout = open(f"{self.cfg.OutputHistosPath}/fit_mass.log", 'w')
        print(f"############ FIT MASS @ {step} ##############")
        print("Signal CrystalBall fit")
        fitHiggs.Print("v")
        print("Signal data fit")
        fitdata.Print("v")
        print(f"Fraction sig/(sig.+bkg.) is: {sig_frac_count}")
        sys.stdout.close()
        sys.stdout = stdout
        
        return
        
    
if __name__ == "__main__":
    from checkRequirements import checkRequirements
    from loadconfig import loadSettings
    from datetime import datetime
    from XGBoostHiggs import traintest, apply_bdt

    start=datetime.now()

    logging.basicConfig(filename='proclog.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(screen_handler)
    checkRequirements()
    AnaCfg = loadSettings("conf.yaml")
    AnaVars= HiggsProcessor(AnaCfg)
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
    
    boosted_tree = traintest(AnaCfg)
    apply_bdt(AnaCfg, boosted_tree)
    
    logging.info("Analysis and plot run in: {}".format(datetime.now()-start))
