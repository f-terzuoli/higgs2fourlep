#XGBoostHiggs.py
"""
This module manages the Machine Learning aspect of the Higgs analysis. It trains and tests the XGBoost Classifier and apllies it to the real data sample.
"""

import ROOT
import numpy as np
#import pickle
import glob
import logging
import matplotlib.pyplot as plt
from .processData import HiggsProcessor, json

logger = logging.getLogger(__name__)
variables = ["m4l", "pt4l", "eta4l", "mZ1", "ptZ1", "etaZ1", "mZ2", "ptZ2", "etaZ2", "deltaphiZ1Z2", "goodjet_n", "pt1", "pt3", "eta1", "eta3", "costheta1_Z1", "costheta2_Z2"]

def load_data(cfg, testtrain):
    """
    Method loading the data from the train/test samples as Numpy arrays.
    
    :param cfg: The loaded configuration settings.
    :type cfg: AnalysisCfg
    :param testtrain: String identifing if 'Training' or 'Testing' samples are to be loaded.
    :type testtrain: str
    :returns:
        - x(:py:class:`numpy.ndarray`) - input observables to the boosted decision tree
        - y(:py:class:`numpy.ndarray`) - boolean discrimination between signal and background
        - w(:py:class:`numpy.ndarray`) - weight of every event processed
    
    """
    ROOT.ROOT.EnableImplicitMT(cfg.NThreads)
    fileNtuples = glob.glob(f"{cfg.SkimmedNtuplesPath}/*.root")
    signalNtuples = ROOT.std.vector("string")()
    bkgNtuples = ROOT.std.vector("string")()
    # Read data from ROOT files
    for f in fileNtuples:
        if testtrain in f:
            if 'H125' in f:
                signalNtuples.push_back(f)
            else:
                bkgNtuples.push_back(f)
        else: pass
    
            
    
    data_sig1 = ROOT.RDataFrame("RecoTree", signalNtuples)\
                                                        .Filter("m4l>110. && m4l<140 && weight>0")\
                                                        .Define("pt1", "goodlep_pt[goodlep_idx[0]]/1000.")\
                                                        .Define("pt3", "goodlep_pt[goodlep_idx[2]]/1000.")\
                                                        .Define("eta1", "goodlep_eta[goodlep_idx[0]]/1000.")\
                                                        .Define("eta3", "goodlep_eta[goodlep_idx[2]]/1000.")
    data_bkg1 = ROOT.RDataFrame("RecoTree", bkgNtuples).Filter("m4l>110. && m4l<138 && weight>0")\
                                                      .Define("pt1", "goodlep_pt[goodlep_idx[0]]/1000.")\
                                                      .Define("pt3", "goodlep_pt[goodlep_idx[2]]/1000.")\
                                                        .Define("eta1", "goodlep_eta[goodlep_idx[0]]/1000.")\
                                                        .Define("eta3", "goodlep_eta[goodlep_idx[2]]/1000.")
                                                        
    sig_count = data_sig1.Sum("weight").GetValue()
    bkg_count = data_bkg1.Sum("weight").GetValue()
    
    #print(sig_count)
    #print(bkg_count)
    
    data_sig = data_sig1.AsNumpy()
    data_bkg = data_bkg1.AsNumpy()

    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([data_sig[var] for var in variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in variables]).T
    x = np.vstack([x_sig, x_bkg])
 
    # Create labels
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
 
    # Compute weights
    if 'Testing' in testtrain:
        w = np.hstack([20*data_sig["weight"], 20*data_bkg["weight"]])
    else:
        w = np.hstack([20*data_sig["weight"], 20*data_bkg["weight"]])
    return x, y, w


def traintest(cfg):
    """
    Method for training and testing the XGBClassifier, computing and saving the ROC Curve, the learning curve, the Accuracy and the F1-Score.
    
    :param cfg: The loaded configuration settings.
    :type cfg: AnalysisCfg
    :returns: trained XGBClassifier
    :rtype: XGBClassifier
    
    """
    # Load data
    x, y, w = load_data(cfg, 'Testing')
 
    # Load data
    x_test, y_test, w_test = load_data(cfg, 'Training')
    
    # Fit xgboost model
    from xgboost import XGBClassifier
    bdt = XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=3, n_estimators=5000, eta=0.3, subsample=1, colsample_bytree=1, nthread=cfg.NThreads, min_child_weight = 0.001)
    bdt.fit(x, y, sample_weight=w, eval_set =  [(x,y),(x_test,y_test)], sample_weight_eval_set=[w,w_test])
    
    # Make prediction
    y_pred = bdt.predict(x_test)
    y_pred_p = bdt.predict_proba(x_test).T[1]
    
    
    # Compute ROC using sklearn
    from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_p, sample_weight=w_test)
    score = auc(fpr, tpr)
    
    fp, tp, _ = roc_curve(y_test, y_pred, sample_weight=w_test)
    
    f1_score = f1_score(y_test, y_pred, sample_weight=w_test)
    #print(f1_score)
    
    acc_score = accuracy_score(y_test, y_pred, sample_weight=w_test)
    #print(acc_score)
     
    # Plot ROC
    c = ROOT.TCanvas("roc", "", 600, 600)
    g = ROOT.TGraph(len(fpr), fpr, tpr)
    g.SetTitle("AUC = {:.3f}".format(score))
    g.SetLineWidth(2)
    g.SetLineColor(ROOT.kRed)
    g.Draw("AL")
    g.GetXaxis().SetRangeUser(0, 1)
    g.GetYaxis().SetRangeUser(0, 1)
    g.GetXaxis().SetTitle("False-positive rate")
    g.GetYaxis().SetTitle("True-positive rate")
    
    p = ROOT.TGraph(1, fp[1], tp[1])
    p.SetMarkerStyle(41)
    p.SetMarkerSize(1.15)
    p.SetMarkerColor(ROOT.kBlue)
    p.Draw("P same")
    
    c.SaveAs("roc.pdf")
    
    # retrieve performance metrics
    results = bdt.evals_result()
    # plot learning curves
    plt.title(f"Learning curve acc={acc_score:.2f} f1score={f1_score:.2f}")
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='test')
    #plt.ylim([0, 0.7])
    #plt.yscale('log')
    plt.grid()
    # show the legend
    plt.legend()
    plt.savefig('logloss.pdf', dpi=300)
    # show the plot
    plt.draw()
    
    return bdt
    
def apply_bdt(cfg, trainedtree):
    """
    Method for applying the trained XGBClassifier on the real data sample, after it is converted to Numpy multidimentional arrays.
    It is also re-applied to the testing samples. Then all the datasets are reconverted as ROOT::RDataframe, with the addition of the discriminating variable.
    It makes use of an internal instance of HiggsProcessor for producing the outputs.
    The 4-lepton invariant mass is plotted using the HiggsProcessor.plotHistos() method and the the mass measurement extracted via HiggsProcessor.fitMass().
    
    :param cfg: The loaded configuration settings.
    :type cfg: AnalysisCfg
    :param trainedtree: trained XGBClassifier
    :type trainedtree: XGBClassifier
    :raises: RuntimeError if some sample contains no data. It is handled by striking down the sample from the analysis workflow.
    """
    internalProcessor= HiggsProcessor(cfg)
    internalProcessor.initialized = True
    internalProcessor.reconstructedHiggs = True
    internalProcessor.regions.append('4l')
    internalProcessor.regionlabel.append('4l XGBoost')
    internalProcessor.filterRegion.append("kTRUE")
    internalProcessor.particles.append('BDT')
    with open(internalProcessor.cfg.InputAnalysisJSONFile) as f:
        internalProcessor.files = json.load(f)
    internalProcessor.processes = internalProcessor.files.keys()
    for p in internalProcessor.processes:
        for d in internalProcessor.files[p]:
            # Construct the dataframes
            folder = d[0] # Folder name
            nlep = d[1] # Lepton multiplicity
            sample = d[2] # Sample name
            internalProcessor.xsecs[sample] = d[3] # Cross-section
            internalProcessor.sumws[sample] = d[4] # Sum of weights
            internalProcessor.samples.append(sample)
    
    for s in internalProcessor.samples:
        Ntuples = ""
        if 'data' in s:
            Ntuples = "{}/Reco_{}.root".format(internalProcessor.cfg.SkimmedNtuplesPath,s)
        else:
            Ntuples = "{}/Training_{}.root".format(internalProcessor.cfg.SkimmedNtuplesPath,s)
        
        try:
            data_df = ROOT.RDataFrame("RecoTree", Ntuples).Filter("m4l>110. && m4l<140 && weight>0")\
                                                                 .Define("pt1", "goodlep_pt[goodlep_idx[0]]/1000.")\
                                                                 .Define("pt3", "goodlep_pt[goodlep_idx[2]]/1000.")\
                                                            .Define("eta1", "goodlep_eta[goodlep_idx[0]]/1000.")\
                                                            .Define("eta3", "goodlep_eta[goodlep_idx[2]]/1000.")
            
            if data_df.Count().GetValue() == 0:
                raise RuntimeError(f"No events for {dataNtuples} surviving selections")
            
            bdt_data = data_df.AsNumpy()
            
            # Convert inputs to format readable by machine learning tools
            x_sig = np.vstack([bdt_data[var] for var in variables]).T
            x = x_sig
         
            # Predict calssification on real data
            y = trainedtree.predict(x)
     
            bdt_data["bdt"] = y
    
            newdf ={}
            for var, values in bdt_data.items():
                if (var in variables) or (var in ["weight", "bdt"]):
                    newdf[var]=values
            #print(newdf)
            internalProcessor.df[s] = ROOT.RDF.MakeNumpyDataFrame(newdf)
            if 'data' not in s:
                internalProcessor.df[s] = internalProcessor.df[s].Define("weight2", "weight*2")
            else: internalProcessor.df[s] = internalProcessor.df[s].Define("weight2", "weight")
            snapping = internalProcessor.df[s].Snapshot("bdt_Higgs", f"{internalProcessor.cfg.SkimmedNtuplesPath}/BDT_{s}.root")
        
            internalProcessor.histos[s] = {}
            for r, filter in zip(internalProcessor.regions, internalProcessor.filterRegion):
                region_df = snapping.Filter(filter).Filter("bdt == 1")
                internalProcessor.histos[s][r] = {}
                internalProcessor.histos[s][r]['m4l'] = region_df.Histo1D(ROOT.RDF.TH1DModel(s, "m4l", 24, 80, 170), "m4l", "weight2")
        except RuntimeError:
            del internalProcessor.samples[s]
            pass
    
    counting = internalProcessor.plotHistos(False)
    internalProcessor.fitMass(bdt=True)
    
    file = f"{cfg.OutputYelds}/RetainRates_BDG.log"
    with open(file, "w+") as f:
        f.write("######### RETENTION RATES  AFTER XGBoost ###########\n")
        for cat in counting.keys():
            try:
                weights = counting[cat][0]
                entries = counting[cat][1]
                error_w = weights*ROOT.TMath.Sqrt(entries)/entries
            except ZeroDivisionError:
                error_w = 0.
            f.write("Events weights for {}: {} -> {:.5f}+-{:.5f}\n\n".format(cat, 'XGBoost', weights, error_w))
            f.write("-----------------------------------------------------\n")
        f.write("####################################################\n")
    return
 
if __name__ == "__main__":
    
    from loadconfig import loadSettings
    from datetime import datetime
    import sys
    from processData import HiggsProcessor
    import json

    start=datetime.now()

    logging.basicConfig(filename='loadlog.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', filemode = "w+")
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(screen_handler)
    AnaCfg = loadSettings("conf_bdt.yaml")
    
    boosted_tree = traintest(AnaCfg)
    apply_bdt(AnaCfg, boosted_tree)

