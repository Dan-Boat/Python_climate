# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:04:18 2020

@author: Boateng
"""

#modules and packages 

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from stat_downscaling_tools import MonthlyStandardizer, BlockBootstrapper
from stat_downscaling_tools.weatherstation_utils import read_station_from_csv, read_weatherstationnames

from regression_settings import *
from data import *
from io_utils import *
from plot_all import *
from plot_utils import predictions_plot



def predict_from_ML_station(i, variable):
    stationname = stationnames[i]
    filename = stationname.replace(" ", "_")
    ws = read_station_from_csv(os.path.join(station_datadir, filename + ".csv"))
    anomalies = variable = "Precipitation"
    
    #setting predictors
    #----Using predictors without regional dynamical control (eg. NAO, AO)
    # If implemented, the str must also be updated in the set_predictors function
    ws.set_predictors(variable, selected_predictors, predictordir, radius)
    
    #standardizing the dataset (to remove seasonal cycel and possible linear trend)
    ws.set_standardizer(variable, MonthlyStandardizer(detrending= False))
    
    #setting statistical model for fitting 
    #---(eg. OLSForward: Forward selection OLS regression, Lasso: least absolute shrinkage 
    # and selection operator, gamma: Gamma regression)
    #ws.set_model(variable, "svr", cv=None)
    ws.set_model(variable, "ann", cv=None)
    #fitting of model 
    ws.fit_predictor(variable, selected_predictors, fullERA, ERAData)
    
    
    #checking prediction score ---> 1979 to 2000
    yhat_01to15, score_79to00 = ws.fit_predict_score(variable, from79to00, from01to15, ERAData, 
                                                     anomalies= anomalies, fit_predictors=False
                                                     ) 
    
    #checking prediction score ---> 2001 to 2015
    yhat_79to00, score_01to15 = ws.fit_predict_score(variable, from01to15, from79to00, ERAData, 
                                                     anomalies= anomalies, fit_predictors=False)
    
    

    #full model 
    ws.set_predictors(variable, selected_predictors, predictordir, radius=radius)
    yhat_ERA_full, score_ERA_full = ws.fit_predict_score(variable, fullERA, fullERA, ERAData, anomalies=anomalies,
                                                         fit_predictors=False)
    yhat_ERA_no_anomalies = ws.predict(variable, fullERA, ERAData, anomalies= False)
    
    score_era= {"score 1979-2000": score_79to00["prediction score"],
                "score 2001-2015": score_01to15["prediction score"],
                "fit score": score_ERA_full["fitting score"],
                "rmse 1979-2000":score_79to00["prediction rmse"],
                "rmse 2001-2015": score_01to15["prediction rmse"],
                "fit_rmse": score_ERA_full["fitting rmse"]}
    store_pickle(stationname, "score_era_"+ variable, score_era)
    
    #ploting of pairplot..(I implemenented it the weatherstation package)
    #ws.pairplot(variable, fullERA, ERAData, fit=False)
    
  
    
    # using echam dataset 
    ws.set_predictors(variable, selected_predictors, predictordir, radius=radius)
    ws.fit_predictor(variable, selected_predictors, fullECHAM, ECHAMData)

    # prediction without fitting transfer function (use patterns from to include indices when implemented)
    yhat_ECHAM = ws.predict(variable, fullECHAM, ECHAMData,
                                anomalies=anomalies, params_from="ERA", patterns_from="ERA") 
    #Problem from the patterns_from
    yhat_ECHAM_no_anomalies = ws.predict(variable, fullECHAM, ECHAMData,
                                anomalies=False, params_from="ERA", patterns_from="ERA")
    scores_echam = {
        'fit score': ws.score(variable, fullECHAM, fullECHAM, ECHAMData),
        'fit rmse': ws.rmse(variable, fullECHAM, ECHAMData),
    }
    store_pickle(stationname, 'scores_echam_' + variable, scores_echam)
    
    #full model + CMIP5 Data
    #set predictors, standizer, model 
    ws.set_predictors(variable, selected_predictors, predictordir, radius=radius)
    ws.fit_predictor(variable, selected_predictors, fullAMIP, CMIP5_AMIP_R1)
    print("fitting of AMIP predictors based on full model")
    
    yhat_CMIP5_AMIP_R1 = ws.predict(variable, fullAMIP, CMIP5_AMIP_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 2.6 for full model")
    yhat_CMIP5_RCP26_R1 = ws.predict(variable, fullCMIP5, CMIP5_RCP26_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 4.5--realisationn 1 for full model")
    yhat_CMIP5_RCP45_R1 = ws.predict(variable, fullCMIP5, CMIP5_RCP45_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 4.5-- realisation 2 for full model")
    yhat_CMIP5_RCP45_R2 = ws.predict(variable, fullCMIP5, CMIP5_RCP45_R2, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 4.5-- realisation 3 for full model")
    yhat_CMIP5_RCP45_R3 = ws.predict(variable, fullCMIP5, CMIP5_RCP45_R3, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 8.5 for full model")
    yhat_CMIP5_RCP85_R1 = ws.predict(variable, fullCMIP5, CMIP5_RCP85_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
   
    
    #predicting using CMIP5 Data + 79to00 Model
   
    #set predictors, standardize, model, predict
    ws.set_predictors(variable, selected_predictors, predictordir, radius=radius)
    ws.fit_predictor(variable, selected_predictors, from79to00, CMIP5_AMIP_R1)
    
    print("Fitting params from AMIP and predict, model:79to00")
    yhat_m79to00_CMIP5_AMIP_R1 = ws.predict(variable, fullAMIP, CMIP5_AMIP_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 2.6 for full model")
    yhat_m79to00_CMIP5_RCP26_R1 = ws.predict(variable, fullCMIP5, CMIP5_RCP26_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 4.5--realisationn 1 for full model")
    yhat_m79to00_CMIP5_RCP45_R1 = ws.predict(variable, fullCMIP5, CMIP5_RCP45_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 4.5-- realisation 2 for full model")
    yhat_m79to00_CMIP5_RCP45_R2 = ws.predict(variable, fullCMIP5, CMIP5_RCP45_R2, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 4.5-- realisation 3 for full model")
    yhat_m79to00_CMIP5_RCP45_R3 = ws.predict(variable, fullCMIP5, CMIP5_RCP45_R3, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
    
    print("predict based on RCP 8.5 for full model")
    yhat_m79to00_CMIP5_RCP85_R1 = ws.predict(variable, fullCMIP5, CMIP5_RCP85_R1, anomalies=anomalies, params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
   
    # STORING RESULTS
    # ===============

    # store all the predictions in a file to create a single plot later
    obs = ws.get_var(variable, fullERA, True)  # read raw observational data (original)    
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):    
    #    print(obs)    
    obs_anm = ws.get_var(variable, fullERA, True)   # read observational data (and calculate anomalies)  
    predictions = pd.DataFrame({
        'obs': obs,   #this has a problem because of reindex
        'obs anomalies': obs_anm,        
        'ERA 1979-2000': yhat_79to00,
        'ERA 2001-2015': yhat_01to15,
        'ERA fit': yhat_ERA_full,
        "ECHAM": yhat_ECHAM,
        'M-FULL AMIP R1': yhat_CMIP5_AMIP_R1,
        'M-FULL RCP4.5 R1': yhat_CMIP5_RCP45_R1,
        'M-FULL RCP4.5 R2': yhat_CMIP5_RCP45_R2,
        'M-FULL RCP4.5 R3': yhat_CMIP5_RCP45_R3,
        'M-FULL RCP8.5 R1': yhat_CMIP5_RCP85_R1,
        'M-FULL RCP2.6 R1': yhat_CMIP5_RCP26_R1,        
        'M-79to00 AMIP R1': yhat_CMIP5_AMIP_R1,
        'M-79to00 RCP4.5 R1': yhat_m79to00_CMIP5_RCP45_R1,
        'M-79to00 RCP4.5 R2': yhat_m79to00_CMIP5_RCP45_R2,
        'M-79to00 RCP4.5 R3': yhat_m79to00_CMIP5_RCP45_R3,
        'M-79to00 RCP8.5 R1': yhat_m79to00_CMIP5_RCP85_R1,
        'M-79to00 RCP2.6 R1': yhat_m79to00_CMIP5_RCP26_R1,
        })
    store_pickle(stationname, 'predictions_' + variable, predictions)    
    store_csv(stationname, 'predictions_' + variable, predictions)

    # different labels and fill with specific missing value and re-write (files used by post-processor)    
    predictions2 = pd.DataFrame({
        'obs_raw': obs,
        'obs_anm': obs_anm,        
        'era_1979_2000': yhat_79to00,
        'era_2001_2015': yhat_01to15,
        'era_fit': yhat_ERA_full,
        "mpi_fit": yhat_ECHAM,
        'md_7915_amip_r1': yhat_CMIP5_AMIP_R1,
        'md_7915_rcp45_r1': yhat_CMIP5_RCP45_R1,
        'md_7915_rcp45_r2': yhat_CMIP5_RCP45_R2,
        'md_7915_rcp45_r3': yhat_CMIP5_RCP45_R3,
        'md_7915_rcp85_r1': yhat_CMIP5_RCP85_R1,
        'md_7915_rcp26_r1': yhat_CMIP5_RCP26_R1,        
        'md_7900_amip_r1': yhat_CMIP5_AMIP_R1,
        'md_7900_rcp45_r1': yhat_m79to00_CMIP5_RCP45_R1,
        'md_7900_rcp45_r2': yhat_m79to00_CMIP5_RCP45_R2,
        'md_7900_rcp45_r3': yhat_m79to00_CMIP5_RCP45_R3,
        'md_7900_rcp85_r1': yhat_m79to00_CMIP5_RCP85_R1,
        'md_7900_rcp26_r1': yhat_m79to00_CMIP5_RCP26_R1,
    })
    predictions2.fillna(-9999, inplace=True)    
    store_csv(stationname, 'predictions_fmt01_' + variable, predictions2)

    # SOME PLOTS
    # ==========

    #plot the predictions
    #apply_style(fontsize=16)
    #fig, ax = plt.subplots(1, 1, figsize=(12,6))
    era = predictions['ERA 1979-2000'].values
    era[np.isnan(era)] = predictions['ERA 2001-2015'].values[np.isnan(era)]
    era[np.isnan(era)] = predictions['ERA fit'].values[np.isnan(era)]
    era = era[~np.isnan(era)]
    echam = predictions["ECHAM"].dropna().values
    # predictions_plot(ax, obs, era, echam)
    # plt.legend()
    # if variable == "Temperature":
    #     plt.ylabel('Temperature anomalies in K')
    # else:
    #     plt.ylabel('Precipitation in mm/month')
    #     plt.savefig(os.path.join(figuredir, 'stations', variable + '_{:02d}_'.format(i) + filename + '_prediction.png'))
    # plt.savefig(os.path.join(figuredir, 'stations', variable + '_' + filename + '_prediction.png'))
    # plt.close()
    
    
    
#runing for all stations     
variable = "Precipitation" 
for i in range(20):
    predict_from_ML_station(i,variable)
#all_plots(variable)


#runing for one station
# predict_from_ML_station(2,variable)