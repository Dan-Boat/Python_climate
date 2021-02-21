# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:20:25 2020

@author: Boateng
"""
#This code generates the plots for simulated predictors until end of 21st century 
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


def plot_CMIP_variable(i, variable, predictor):
    stationname = stationnames[i]
    filename = stationname.replace(" ", "_")
    ws = read_station_from_csv(os.path.join(station_datadir, filename + ".csv"))
    anomalies = variable = "Precipitation"
    ws.set_predictors(variable, predictors, predictordir, radius)
    predictor_data_26 = ws._get_predictor_data(variable, fullCMIP5, CMIP5_RCP26_R1,fit=True)
    predictor_data_45 = ws._get_predictor_data(variable, fullCMIP5, CMIP5_RCP45_R1,fit=True)
    predictor_data_85 = ws._get_predictor_data(variable, fullCMIP5, CMIP5_RCP85_R1,fit=True)
    
    predictor_26 = predictor_data_26[predictor]
    predictor_45 = predictor_data_45[predictor]
    predictor_85 = predictor_data_85[predictor]
    
    apply_style_2()
    fig, axes = plt.subplots(3, 1,figsize= (textwidth, 3*0.35*textwidth), constrained_layout = True, sharex=True, sharey = True)
    plt.subplots_adjust(left=0.02, right=1-0.02, top=0.94, bottom=0.45, hspace=0.25)
    predictor_26 = predictor_26.rolling(5*12, min_periods=1, win_type= "hann", center=True).mean()
    predictor_45 = predictor_45.rolling(5*12, min_periods=1, win_type= "hann", center=True).mean()
    predictor_85 = predictor_85.rolling(5*12, min_periods=1, win_type= "hann", center=True).mean()
    
    predictor_26.plot(kind ="line", ax = axes[0], color=black, label='RCP 2.6')
    predictor_45.plot(kind ="line", ax = axes[1], color=red, label='RCP 4.5')
    predictor_85.plot(kind ="line", ax = axes[2], color=blue, label='RCP 8.5')
    
    axes[0].set_title("RCP 2.6", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[1].set_title("RCP 4.5", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[2].set_title("RCP 8.5", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    
    
    axes[0].axhline(y=0, linestyle = "--", color= black)
    axes[1].axhline(y=0, linestyle = "--", color= black)
    axes[2].axhline(y=0, linestyle = "--", color= black)
    if predictor == "PC1_NAO":
        fig.suptitle("NAO")
    else:
        
        fig.suptitle(predictor)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, 'cmip_' + predictor + '.svg'), format= "svg")
    plt.close()
    plt.clf()
    
    
   



variable = "Precipitation"
plot_CMIP_variable(5, variable, "PC1_NAO")
#same stations used for plotting prediction examples