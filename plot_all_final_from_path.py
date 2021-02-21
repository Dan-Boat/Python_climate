# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:20:07 2020
This script generates all the plots in the master thesis submited by Boateng Daniel on the 
topic: Prediction of precipitation response to different emission scenarios in the Ammer 
catchment
@author: Boateng
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator
import pandas as pd
import numpy as np
from plot_utils import *
from io_utils import *
from data import *
from regression_settings import *
import seaborn as sns

#generating plot for explained variance and score

def plot_exp_var_and_score(variable, path):
    
    df_scores = load_all_stations("score_era_" + variable, path)
    df_exp_var = load_all_stations("exp_var_" + variable, path)
    
    score_79to20 = df_scores["score 1979-2000"]
    is_fit_score = np.isnan(score_79to20)
    score_79to20[is_fit_score] = df_scores["fit score"].values[is_fit_score]
    #score_79to20.index = score_79to20.index.str.replace("_", " ")
    df_exp_var.index = df_exp_var.index.str.replace("_", " ")
    df_exp_var.columns = df_exp_var.columns.str.replace("_"," ")
    
    apply_style_2()
    fig, axes = plt.subplots(2, 1, sharex= True, figsize= (big_width, 0.7*big_width), constrained_layout = True)
    score_barplot(axes[0], score_79to20, is_fit_score)
    axes[0].set_ylim([-0.1, 1])
    exp_var_barplot(axes[1], df_exp_var)
    plt.xticks(rotation=45, ha = "right")
    
    #plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, "exp_var_score_" + variable + ".svg"),
                format = "svg")
    plt.close()
    plt.clf()

def plot_exp_var(variable, path):

    # get explained variance
    df_exp_var = load_all_stations('exp_var_' + variable, path)
    df_exp_var.index = df_exp_var.index.str.replace("_", " ")
    df_exp_var.columns = df_exp_var.columns.str.replace("_"," ")

    # plotting
    #apply_style()
    #plt.style.use('seaborn')
    #plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(1, 1, figsize=(big_width, 0.5*big_width))

    exp_var_barplot(ax, df_exp_var)    
    plt.xticks(rotation=45, ha='right')
       
    #ax.legend(bbox_to_anchor=(0.00, 1.02, 1., 0.102), loc=10, ncol=10, borderaxespad=0.)
#    ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=10, ncol=10, borderaxespad=0.)
    plt.tight_layout() # make sure nothing is cut off, replaces the experimental constrained_layout       

    plt.savefig(os.path.join(figuredir, 'exp_var_' + variable + '.svg'), format= "svg")    
    plt.close()
    plt.clf()  

def score_and_fit(variable, path):
    predict_patch = mpatches.Patch(color=score_colors['predict'], label='prediction score')
    fit_patch = mpatches.Patch(color=score_colors['fit'], label='fit score')
    
    df_scores = load_all_stations("score_era_" + variable, path)
    df_scores.index = df_scores.index.str.replace("_", " ")
    score_79to20 = df_scores["score 1979-2000"]
    fit_score   = df_scores["fit score"]
    apply_style_2()
    
    fig, axes = plt.subplots(2, 1, sharex= True, figsize= (big_width, 0.7*big_width), constrained_layout = False)
    score_79to20.plot(kind = "bar", ax= axes[0], color = black)
    axes[0].set_ylim([-0.1, 1])
    axes[0].set_ylabel('climate score')
    axes[0].legend(handles=[predict_patch], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    fit_score.plot(kind = "bar", ax = axes[1], color = orange)
    axes[1].set_ylim([-0.1,1])
    axes[1].set_ylabel('climate score')
    axes[1].legend(handles=[fit_patch], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45, ha = "right")
    
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, "score_fit_" + variable + ".svg"),
                format = "svg")
    plt.close()
    plt.clf()
def prediction_examples(variable, paths):
    example_station = 5
    apply_style_2()
    n = len(paths)
    text = "Precipitation anomalies [mm/month]"
    fig, axes = plt.subplots(n, 1, figsize= (textwidth, n*0.35*textwidth), sharex=True)
    fig.text(0.02, 0.5, text, va= "center", rotation= "vertical")
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.94, bottom=0.06, hspace=0.02)
    
    for i,idx in enumerate(paths):
        stationname = stationnames[example_station]
        df = load_csv(stationname, 'predictions_' + variable, paths[i])
        obs_an = df['obs anomalies']        
        echam = df['ECHAM'].dropna()
        era = df['ERA fit'].loc[echam.index]
        smooth_obs = obs_an.rolling(12, min_periods=1, win_type='hann', center=True).mean()
        smooth_echam = echam.rolling(12, min_periods=1, win_type='hann', center=True).mean()
        smooth_era = era.rolling(12, min_periods=1, win_type='hann', center=True).mean()

        axes[i].plot(smooth_era, '--', color=red, label='ERA')
        axes[i].plot(smooth_echam, '-.', color=blue, label='ECHAM')
        axes[i].plot(smooth_obs, '-', color=green, label='Station Observations')        
        axes[i].xaxis.set_major_locator(YearLocator(10))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[0].legend(bbox_to_anchor=(0.2, 1.02, 1., 0.102), loc=3, ncol=3, borderaxespad=0., frameon = True, fontsize = 8)
    axes[0].set_title("OLSForward", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[1].set_title("LASSO", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[2].set_title("MLP", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[3].set_title("SVR", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, 'prediction_examples_' + variable + '.svg'), format= "svg")
    plt.close()
    plt.clf()
    
    
    
def plot_smooth_examples(variable, path):
    example_stations = [19, 5, 9, 11]
    apply_style_2()
    n = len(example_stations)
    text = "Precipitation anomalies [mm/month]"
    fig, axes = plt.subplots(n, 1, figsize= (textwidth, n*0.35*textwidth), sharex=True)
    fig.text(0.02, 0.5, text, va= "center", rotation= "vertical")
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.94, bottom=0.06, hspace=0.02)
   
    for i, idx in enumerate(example_stations):
        stationname = stationnames[idx]
        df = load_csv(stationname, 'predictions_' + variable, path)
        obs_an = df['obs anomalies']        
        echam = df['ECHAM'].dropna()
        era = df['ERA fit'].loc[echam.index]

        smooth_obs = obs_an.rolling(12, min_periods=1, win_type='hann', center=True).mean()
        smooth_echam = echam.rolling(12, min_periods=1, win_type='hann', center=True).mean()
        smooth_era = era.rolling(12, min_periods=1, win_type='hann', center=True).mean()

        axes[i].plot(smooth_era, '--', color=red, label='ERA')
        axes[i].plot(smooth_echam, '-.', color=blue, label='ECHAM')
        axes[i].plot(smooth_obs, '-', color=green, label='Station Observations')        
        axes[i].xaxis.set_major_locator(YearLocator(10))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[0].legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=3, borderaxespad=0., frameon = True, fontsize = 8)
    plt.savefig(os.path.join(figuredir, 'examples_smooth_' + variable + '.svg'), format= "svg")
    plt.close()
    plt.clf()
    
def plot_high_frequency_examples(variable, paths):
    example_station = 5
    apply_style_2()
    n = len(paths)
    text = "Precipitation anomalies [mm/month]"
    fig, axes = plt.subplots(n, 1, figsize= (textwidth, n*0.35*textwidth), sharex=True)
    fig.text(0.02, 0.5, text, va= "center", rotation= "vertical")
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.94, bottom=0.06, hspace=0.02)
    
    for i,idx in enumerate(paths):
        stationname = stationnames[example_station]
        df = load_csv(stationname, 'predictions_' + variable, paths[i])
        obs_an = df['obs anomalies']        
        echam = df['ECHAM'].dropna()
        era = df['ERA fit'].loc[echam.index]
        smooth_obs = obs_an.rolling(12, min_periods=1, win_type='hann', center=True).mean()
        smooth_echam = echam.rolling(12, min_periods=1, win_type='hann', center=True).mean()
        smooth_era = era.rolling(12, min_periods=1, win_type='hann', center=True).mean()
   
        obs = obs_an - smooth_obs
        era_high = era - smooth_era
        echam_high = echam - smooth_echam
        axes[i].plot(era_high, '--', color=red, label='ERA', linewidth=1)
        axes[i].plot(echam_high, '-.', color=blue, label='ECHAM')
        axes[i].plot(obs, '-', color=green, label='Station Observations')        
        axes[i].xaxis.set_major_locator(YearLocator(2))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
    axes[0].legend(bbox_to_anchor=(0.2, 1.02, 1., 0.102), loc=3, ncol=3, borderaxespad=0., frameon = True, fontsize = 8)
    axes[0].set_title("OLSForward", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[1].set_title("LASSO", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[2].set_title("MLP", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[3].set_title("SVR", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, 'high_prediction_examples_' + variable + '.svg'), format= "svg")
    plt.close()
    plt.clf()
        

def plot_complete_smooth_stations(variable, path):
    #apply_style_2()
    p = 1
    example_stations = [19, 5, 9, 11]
    apply_style_2()
    n = len(example_stations)
    text = "Precipitation anomalies [mm/month]"
    fig, axes = plt.subplots(p, 1, figsize= (big_width, 0.7*big_width), constrained_layout = True)
    fig.text(0.01, 0.5, text, va= "center", rotation= "vertical", fontweight = "bold")
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.94, bottom=0.06, hspace=0.02)
    for i, idx in enumerate(example_stations):
        stationname = stationnames[idx]
        df = load_csv(stationname, 'predictions_' + variable, path)
        obs_an = df['obs anomalies']
        obs_an = obs_an[from01to15].dropna()        
        cmip5_rcp26 = df['M-79to00 RCP2.6 R1'].dropna()
        cmip5_rcp45 = df['M-79to00 RCP4.5 R1'].dropna()
        cmip5_rcp85 = df['M-79to00 RCP8.5 R1'].dropna()
    

        smooth_obs = obs_an.rolling(12*3, min_periods=1, win_type='hann', center=True).mean()
        smooth_rcp26 = cmip5_rcp26.rolling(12*5, min_periods=1, win_type='hann', center=True).mean()
        smooth_rcp45 = cmip5_rcp45.rolling(12*5, min_periods=1, win_type='hann', center=True).mean()
        smooth_rcp85 = cmip5_rcp85.rolling(12*5, min_periods=1, win_type='hann', center=True).mean()
        

        axes.plot(smooth_rcp26, '--', color=black, label='RCP 2.6')
        axes.plot(smooth_rcp45, '--', color=red, label='RCP 4.5')
        axes.plot(smooth_rcp85, '--', color=blue, label='RCP 8.5')
        axes.plot(smooth_obs, '-', color=green, label='Station Observations')        
        axes.xaxis.set_major_locator(YearLocator(20))
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if idx == 19:
            
            axes.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=1, ncol=3, borderaxespad=0., frameon = True, fontsize = 10)
        else:
            None
    plt.tight_layout()
    plt.savefig(os.path.join(figuredir, 'complete_smooth_' + variable + '.svg'), format= "svg")
    plt.close()
    plt.clf()
    
def long_term_average(variable, path, datarange, timespan):
    n = 20 #stations
    df_obj = pd.DataFrame(index = stationnames, columns = ["RCP26", "RCP45", "RCP85"])
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_" + variable, path)
        obs_an = df['obs anomalies'].dropna().resample("Q-NOV").mean()
        cmip5_rcp26 = df['M-79to00 RCP2.6 R1'].dropna().resample("Q-NOV").mean()[datarange].mean()
        cmip5_rcp45 = df['M-79to00 RCP4.5 R1'].dropna().resample("Q-NOV").mean()[datarange].mean()
        cmip5_rcp85 = df['M-79to00 RCP8.5 R1'].dropna().resample("Q-NOV").mean()[datarange].mean()
        
        
        df_obj.loc[stationname].RCP26 = cmip5_rcp26
        df_obj.loc[stationname].RCP45 = cmip5_rcp45
        df_obj.loc[stationname].RCP85 = cmip5_rcp85
    df_obj.index = df_obj.index.str.replace("_", " ")    
    df_obj = df_obj.T
    df_obj.index = df_obj.index.str.replace("RCP26","RCP2.6")
    df_obj.index = df_obj.index.str.replace("RCP45","RCP4.5")
    df_obj.index = df_obj.index.str.replace("RCP85","RCP8.5")
    df = df_obj.astype(float)
        
    
    fig, axes = plt.subplots(1, 1, figsize= (big_width, 0.7*big_width), constrained_layout = False)
    
    #cbar_ax = fig.add_axes([.905, .3, .05, .3])
    sns.heatmap(ax = axes, data= df, cmap= "RdBu", center = 0, cbar_kws= {"label": "mm/month", "shrink":.50}, square= True, vmin = -10, vmax = 10)
    plt.title("Precipitation Anomalies (" + timespan + ")", fontsize=10)
    
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "long_term_mean_" + variable + "_" +timespan +".svg"),
                format = "svg")
    plt.close()
    plt.clf()



def long_term_seasonal_mean(variable, path, datarange, timespan):
    n = 20 #stations
    df_26 = pd.DataFrame(index = stationnames, columns = ["DJF", "MAM", "JJA", "SON"])
    df_45 = pd.DataFrame(index = stationnames, columns = ["DJF", "MAM", "JJA", "SON"])
    df_85 = pd.DataFrame(index = stationnames, columns = ["DJF", "MAM", "JJA", "SON"])
    df_obs = pd.DataFrame(index = stationnames, columns = ["DJF", "MAM", "JJA", "SON"])
    
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_" + variable, path)
        obs_an = df['obs anomalies'].dropna().resample("Q-NOV").mean()
        cmip5_rcp26 = df['M-79to00 RCP2.6 R1'].dropna().resample("Q-NOV").mean()[datarange]
        cmip5_rcp45 = df['M-79to00 RCP4.5 R1'].dropna().resample("Q-NOV").mean()[datarange]
        cmip5_rcp85 = df['M-79to00 RCP8.5 R1'].dropna().resample("Q-NOV").mean()[datarange]
        
        winter_26 = cmip5_rcp26[cmip5_rcp26.index.quarter == 1].mean()
        spring_26 = cmip5_rcp26[cmip5_rcp26.index.quarter == 2].mean()
        summer_26 = cmip5_rcp26[cmip5_rcp26.index.quarter == 3].mean()
        autumn_26 = cmip5_rcp26[cmip5_rcp26.index.quarter == 4].mean()
        
        winter_45 = cmip5_rcp45[cmip5_rcp45.index.quarter == 1].mean()
        spring_45 = cmip5_rcp45[cmip5_rcp45.index.quarter == 2].mean()
        summer_45 = cmip5_rcp45[cmip5_rcp45.index.quarter == 3].mean()
        autumn_45 = cmip5_rcp45[cmip5_rcp45.index.quarter == 4].mean()
        
        winter_85 = cmip5_rcp85[cmip5_rcp85.index.quarter == 1].mean()
        spring_85 = cmip5_rcp85[cmip5_rcp85.index.quarter == 2].mean()
        summer_85 = cmip5_rcp85[cmip5_rcp85.index.quarter == 3].mean()
        autumn_85 = cmip5_rcp85[cmip5_rcp85.index.quarter == 4].mean()
        
        
        df_26.loc[stationname].DJF = winter_26
        df_26.loc[stationname].MAM = spring_26
        df_26.loc[stationname].JJA = summer_26
        df_26.loc[stationname].SON = autumn_26
        
        df_45.loc[stationname].DJF = winter_45
        df_45.loc[stationname].MAM = spring_45
        df_45.loc[stationname].JJA = summer_45
        df_45.loc[stationname].SON = autumn_45
        
        df_85.loc[stationname].DJF = winter_85
        df_85.loc[stationname].MAM = spring_85
        df_85.loc[stationname].JJA = summer_85
        df_85.loc[stationname].SON = autumn_85

    df_26.index = df_26.index.str.replace("_", " ")
    df_26 = df_26.T
    df_26 = df_26.astype(float)
    
    df_45.index = df_45.index.str.replace("_", " ")
    df_45 = df_45.T
    df_45 = df_45.astype(float)
    
    df_85.index = df_85.index.str.replace("_", " ")
    df_85 = df_85.T
    df_85 = df_85.astype(float)
        
 #TO DO : Try to create a plot using heatmap or pcolor to display the long term average 
# Try to fix the seasonal challenges    
      
    fig, axes = plt.subplots(3, 1,figsize= (textwidth, 3*0.35*textwidth), constrained_layout = True, sharex=True, sharey = True)
    plt.subplots_adjust(left=0.02, right=1-0.02, top=0.94, bottom=0.45, hspace=0.25)
    
    cbar_ax = fig.add_axes([.89, .5, .02, .35])
    sns.heatmap(ax = axes[0], data= df_26, cmap= "RdBu", center = 0, square= True, cbar= False, vmin = -20, vmax = 20)
    sns.heatmap(ax = axes[1], data= df_45, cmap= "RdBu", center = 0, square= True, cbar = False, vmin = -20, vmax = 20)
    sns.heatmap(ax = axes[2], data= df_85, cmap= "RdBu", center = 0, cbar_kws= {"label": "mm/month", "shrink":.50}, square= True, cbar_ax= cbar_ax, vmin = -20, vmax = 20)
    plt.suptitle("Precipitation Anomalies (" + timespan + ")", fontsize=10)
    
    axes[0].set_title("RCP 2.6", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[1].set_title("RCP 4.5", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    axes[2].set_title("RCP 8.5", loc = "left", pad = 0.1, fontsize = 8, fontweight = "bold")
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "long_term_seasonal_" + variable + "_" + timespan + ".svg"),
                format = "svg")
    
    plt.close()
    plt.clf()
    
def observed_mean(variable, path, datarange, timespan):
    n = 20 #stations
    df_obs = pd.DataFrame(index = stationnames, columns = ["Mean", "DJF", "MAM", "JJA", "SON"])
    
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_" + variable, path)
        obs = df['obs'].resample("Q-NOV").mean()[datarange]
        winter = obs[obs.index.quarter == 1].mean()
        spring = obs[obs.index.quarter == 2].mean()
        summer = obs[obs.index.quarter == 3].mean()
        autumn = obs[obs.index.quarter == 4].mean()
        long_term_mean = obs.mean()
        
        df_obs.loc[stationname].DJF = winter
        df_obs.loc[stationname].MAM = spring
        df_obs.loc[stationname].JJA = summer
        df_obs.loc[stationname].SON = autumn
        df_obs.loc[stationname].Mean= long_term_mean
        
    df_obs.index = df_obs.index.str.replace("_", " ")
    df_obs = df_obs.T
    df_obs = df_obs.astype(float)
    
    fig, axes = plt.subplots(1, 1, figsize= (big_width, 0.9*big_width), constrained_layout = False)
    
    #cbar_ax = fig.add_axes([.905, .3, .05, .3])
    sns.heatmap(ax = axes, data= df_obs, cmap= "Blues", cbar_kws= {"label": "mm/month", "shrink":.50}, square= True)
    plt.title("Total Monthly Precipitation (" + timespan + ")", fontsize=10)
    
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "observed_total_prep" + variable +".svg"),
                format = "svg")
    plt.close()
    plt.clf()

def observed_anomaly_mean(variable, path, datarange, timespan):
    n = 20 #stations
    df_obs = pd.DataFrame(index = stationnames, columns = ["Mean", "DJF", "MAM", "JJA", "SON"])
    
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_" + variable, path)
        obs = df['obs anomalies'].resample("Q-NOV").mean()[datarange]
        winter = obs[obs.index.quarter == 1].mean()
        spring = obs[obs.index.quarter == 2].mean()
        summer = obs[obs.index.quarter == 3].mean()
        autumn = obs[obs.index.quarter == 4].mean()
        long_term_mean = obs.mean()
        
        df_obs.loc[stationname].DJF = winter
        df_obs.loc[stationname].MAM = spring
        df_obs.loc[stationname].JJA = summer
        df_obs.loc[stationname].SON = autumn
        df_obs.loc[stationname].Mean= long_term_mean
        
    df_obs.index = df_obs.index.str.replace("_", " ")
    df_obs = df_obs.T
    df_obs = df_obs.astype(float)
    
    fig, axes = plt.subplots(1, 1, figsize= (big_width, 0.9*big_width), constrained_layout = False)
    
    #cbar_ax = fig.add_axes([.905, .3, .05, .3])
    sns.heatmap(ax = axes, data= df_obs, cmap= "RdBu", cbar_kws= {"label": "mm/month", "shrink":.50}, square= True)
    plt.title("Precipitation Anomalies (" + timespan + ")", fontsize=10)
    
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "observed_prep_anomaly" + variable +".svg"),
                format = "svg")
    plt.close()
    plt.clf()

        
        
def plot_data_available(variable, path, datarange, timespan):
    n = 20
    df_obs = pd.DataFrame(index = datarange, columns = stationnames)
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_" + variable, path)
        obs = df['obs'].resample("Q-NOV").mean()[datarange]
        
        df_obs[stationname] = obs
       
    df_obs = df_obs.T
    df_obs.index = df_obs.index.str.replace("_", " ")
    df_obs = df_obs.astype(float)
    df_obs[~df_obs.isnull()] = 1
    df_obs[df_obs.isnull()] = 0
    df_obs.columns = df_obs.columns.strftime("%Y")
    
    fig, axes = plt.subplots(1, 1, figsize= (big_width, 0.7*big_width), constrained_layout = True)
    
    sns.heatmap(ax = axes, data= df_obs, cmap= "Blues", cbar = False, linewidth = 0.3)
    
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "weather_station_overview" + variable +".svg"),
                format = "svg", dpi = 1200)
    plt.close()
    plt.clf()

        
        
        
    
#test section
variable = "Precipitation"
# paths = [OLSForward_dir, lasso_dir, MLP_dir, SVR_dir]
# prediction_examples(variable, paths) 
# plot_high_frequency_examples(variable, paths)
#plot_exp_var_and_score(variable, lasso_dir)
#score_and_fit(variable, SVR_dir) 
#long_term_average(variable, MLP_dir, from80to21, "2080-2100")
observed_mean(variable, OLSForward_dir, from79to14, "1979-2014")
      
    
 

        
#using above functions to generate required plots 
 #MLP_dir, SVR_dir, OLSForward_dir, lasso_dir     
variable = "Precipitation"  
def plot_station(variable):
    plot_data_available(variable, OLSForward_dir, from79to14, "1979-2014")
    observed_mean(variable, OLSForward_dir, from79to14, "1979-2014")
    observed_anomaly_mean(variable, OLSForward_dir, from79to14, "1979-2014")

observed_mean(variable, OLSForward_dir, from79to14, "1979-2014")
#score_and_fit(variable, MLP_dir)

def plot_lasso(variable):
    plot_exp_var_and_score(variable, lasso_dir)
    plot_exp_var(variable, lasso_dir)
    #score_and_fit(variable, lasso_dir)
    plot_smooth_examples(variable, lasso_dir)
    plot_complete_smooth_stations(variable, lasso_dir)
    long_term_average(variable, lasso_dir, from20to40, "2020-2040")
    long_term_average(variable, lasso_dir, from40to60, "2040-2060")
    long_term_average(variable, lasso_dir, from60to80, "2060-2080")
    long_term_average(variable, lasso_dir, from80to21, "2080-2100")
    long_term_seasonal_mean(variable, lasso_dir, from20to40, "2020-2040")
    long_term_seasonal_mean(variable, lasso_dir, from40to60, "2040-2060")
    long_term_seasonal_mean(variable, lasso_dir, from60to80, "2060-2080")
    long_term_seasonal_mean(variable, lasso_dir, from80to21, "2080-2100")
    
def plot_OLSForward(variable):
    plot_exp_var_and_score(variable, OLSForward_dir)
    plot_exp_var(variable, OLSForward_dir)
    #score_and_fit(variable, OLSForward_dir)
    plot_smooth_examples(variable, OLSForward_dir)
    plot_complete_smooth_stations(variable, OLSForward_dir)
    long_term_average(variable, OLSForward_dir, from20to40, "2020-2040")
    long_term_average(variable, OLSForward_dir, from40to60, "2040-2060")
    long_term_average(variable, OLSForward_dir, from60to80, "2060-2080")
    long_term_average(variable, OLSForward_dir, from80to21, "2080-2100")
    long_term_seasonal_mean(variable, OLSForward_dir, from20to40, "2020-2040")
    long_term_seasonal_mean(variable, OLSForward_dir, from40to60, "2040-2060")
    long_term_seasonal_mean(variable, OLSForward_dir, from60to80, "2060-2080")
    long_term_seasonal_mean(variable, OLSForward_dir, from80to21, "2080-2100")
    
def plot_MLP(variable):
    #score_and_fit(variable, MLP_dir)
    plot_smooth_examples(variable, MLP_dir)
    plot_complete_smooth_stations(variable, MLP_dir)
    long_term_average(variable, MLP_dir, from20to40, "2020-2040")
    long_term_average(variable, MLP_dir, from40to60, "2040-2060")
    long_term_average(variable, MLP_dir, from60to80, "2060-2080")
    long_term_average(variable, MLP_dir, from80to21, "2080-2100")
    long_term_seasonal_mean(variable, MLP_dir, from20to40, "2020-2040")
    long_term_seasonal_mean(variable, MLP_dir, from40to60, "2040-2060")
    long_term_seasonal_mean(variable, MLP_dir, from60to80, "2060-2080")
    long_term_seasonal_mean(variable, MLP_dir, from80to21, "2080-2100")
    
def plot_SVR(variable):
    #score_and_fit(variable, SVR_dir)
    plot_smooth_examples(variable, SVR_dir)
    plot_complete_smooth_stations(variable, SVR_dir)
    long_term_average(variable, SVR_dir, from20to40, "2020-2040")
    long_term_average(variable, SVR_dir, from40to60, "2040-2060")
    long_term_average(variable, SVR_dir, from60to80, "2060-2080")
    long_term_average(variable, SVR_dir, from80to21, "2080-2100")
    long_term_seasonal_mean(variable, SVR_dir, from20to40, "2020-2040")
    long_term_seasonal_mean(variable, SVR_dir, from40to60, "2040-2060")
    long_term_seasonal_mean(variable, SVR_dir, from60to80, "2060-2080")
    long_term_seasonal_mean(variable, SVR_dir, from80to21, "2080-2100")
    





    

    
    
    
