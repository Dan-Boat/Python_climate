# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:47:29 2020

@author: Boateng
"""
import xarray as xr
import os
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER 
from cartopy.util import add_cyclic_point

import metpy.calc as mpcalc
from metpy.units import units 
import matplotlib.colors as colors
import calendar




#reading files 
 
#In case you are runing this script on the server, kindly change these directories!!!!
echam_pre_industrial = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/GCM-echam/Pre_industrial"
echam_present_day = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/GCM-echam/present_day"

#The data are already preprocessed into long term means and grouped into monthly means as well
data_pre_industrial = xr.open_dataset(os.path.join(echam_pre_industrial, "1003_1017_1m_mlterm.nc")) #reference year of 1850
data_present_day = xr.open_dataset(os.path.join(echam_present_day, "1980_2000_1m_mlterm.nc"))

# variables: long names 
# {aprl: large scale precipitation, aprc: convective precipitation, temp2: 2m temperature}
# Total Precipitation is calculated by adding aprc and aprl


#extracting variabels
#pre_industrial (long term monthly mean)
temp2_pre_industrial = data_pre_industrial["temp2"] - 273.15  #degC
                
prec_pre_industrial = 86400*(data_pre_industrial["aprl"] + data_pre_industrial["aprc"])  #mm/day

#present day (long-term monthly means)
temp2_present_day = data_present_day["temp2"] - 273.15
prec_present_day = 86400*(data_present_day["aprl"] + data_present_day["aprc"])

#long term annual mean
temp2_pre_industrial_annual_mean = temp2_pre_industrial.mean(dim = "time")
prec_pre_industrial_annual_mean = prec_pre_industrial.mean(dim = "time")

temp2_present_day_annual_mean = temp2_present_day.mean(dim = "time")
prec_present_day_annual_mean = prec_present_day.mean(dim = "time")

#annual difference between present day and pre industrial 
temp2_annual_diff = temp2_present_day_annual_mean - temp2_pre_industrial_annual_mean
prec_annual_diff= prec_present_day_annual_mean - prec_pre_industrial_annual_mean


#monthly difference
#synthetic time just for calculation
temp2_pre_industrial["time"] = temp2_present_day.time
prec_pre_industrial["time"] = prec_present_day.time

temp2_monthly_diff = temp2_present_day - temp2_pre_industrial
prec_monthly_diff  = prec_present_day - prec_pre_industrial