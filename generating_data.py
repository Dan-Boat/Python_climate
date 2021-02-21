# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:30:27 2020
This script depends on the following python packages (which can be installed 
                                                      using pip install "name of package"):
    xarray
    os
    pandas
    matplotlib
    cartopy
    metpy   
@author: Boateng
"""

#Importing modules
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


#defining path of datasets directory and path to store figures
# THIS MUST BE CHANGED TO PATH ON THE SERVER!!!

era_datadir     = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/era"
cru_pre_dir     = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/CRU/pre"
cru_t2m_dir     = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/CRU/tmp"
echam_pre_industrial = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/GCM-echam/Pre_industrial"
echam_present_day = "C:/Users/Boateng/Desktop/education/AEG stuff/Master_Thesis/Modules/GCM-echam/present_day"
figuredir       = "C:/Users/Boateng/Desktop/education/Hiwi/Generating_plots"


#class for reading data into memory when its needed 
class Dataset:
    def __init__(self, name, variables):
        #name of dataset
        self.name = name
        #variable is dictionary with variable name as keys and path of dataset as values
        self.variables = variables
        self.data = {}

    def get(self, var):
        try:
            return self.data[var]
        except KeyError:
            # we have to load it first
            self.data[var] = xr.open_dataarray(self.variables[var])
            # returns xr.DataArray
            return self.data[var]
        
        
#functions for calculating annual means, monthly means difference, and annual means difference


def CRU_annual_means(dataset, dataname):
    """
    This function computes the annual means of the CRU dataset from 1901 to 2012
    """
    data = dataset.get(dataname).sel(time = slice("1901-01-01", "2012-12-31"))
    data_annual_means = data.groupby("time.year").mean(dim= "time")
    data_annual_avg = data_annual_means.mean("year")
    return data_annual_avg

def CRU_annual_diff(dataset, dataname):
    """
    This function computes the annual means difference of the CRU datasets between the 
    annual means of 1901-1911 and 2002-2012
    """
    data1 = dataset.get(dataname).sel(time = slice("1901-01-01", "1911-12-31"))
    data_annual_means1 = data1.groupby("time.year").mean(dim= "time")
    data_annual_avg1 = data_annual_means1.mean("year")
    
    data2 = dataset.get(dataname).sel(time = slice("2002-01-01", "2012-12-31"))
    data_annual_means2 = data2.groupby("time.year").mean(dim= "time")
    data_annual_avg2 = data_annual_means2.mean("year")
    diff = data_annual_avg2 - data_annual_avg1
    return diff

def CRU_monthly_diff(dataset, dataname):
    """
    This function computes the monthly means difference of the CRU datasets between the 
    monthly means of 1901-1911 and 2002-2012
    """
    data1 = dataset.get(dataname).sel(time = slice("1901-01-01", "1911-12-31"))
    data_monthly_means1 = data1.groupby("time.month").mean(dim= "time")
    
    data2 = dataset.get(dataname).sel(time = slice("2002-01-01", "2012-12-31"))
    data_monthly_means2 = data2.groupby("time.month").mean(dim= "time")
   
    diff = data_monthly_means2 - data_monthly_means1
    return diff


#annual means 
def ERA_annual_means(dataset, dataname):
    """
    This function computes the annual means of the ERA_Interim dataset from 1979 to 2015
    """
    data = dataset.get(dataname).sel(time = pd.date_range(start='1979-01-01', end='2015-12-31', freq='MS'))
    #data = dataset.get(dataname).sel(time= slice("1979-01-01", "2015-12-31"))
    if  dataname == "t2m":
        data.metpy.convert_units("degC") 
    else:
        None
    data_annual_means= data.groupby("time.year").mean(dim= "time")
    data_annual_avg = data_annual_means.mean("year")
    
    return data_annual_avg

def ERA_annual_diff(dataset, dataname):
    """
    This function computes the annual means difference of the ERA_Interim datasets between the 
    monthly means of 1979-1989 and 1995-2015
    """
    data1 = dataset.get(dataname).sel(time = slice("1979-01-01", "1989-12-31"))
    if  dataname == "t2m":
        data1.metpy.convert_units("degC") 
    else:
        None
    data_annual_means1 = data1.groupby("time.year").mean(dim= "time")
    data_annual_avg1 = data_annual_means1.mean("year")
    
    data2 = dataset.get(dataname).sel(time = slice("2005-01-01", "2015-12-31"))
    if  dataname == "t2m":
        data2.metpy.convert_units("degC") 
    else:
        None
    data_annual_means2 = data2.groupby("time.year").mean(dim= "time")
    data_annual_avg2 = data_annual_means2.mean("year")
    diff = data_annual_avg2 - data_annual_avg1
    return diff

def ERA_monthly_diff(dataset, dataname):
    """
    This function computes the monthly means difference of the ERA_Interim datasets between the 
    monthly means of 1979-1989 and 1995-2015
    """
    data1 = dataset.get(dataname).sel(time = slice("1979-01-01", "1989-12-31"))
    if  dataname == "t2m":
        data1.metpy.convert_units("degC") 
    else:
        None
    data_monthly_means1 = data1.groupby("time.month").mean(dim= "time")
    
    data2 = dataset.get(dataname).sel(time = slice("2005-01-01", "2015-12-31"))
    if  dataname == "t2m":
        data2.metpy.convert_units("degC") 
    else:
        None
    data_monthly_means2 = data2.groupby("time.month").mean(dim= "time")
   
    diff = data_monthly_means2 - data_monthly_means1
    return diff


#reading data using the Dataset Class and the directory of the dataset.
#reading data 
ERAData = Dataset("ERA",
                  {"t2m": os.path.join(era_datadir, 't2m_monthly.nc'),
                    "tp": os.path.join(era_datadir, 'tp_monthly.nc')
                    })
CRUData = Dataset("CRU", {
                          "t2m": os.path.join(cru_t2m_dir,"cru_ts3.21.1901.2012.tmp.dat.nc"),
                          "tp": os.path.join(cru_pre_dir, "cru_ts3.21.1901.2012.pre.dat.nc")
                    }) 


#calculating the annual means of CRU and ERA-Interim for precipitation (tp) and temperature (t2m) 
t2m_era_annual_means = ERA_annual_means(ERAData, "t2m")
tp_era_annual_means  = ERA_annual_means(ERAData, "tp")
t2m_cru_annual_means = CRU_annual_means(CRUData, "t2m")
tp_cru_annual_means  = CRU_annual_means(CRUData, "tp")


#mean annual difference
t2m_era_annual_mean_diff = ERA_annual_diff(ERAData, "t2m")
tp_era_annual_mean_diff  = ERA_annual_diff(ERAData, "tp")
t2m_cru_annual_mean_diff = CRU_annual_diff(CRUData, "t2m")
tp_cru_annual_mean_diff  = CRU_annual_diff(CRUData, "tp") 

#mean monthly difference
t2m_era_monthly_mean_diff = ERA_monthly_diff(ERAData, "t2m")
tp_era_monthly_mean_diff  = ERA_monthly_diff(ERAData, "tp")
t2m_cru_monthly_mean_diff = CRU_monthly_diff(CRUData, "t2m")
tp_cru_monthly_mean_diff  = CRU_monthly_diff(CRUData, "tp")




