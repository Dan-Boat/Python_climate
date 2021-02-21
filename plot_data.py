# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:48:58 2020

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
from cartopy.mpl.ticker import (LatitudeLocator, LongitudeLocator) 
from cartopy.util import add_cyclic_point

import metpy.calc as mpcalc
from metpy.units import units 
import matplotlib.colors as colors
import calendar

#importing dataset required for plotting 
from generating_data import *
from generating_data_from_GCM import *

def plot_background(ax):
    ax.set_global()
    ax.coastlines(resolution = "50m")
    #adding countries borders
    ax.add_feature(cfeature.BORDERS, color= "black", alpha= 1, linewidth= 0.5)
    #select the cordinates of required map view (eg. Europe, Asia, Africa and others)
    ax.set_extent([-10, 23.5, 35, 62], ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,linewidth= 1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 15, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"color": "black", "size": 15, "weight": "bold"}
    
    return ax
    
#function for ploting annual means of temperature and precipitation
def plot_annual_mean(variable, dataset, datatype, vmin, vmax):
    """
    this function generates the plot for long term annual average
    Variable = str  eg. Temperature or Precipitation
    dataset = xr.Dataset
    datatype = str  eg. ERA or CRU
    vmin = float (minimun value of the data)
    vmax = float (maximum value of the data)

    """
    #defining projection for the basemap
    projection = ccrs.Orthographic(central_latitude= 32)
    fig, ax = plt.subplots(1, 1, sharex = False, figsize = (15, 13), subplot_kw={"projection": projection})
    # Interpolating between the 0 and 360 latitude to prevent display of white line
    if hasattr(dataset, "longitude") & hasattr(dataset, "latitude"):
        data, lon = add_cyclic_point(dataset, coord = dataset.longitude)
        lat = dataset.latitude
    else:
        data, lon = add_cyclic_point(dataset, coord = dataset.lon)
        lat = dataset.lat

    plot_background(ax)

    if variable == "Temperature":
        
        img1 = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), cmap= plt.cm.RdBu_r,
                    vmin = vmin, vmax= vmax, levels= 37)
    
        num_levels = 37  # represents the color intervals (can be manually adjusted)
        midpoint = 0
        levels = np.linspace(vmin, vmax, num_levels)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colors_ = plt.cm.RdBu_r(vals)
        cmap, norm = mpl.colors.from_levels_and_colors(levels, colors_)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        cb = plt.colorbar(mappable = m, cax=None, ax= ax,
                              orientation= "vertical", pad= 0.1, extend = "both")
        cb.set_label(label="Temperature [°C]", size = 25, weight = "bold")
        cb.ax.tick_params(labelsize = 20)
        cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
        cb.dividers.set_linewidth(2)
        
    elif variable == "Precipitation":
        img1 = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), cmap=plt.cm.Blues, 
                    vmin = vmin, vmax= vmax, levels = 37)
        if datatype == "ERA":
            bounds = np.linspace(vmin, vmax, 37)
            
        elif datatype == "GCM_present_day" or datatype == "GCM_pre_industrial":
            bounds = np.linspace(vmin, vmax, 37)
            
        else:
            bounds = np.linspace(vmin,vmax, 37)
        cmap=plt.cm.Blues
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # # using DiverginNorm sets the midpoint to zero but the discrets colors want show
        # #norm = mpl.colors.DivergingNorm(vmax= vmax, vmin= vmin, vcenter=0)
        # num_levels = 20  # represents the color intervals (can be manually adjusted)
        # midpoint = 0
        # levels = np.linspace(vmin, vmax, num_levels)
        # midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        # vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        # colors_ = plt.cm.Blues(vals)
        # cmap, norm = mpl.colors.from_levels_and_colors(levels, colors_)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        cb = plt.colorbar(mappable=m, cax=None, ax= ax,
                              orientation= "vertical", pad= 0.1)
        if datatype == "ERA":
            
            cb.set_label(label="Precipitation [mm/day]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        elif datatype == "GCM_present_day" or datatype == "GCM_pre_industrial":
            
            cb.set_label(label="Precipitation [mm/day]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        else:
            cb.set_label(label="Precipitation [mm]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
    else:
        print("Define variable")
    if datatype == "ERA":
    
        plt.title('Long term (1979-2015) annual mean [ERA-Interim]' , fontsize=20, weight = "bold")
        
    elif datatype == "GCM_present_day":
    
        plt.title('Long term (Present Day) annual mean [GCM]' , fontsize=20, weight = "bold")
        
    elif datatype == "GCM_pre_industrial":
    
        plt.title('Long term (Pre Industrial Age) annual mean [GCM]' , fontsize=20, weight = "bold")
    
    else:
          plt.title('Long term (1901-2012) annual mean [CRU]' , fontsize=20, weight = "bold")
    
    plt.savefig(os.path.join(figuredir, "annual_mean_" + variable +"_"+ datatype + '.png'))    
    plt.close()
    plt.clf() 
    
#function for ploting annual mean difference of temperature and precipitation
def plot_annual_mean_diff(variable, dataset, datatype, vmin, vmax):
    """
    this function generates the plot for long term annual average difference
    Variable = str  eg. Temperature or Precipitation
    dataset = xr.Dataset
    datatype = str  eg. ERA or CRU
    vmin = float (minimun value of the data)
    vmax = float (maximum value of the data)
    """
    fig, ax = plt.subplots(1, 1, sharex = False, figsize = (15, 13))
    # Interpolating between the 0 and 360 latitude to prevent display of white line
    if datatype == "ERA":
        data, lon = add_cyclic_point(dataset, coord = dataset.longitude)
        lat = dataset.latitude
    else:
        data, lon = add_cyclic_point(dataset, coord = dataset.lon)
        lat = dataset.lat
    #defining projection for the basemap
    projection = ccrs.Orthographic(central_latitude= 32)
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.coastlines(resolution = "50m")
    ax.add_feature(cfeature.BORDERS, color= "black", alpha= 1, linewidth= 0.5)
    #select the cordinates of required map view (eg. Europe, Asia, Africa and others)
    ax.set_extent([-10, 23.5, 35, 62], ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,linewidth= 1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 15, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"color": "black", "size": 15, "weight": "bold"}
    if variable == "Temperature":
        # levels choose nice color levels within the defined vmin and vmax for better display
        img1 = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), cmap= plt.cm.RdBu_r,
                    vmin = vmin, vmax= vmax, levels = 37)
        #bounds = np.arange(vmin,vmax)
        #cmap=plt.cm.RdBu_r
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend= "both")
        #using DiverginNorm sets the midpoint to zero but the discrets colors want show
        #norm = mpl.colors.DivergingNorm(vmax= vmax, vmin= vmin, vcenter=0)
        num_levels = 20  # represents the color intervals on colormap (can be manually adjusted)
        midpoint = 0
        levels = np.linspace(vmin, vmax, num_levels)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colors_ = plt.cm.RdBu_r(vals)
        cmap, norm = mpl.colors.from_levels_and_colors(levels, colors_)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        cb = plt.colorbar(mappable=m, cax=None, ax= ax,
                              orientation= "vertical", pad= 0.1)
        cb.set_label(label="Temperature [°C]", size = 25, weight = "bold")
        cb.ax.tick_params(labelsize = 20)
        cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        
    elif variable == "Precipitation":
        img1 = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), cmap=plt.cm.RdBu, 
                    vmin = vmin, vmax= vmax, levels = 37)
        #bounds = np.arange(vmin,vmax)
        #cmap=plt.cm.RdBu_r
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend= "both")
        #using DiverginNorm sets the midpoint to zero but the discrets colors want show
        #norm = mpl.colors.DivergingNorm(vmax= vmax, vmin= vmin, vcenter=0)
        num_levels = 20  # represents the color intervals on colormap (can be manually adjusted)
        midpoint = 0
        levels = np.linspace(vmin, vmax, num_levels)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colors_ = plt.cm.RdBu(vals)
        cmap, norm = mpl.colors.from_levels_and_colors(levels, colors_)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        cb = plt.colorbar(mappable=m, cax=None, ax= ax,
                              orientation= "vertical", pad= 0.1)
        if datatype == "ERA":
            
            cb.set_label(label="Precipitation [mm/day]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        elif datatype == "GCM":
            
            cb.set_label(label="Precipitation [mm/day]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        else:
            cb.set_label(label="Precipitation [mm]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
    else:
        print("Define variable")
    if datatype == "ERA":
    
        plt.title('Mean annual difference([2005-2015]-[1979-1989]) [ERA-Interim]' , fontsize=20, weight = "bold")
        
    elif datatype == "GCM":
    
        plt.title('Mean annual difference(Present Day - Pre Industrial) [GCM]' , fontsize=20, weight = "bold")
    
    else:
          plt.title('Mean annual difference([2002-2012]-[1901-1911]) [CRU]' , fontsize=20, weight = "bold")
    
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "annual_mean_diff_" + variable +"_"+ datatype + '.png'))    
    plt.close()
    plt.clf()


#function for ploting annual mean difference of temperature and precipitation
def plot_monthly_mean_diff(variable, dataset, datatype, vmin, vmax, month_name):
    """
    this function generates the plot for long term monthly average difference for a specific month
    Variable = str  eg. Temperature or Precipitation
    dataset = xr.Dataset
    datatype = str  eg. ERA or CRU
    vmin = float (minimun value of the data)
    vmax = float (maximum value of the data)
    month_name = str--> eg. December, January...etc
    """
    fig, ax = plt.subplots(1, 1, sharex = False, figsize = (15, 13))
    # Interpolating between the 0 and 360 latitude to prevent display of white line
    
    mnames = [calendar.month_name[im+1] for im in np.arange(12)]
    for i, idx in enumerate(mnames):
        if idx == month_name:
            
            if datatype == "ERA":
                data, lon = add_cyclic_point(dataset[i], coord = dataset[i].longitude)
                lat = dataset[i].latitude
            else:
                data, lon = add_cyclic_point(dataset[i], coord = dataset[i].lon)
                lat = dataset[i].lat
        else:
            None
    #defining projection for the basemap
    projection = ccrs.Orthographic(central_latitude= 32)
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.coastlines(resolution = "50m")
    ax.add_feature(cfeature.BORDERS, color= "black", alpha= 1, linewidth= 0.5)
    #select the cordinates of required map view (eg. Europe, Asia, Africa and others)
    ax.set_extent([-10, 23.5, 35, 62], ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,linewidth= 1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 15, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"color": "black", "size": 15, "weight": "bold"}
    if variable == "Temperature":
        # levels choose nice color levels within the defined vmin and vmax for better display
        img1 = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), cmap= plt.cm.RdBu_r,
                    vmin = vmin, vmax= vmax, levels = 37)
        #bounds = np.arange(vmin,vmax)
        #cmap=plt.cm.RdBu_r
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend= "both")
        #using DiverginNorm sets the midpoint to zero but the discrets colors want show
        #norm = mpl.colors.DivergingNorm(vmax= vmax, vmin= vmin, vcenter=0)
        num_levels = 20  # represents the color intervals on colormap (can be manually adjusted)
        midpoint = 0
        levels = np.linspace(vmin, vmax, num_levels)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colors_ = plt.cm.RdBu_r(vals)
        cmap, norm = mpl.colors.from_levels_and_colors(levels, colors_)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        cb = plt.colorbar(mappable=m, cax=None, ax= ax,
                              orientation= "vertical", pad= 0.1)
        cb.set_label(label="Temperature [°C]", size = 25, weight = "bold")
        cb.ax.tick_params(labelsize = 20)
        cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        
    elif variable == "Precipitation":
        img1 = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), cmap=plt.cm.RdBu, 
                    vmin = vmin, vmax= vmax, levels = 37)
        #bounds = np.arange(vmin,vmax)
        #cmap=plt.cm.RdBu_r
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend= "both")
        #using DiverginNorm sets the midpoint to zero but the discrets colors want show
        #norm = mpl.colors.DivergingNorm(vmax= vmax, vmin= vmin, vcenter=0)
        num_levels = 20  # represents the color intervals on colormap (can be manually adjusted)
        midpoint = 0
        levels = np.linspace(vmin, vmax, num_levels)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colors_ = plt.cm.RdBu(vals)
        cmap, norm = mpl.colors.from_levels_and_colors(levels, colors_)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        cb = plt.colorbar(mappable=m, cax=None, ax= ax,
                              orientation= "vertical", pad= 0.1)
        if datatype == "ERA":
            
            cb.set_label(label="Precipitation [mm/day]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
            
        elif datatype == "GCM":
            
            cb.set_label(label="Precipitation [mm/day]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        else:
            cb.set_label(label="Precipitation [mm]", size = 25, weight = "bold")
            cb.ax.tick_params(labelsize = 20)
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
    else:
        print("Define variable")
    if datatype == "ERA":
    
        plt.title("Mean monthly(" + month_name +") difference([2005-2015] -[1979-1989]) [ERA-Interim]" , fontsize=17, weight = "bold")
    elif datatype == "GCM":
    
        plt.title("Mean monthly(" + month_name +") difference(Present Day - Pre Industrial) [GCM]" , fontsize=20, weight = "bold")
    
    else:
          plt.title("Mean monthly(" + month_name +") difference([2002-2012] -[1901-1911]) [CRU]" , fontsize=17, weight = "bold")
    
    #plt.tight_layout()
    plt.savefig(os.path.join(figuredir, "monthly_mean_diff_" + variable +"_"+ datatype +"_"+ month_name +'.png'))    
    plt.close()
    plt.clf()
    
    
#function for calculating the minimum and maximum for the set_extent on cartopy
def Cut_Europe_min(data):
    if hasattr(data, "latitude") & hasattr(data, "longitude"):
        data = data.where((data.latitude >= 35) & (data.latitude <= 62), drop=True)
        data = data.where((data.longitude >= 350) & (data.longitude <= 360) | (data.longitude >= 0) & (data.longitude <= 23))
    elif hasattr(data, "lat") & hasattr(data, "long"):
        data = data.where((data.lat >= 35) & (data.lat <= 62), drop=True)
        data = data.where((data.lon >= 350) & (data.lon <= 360) | (data.lon >= 0) & (data.lon <= 23))
    
    return data.min()
def Cut_Europe_max(data):
    if hasattr(data, "latitude") & hasattr(data, "longitude"):
        data = data.where((data.latitude >= 35) & (data.latitude <= 62), drop=True)
        data = data.where((data.longitude >= 350) & (data.longitude <= 360) | (data.longitude >= 0) & (data.longitude <= 23))
    elif hasattr(data, "lat") & hasattr(data, "long"):
        data = data.where((data.lat >= 35) & (data.lat <= 62), drop=True)
        data = data.where((data.lon >= 350) & (data.lon <= 360) | (data.lon >= 0) & (data.lon <= 23))
    
    return data.max()


    
# plotting annual means average and annual mean difference uning the above functions

#annual mean    
# plot_annual_mean("Temperature", t2m_era_annual_means, "ERA", Cut_Europe_min(t2m_era_annual_means), Cut_Europe_max(t2m_era_annual_means))   
# plot_annual_mean("Precipitation", tp_era_annual_means, "ERA",Cut_Europe_min(tp_era_annual_means), Cut_Europe_max(tp_era_annual_means))
# plot_annual_mean("Temperature", t2m_cru_annual_means, "CRU", Cut_Europe_min(t2m_cru_annual_means), Cut_Europe_max(t2m_cru_annual_means))   
# plot_annual_mean("Precipitation", tp_cru_annual_means, "CRU", Cut_Europe_min(tp_cru_annual_means), Cut_Europe_max(tp_cru_annual_means)) 
 
# #annual mean difference
# plot_annual_mean_diff("Temperature", t2m_era_annual_mean_diff, "ERA", Cut_Europe_min(t2m_era_annual_mean_diff), Cut_Europe_max(t2m_era_annual_mean_diff))   
# plot_annual_mean_diff("Precipitation", tp_era_annual_mean_diff, "ERA", Cut_Europe_min(tp_era_annual_mean_diff), Cut_Europe_max(tp_era_annual_mean_diff))
# plot_annual_mean_diff("Temperature", t2m_cru_annual_mean_diff, "CRU", Cut_Europe_min(t2m_cru_annual_mean_diff), Cut_Europe_max(t2m_cru_annual_mean_diff))   
# plot_annual_mean_diff("Precipitation", tp_cru_annual_mean_diff, "CRU", Cut_Europe_min(tp_cru_annual_mean_diff), Cut_Europe_max(tp_cru_annual_mean_diff)) 


#monthly mean difference
#plot_monthly_mean_diff("Temperature", t2m_era_monthly_mean_diff, "ERA", -1, 3, "December")
# mnames = [calendar.month_name[im+1] for im in np.arange(12)]
# for i, idx in enumerate(mnames):
#     # plot_monthly_mean_diff("Temperature", t2m_era_monthly_mean_diff, "ERA", Cut_Europe_min(t2m_era_monthly_mean_diff[i]), Cut_Europe_max(t2m_era_monthly_mean_diff[i]), idx)
#     # plot_monthly_mean_diff("Precipitation", tp_era_monthly_mean_diff, "ERA", Cut_Europe_min(tp_cera_monthly_mean_diff[i]), Cut_Europe_max(tp_era_monthly_mean_diff[i]), idx)
#     plot_monthly_mean_diff("Temperature", t2m_cru_monthly_mean_diff, "CRU", Cut_Europe_min(t2m_cru_monthly_mean_diff[i]), Cut_Europe_max(t2m_cru_monthly_mean_diff[i]), idx)
#     plot_monthly_mean_diff("Precipitation", tp_cru_monthly_mean_diff, "CRU", Cut_Europe_min(tp_cru_monthly_mean_diff[i]), Cut_Europe_max(tp_cru_monthly_mean_diff[i]), idx)
    


#plotting GCM simulations 

#annual mean
# plot_annual_mean("Temperature", temp2_pre_industrial_annual_mean, "GCM_pre_industrial", Cut_Europe_min(temp2_pre_industrial_annual_mean), Cut_Europe_max(temp2_pre_industrial_annual_mean))
# plot_annual_mean("Precipitation", prec_pre_industrial_annual_mean, "GCM_pre_industrial", Cut_Europe_min(prec_pre_industrial_annual_mean), Cut_Europe_max(prec_pre_industrial_annual_mean))
# plot_annual_mean("Temperature", temp2_present_day_annual_mean, "GCM_present_day", Cut_Europe_min(temp2_present_day_annual_mean), Cut_Europe_max(temp2_present_day_annual_mean))
# plot_annual_mean("Precipitation", prec_present_day_annual_mean, "GCM_present_day", Cut_Europe_min(prec_present_day_annual_mean), Cut_Europe_max(prec_present_day_annual_mean))
    
#annual difference
# plot_annual_mean_diff("Temperature", temp2_annual_diff, "GCM", Cut_Europe_min(temp2_annual_diff), Cut_Europe_max(temp2_annual_diff))
# plot_annual_mean_diff("Precipitation", prec_annual_diff, "GCM", Cut_Europe_min(prec_annual_diff), Cut_Europe_max(prec_annual_diff))

#monthly difference
# mnames = [calendar.month_name[im+1] for im in np.arange(12)]
# for i, idx in enumerate(mnames):
#       plot_monthly_mean_diff("Temperature", temp2_monthly_diff, "GCM", Cut_Europe_min(temp2_monthly_diff[i]), Cut_Europe_max(temp2_monthly_diff[i]), idx)
#       plot_monthly_mean_diff("Precipitation", prec_monthly_diff, "GCM", Cut_Europe_min(prec_monthly_diff[i]), Cut_Europe_max(prec_monthly_diff[i]), idx)



    
    