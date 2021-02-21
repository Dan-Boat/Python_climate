import os
import sys
import socket
import pandas as pd
import numpy as np
import argparse
from cycler import cycler
from stat_downscaling_tools import Dataset
from stat_downscaling_tools.weatherstation_utils import read_weatherstationnames, read_station_from_csv

radius = 250e3


# data location
era_datadir     = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/era"
echam_datadir   = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/echam"
cmip5_datadir   = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/cmip5/mpiesm/data"
station_datadir = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/climate_data_monthly/stations/"
figuredir       = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/figure/downscaling"
latexdir        = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/figure/downscaling/latex/"
predictordir    = os.path.join(os.path.dirname(__file__), '.predictors_' + str(int(radius/1000)))
cachedir        = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache'))


# DATA
# ----
ERAData = Dataset('ERA', {
    't2m':os.path.join(era_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(era_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(era_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(era_datadir, 'v10_monthly.nc'),
    'h700':os.path.join(era_datadir, 'h700_monthly.nc'),
    'tp':os.path.join(era_datadir, 'tp_monthly.nc'),
    'rh850':os.path.join(era_datadir, 'rh850_monthly.nc'),
    'dtd850':os.path.join(era_datadir, 'dtd850_monthly.nc'),
    'u850':os.path.join(era_datadir, 'u850_monthly.nc'),
    'v850':os.path.join(era_datadir, 'v850_monthly.nc'),
    'sst':os.path.join(era_datadir, 'sst_monthly.nc'),
})

ECHAMData = Dataset("ECHAM", {
    't2m':os.path.join(echam_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(echam_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(echam_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(echam_datadir, 'v10_monthly.nc'),
    'h700':os.path.join(echam_datadir, 'h700_monthly.nc'),
    'tp':os.path.join(echam_datadir, 'tp_monthly.nc'),
    'rh850':os.path.join(echam_datadir, 'rh850_monthly.nc'),
    'dtd850':os.path.join(echam_datadir, 'dtd850_monthly.nc'),
    'u850':os.path.join(echam_datadir, 'u850_monthly.nc'),
    'v850':os.path.join(echam_datadir, 'v850_monthly.nc'),
    'sst':os.path.join(echam_datadir, 'sst_monthly.nc'),
})

CMIP5_AMIP_R1= Dataset("CMIP5_AMIP_R1", {
    't2m':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'v10_monthly.nc'),
    'h700':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'h700_monthly.nc'),
    'tp':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'tp_monthly.nc'),
    'rh850':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'rh850_monthly.nc'),
    'dtd850':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'dtd850_monthly.nc'),
    'u850':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'u850_monthly.nc'),
    'v850':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'v850_monthly.nc'),
    'sst':os.path.join(cmip5_datadir, 'amip_r1i1p1', 'postprocessed', 'sst_monthly.nc'),
})

CMIP5_RCP26_R1= Dataset("CMIP5_RCP26_R1", {
    't2m':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'v10_monthly.nc'),
    'h700':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'h700_monthly.nc'),
    'tp':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'tp_monthly.nc'),
    'rh850':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'rh850_monthly.nc'),
    'dtd850':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'dtd850_monthly.nc'),
    'u850':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'u850_monthly.nc'),
    'v850':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'v850_monthly.nc'),
    'sst':os.path.join(cmip5_datadir, 'rcp2.6_r1i1p1', 'postprocessed', 'sst_monthly.nc'),
})

CMIP5_RCP45_R1= Dataset("CMIP5_RCP45_R1", {
    't2m':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'v10_monthly.nc'),
    'h700':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'h700_monthly.nc'),
    'tp':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'tp_monthly.nc'),
    'rh850':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'rh850_monthly.nc'),
    'dtd850':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'dtd850_monthly.nc'),
    'u850':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'u850_monthly.nc'),
    'v850':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'v850_monthly.nc'),
    'sst':os.path.join(cmip5_datadir, 'rcp4.5_r1i1p1', 'postprocessed', 'sst_monthly.nc'),
})

CMIP5_RCP45_R2= Dataset("CMIP5_RCP45_R2", {
    't2m':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'v10_monthly.nc'),
    'h700':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'h700_monthly.nc'),
    'tp':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'tp_monthly.nc'),
    'rh850':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'rh850_monthly.nc'),
    'dtd850':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'dtd850_monthly.nc'),
    'u850':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'u850_monthly.nc'),
    'v850':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'v850_monthly.nc'),
    'sst':os.path.join(cmip5_datadir, 'rcp4.5_r2i1p1', 'postprocessed', 'sst_monthly.nc'),
})

CMIP5_RCP45_R3= Dataset("CMIP5_RCP45_R3", {
    't2m':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'v10_monthly.nc'),
    'h700':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'h700_monthly.nc'),
    'tp':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'tp_monthly.nc'),
    'rh850':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'rh850_monthly.nc'),
    'dtd850':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'dtd850_monthly.nc'),
    'u850':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'u850_monthly.nc'),
    'v850':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'v850_monthly.nc'),
    'sst':os.path.join(cmip5_datadir, 'rcp4.5_r3i1p1', 'postprocessed', 'sst_monthly.nc'),
})

CMIP5_RCP85_R1= Dataset("CMIP5_RCP85_R1", {
    't2m':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'v10_monthly.nc'),
    'h700':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'h700_monthly.nc'),
    'tp':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'tp_monthly.nc'),
    'rh850':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'rh850_monthly.nc'),
    'dtd850':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'dtd850_monthly.nc'),
    'u850':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'u850_monthly.nc'),
    'v850':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'v850_monthly.nc'),
    'sst':os.path.join(cmip5_datadir, 'rcp8.5_r1i1p1', 'postprocessed', 'sst_monthly.nc'),
})

namedict = read_weatherstationnames(station_datadir)
stationnames = list(namedict.values())
