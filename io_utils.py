# functions for storing and reading data 
import pickle
import os
import pandas as pd
from collections import OrderedDict
from data import stationnames, cachedir


#TO DO 
# define the same function but specify the path of the solution stored for swr, lasso, and ML
# just to creat a function for plotting all of them together 
lasso_dir        = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/GLM/model/final_cache_LassoCV"
OLSForward_dir   = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/GLM/model/final_cache_OLSForward"
SVR_dir          = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/GLM/model/final_cache_SVR"          
MLP_dir          = "C:/Users/Boateng/Desktop/education/Master_Thesis/Modules/GLM/model/final_cache_MLP"

def store_pickle(stationname, varname, var):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(cachedir, filename + '_' + varname + '.pickle') 
    with open(fname, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(stationname, varname,path):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(path, filename + '_' + varname + '.pickle')
    #fname = os.path.join(lasso_dir, filename + '_' + varname + '.pickle')
    with open(fname, 'rb') as f:
        return pickle.load(f)

def store_csv(stationname, varname, var):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(cachedir, filename + '_' + varname + '.csv') 
    var.to_csv(fname)


def load_csv(stationname, varname, path):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(path, filename + '_' + varname + '.csv') 
    #fname = os.path.join(cachedir, filename + '_' + varname + '.csv') 
    return pd.read_csv(fname, index_col=0, parse_dates=True)



def load_all_stations(varname, path):
    """
    This assumes that the stored quantity is a dictionary

    Returns a dictionary
    """
    values_dict = OrderedDict()

    for stationname in stationnames:
        values_dict[stationname] = load_pickle(stationname, varname, path)

    df = pd.DataFrame(values_dict).transpose()
    # get right order
    columns = list(values_dict[stationname].keys())
    df =  df.loc[stationnames]
    return df[columns]




