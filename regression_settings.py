import pandas as pd
'''- this script contains the predictor sets for fitting the regression model and
the date range for fitting (training and prediction range)

'''

#Predictors

predictors_no_indices = ["t2m", "msl", "u850", "v850", "h700", "dtd850", "rh850"]
predictors = ["msl","rh850", "u850", "PC1_NAO", "v850", "h700", "dtd850"]

#selected predictors for the machine learning algorithms
selected_predictors = ["msl", "u850", "PC1_NAO", "h700", "rh850"]

#just to understand the synoptic control on local scale
predictors_no_rh850_u850 = ["t2m", "msl", "dtd850", "v850", "PC1_NAO"]



# date ranges for fitting

full       = pd.date_range(start='1979-01-01', end='2100-12-31', freq='MS')
fullERA    = pd.date_range(start='1979-01-01', end='2014-12-31', freq='MS')
from79to01 = pd.date_range(start='1979-01-01', end='2001-12-31', freq='MS')
from02to19 = pd.date_range(start='2002-01-01', end='2019-12-31', freq='MS')
fullAMIP   = pd.date_range(start='1979-01-01', end='2008-12-31', freq='MS')
fullECHAM  = pd.date_range(start='1999-01-01', end='2014-12-31', freq='MS')
from79to00 = pd.date_range(start='1979-01-01', end='2000-12-31', freq='MS')
from01to15 = pd.date_range(start='2001-01-01', end='2014-12-31', freq='MS')
fullCMIP5  = pd.date_range(start='2006-01-01', end='2100-12-31', freq='MS')
from15to00 = pd.date_range(start='2015-01-01', end='2100-12-31', freq='MS')
fullStation= pd.date_range(start='1891-01-01', end='2020-12-31', freq='MS')
from20to40 = pd.date_range(start='2020-01-01', end='2040-12-31', freq='Q-NOV')
from40to60 = pd.date_range(start='2040-01-01', end='2060-12-31', freq='Q-NOV')
from60to80 = pd.date_range(start='2060-01-01', end='2080-12-31', freq='Q-NOV')
from80to21 = pd.date_range(start='2080-01-01', end='2100-12-31', freq='Q-NOV')
from79to14 = pd.date_range(start='1979-01-01', end='2014-12-31', freq='Q-NOV')



