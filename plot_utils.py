import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import pandas as pd
from regression_settings import fullERA, fullECHAM
from data import *
from cycler import cycler

# A4 paper size: 210 mm X 297 mm
cm = 0.3937   # 1 cm in inch for plot size
pt = 1/72.27  # pt in inch from latex geometry package
textwidth = 345*pt
big_width = textwidth + 2*3*cm
# colors
orange = 'orangered'
lightblue = 'teal'
brown = 'sienna'
red = '#a41a36'
blue = '#006c9e'
green = '#55a868'
purple = '#8172b2'
lightbrown = '#ccb974'
pink = 'fuchsia'
lightgreen = 'lightgreen'
skyblue = "skyblue"
tomato = "tomato"
gold = "gold"
magenta = "magenta"
black = "black"

predictor_colors = {
        'rh850':green,
        'NAO':magenta,
        'PC1 NAO':brown,
        't2m':red,
        'msl':blue,
        'u850':orange,
        'v850':lightbrown,
        'dtd850':lightblue,
        'h700':pink,
        'intercept':lightgreen,
        "PC2_scan": purple,
        "PC3_ea": gold,
        "PC4": tomato,
        "PC5": skyblue
        }

score_colors = {'predict':black, 'fit':orange}

def apply_style(fontsize=10):
    small_size=fontsize-4
    plt.style.use('seaborn')
    #plt.style.use("fivethirtyeight")
#    plt.style.use('dark_background')
#    plt.style.use('bmh')    
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    mpl.rc('text', usetex=True)
    mpl.rc('font', size=11, family='serif')
    mpl.rc('xtick', labelsize=small_size)
    mpl.rc('ytick', labelsize=small_size)
    mpl.rc('legend', fontsize=small_size)
    mpl.rc('axes', labelsize=fontsize)
    mpl.rc('lines', linewidth=0.6)
    mpl.rc("font", weight="bold")

def apply_style_2(fontsize=10):
    small_size=fontsize-2
    #plt.style.use('seaborn')
#    plt.style.use('dark_background')
    #plt.style.use("fivethirtyeight")
    plt.style.use('bmh')    
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    mpl.rc('text', usetex=True)
    mpl.rc('font', size=15, family='serif')
    mpl.rc('xtick', labelsize=small_size)
    mpl.rc('ytick', labelsize=small_size)
    mpl.rc('legend', fontsize=small_size)
    mpl.rc('axes', labelsize=fontsize)
    mpl.rc('lines', linewidth=0.8)
    
    


def score_barplot(ax, scores, is_fit_score):

    df = pd.Series(scores, index=stationnames)

    # set colors and get patches
    colors = [score_colors['fit'] if isfit else score_colors['predict'] for isfit in is_fit_score]
    predict_patch = mpatches.Patch(color=score_colors['predict'], label='prediction score')
    fit_patch = mpatches.Patch(color=score_colors['fit'], label='fit score')

    df.plot(kind='bar', ax=ax, color=colors)
    ax.set_ylabel('climate score')
    ax.legend(handles=[predict_patch, fit_patch], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


def exp_var_barplot(ax, df_exp_var):

    predictors = df_exp_var.columns

    colors = [predictor_colors[p] for p in predictors]
    mpl.rcParams['axes.prop_cycle'] = cycler('color', colors)

    df_exp_var.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylim([0,1])
    ax.set_ylabel('explained variance')
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


def predictions_plot(ax, obs, era, echam):
    ax.plot(fullERA, obs, '-', color=green, label='Station values')
    ax.plot(fullERA, era, '--', color=red, label='ERA')
    ax.plot(fullECHAM, echam, '-.', color=blue, label='ECHAM')
