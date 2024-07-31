#Program:   Surplus Food Redistribution Simulations
#Inputs:    TBD
#Outputs:   TBD
#Author:    Surja Sanyal
#Date:      14 JUL 2024
#Comments:  None




##   Start of Code   ##


#   Imports    #

import os
import re
import sys
#import math
#import copy
#import time
#import psutil
#import shutil
import random
import datetime
import traceback
#import itertools
#import matplotlib
import numpy as np
import multiprocessing
#from textwrap import wrap
from matplotlib import rc
from functools import partial
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#from scipy.stats import truncnorm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


##  Global environment   ##

#   Customize here  #
AGENTS                  = 5000                  #Number                             Default number of agent requests
PAYLOAD_MAX             = 100                   #Number                             Volunteer payload limit in kilograms
COORDINATE_MAX          = 50                    #Number                             City start (at 0) to end limit in kilometers
DAY_MAX                 = 18                    #Number                             Day start (at 0) to end limit in hours
SAVE                    = "OFF"                 #ON/OFF                             Save data
SORTING                 = "END"                 #START/END                          Receiver sorting
PREFERENCE              = "ELIGIBLE"            #ORIGINAL/ELIGIBLE/UPDATED          Usage of preference lists
VOLUNTEERS              = "1X"                  #1X/2X/4X/8X/16X/32X                Volunteer availability (1/2/4/8/16/32) times donors
MANIPULATION            = "ON"                  #ON/OFF                             Manipulation of preferences

#   Thresholds  #
To      = 0.25                                  #Overlap time (hours)
Tl      = 5                                     #Off-routing (percentage)
Tm      = 1                                     #Meal size  (kilograms)
Ta      = 20                                    #Extra payload (percentage)
Tpm     = 5000                                  #Default perishable food travel distance (kilometers)
Tpnm    = 1000                                  #Default perishable food travel distance (kilometers)
Tnp     = 10000                                 #Default non-perishable food travel distance (kilometers)
Td      = 2                                     #Process advance start threshold for donors (hours)
Tr      = 3                                     #Process advance start threshold for receivers (hours)
Tw      = 10                                    #Match acceptance window (minutes)

#   Do not change   #
RESOLUTION          = 1000                                                  #Output resolution
LOCK                = multiprocessing.Lock()                                #Multiprocessing lock
#CPU_COUNT           = multiprocessing.cpu_count()                           #Logical CPUs
#MEMORY              = math.ceil(psutil.virtual_memory().total/(1024.**3))                   #RAM capacity
DATA_LOAD_LOCATION  = os.path.dirname(sys.argv[0]) + "/Statistics/"         #Data load location
#DATA_LOAD_LOCATION = os.path.dirname(sys.argv[0]) + "/Statistics/Incentive Module/Foolproof/"          #Data load location
DATA_STORE_LOCATION = os.path.dirname(sys.argv[0]) + "/Graphs/"             #Data store location
#DATA_STORE_LOCATION    = os.path.dirname(sys.argv[0]) + "/Graphs/Incentive Module/Foolproof/"              #Data store location


plt.rc('xtick', labelsize=23)
plt.rc('ytick', labelsize=23)


##  Function definitions    ##


#   Print with lock    #
def print_locked(*content, sep=" ", end="\n"):

    store = DATA_STORE_LOCATION
    
    with open(os.path.dirname(sys.argv[0]) + "/Logs/" + sys.argv[0].split(os.sep)[-1].split('.')[0].replace(" ", "_") + "_Log_File.txt", "a") as log_file:
    
        try:
        
            with LOCK:
            
                print (*content, sep = sep, end = end)
                print (*content, sep = sep, end = end, file=log_file)

        except Exception:
        
            print ("\n==> Failed to print below content to file.\n")
            print (*content, sep = sep, end = end)
            print ("\n==> Content not in log file ends here.\n")


#   Convert string to int or float    #
def convert(some_value):

    try:
        return int(some_value)
        
    except ValueError:
    
        try:
        
            return float(some_value)
        
        except ValueError:
        
            print_locked(traceback.format_exc())


#   display_votes   #
def display_votes(load, store, file_name):

    figsize = (10, 6)
    handleheight, handlelength = 2, 1
    loc = "upper center"
    markers = [".", "-", "x", "+", "xx||", "++", 'o', '.', '+', 'x', '*', 's']
    #offset = [bar_width * val for val in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]
    IDSes = ["Residents", "Tourists", "Ministers", "Total"]
    n_bars = len(IDSes)
    bar_width = 0.18
    ncols = n_bars
    columnspacing = 1
    offset = [bar_width * val for val in list(np.arange(-(n_bars - 1)/2, (n_bars - 1)/2 + 0.5, 1))]
    #data_backup = '{0}'.join(present_working_directory.split(os.sep)[:-1] + ["WSL-Backup", "DataBackup"]).format(os.sep)

    with open(load + file_name + ".txt", "r") as fp:
        
        for row, line in enumerate(fp):
        
            pieces = [convert(stat) for stat in line.strip().split()]
            if (row == 0):
                projects = pieces
            elif (row == 2):
                residents = pieces
            elif (row == 3):
                tourists = pieces
            elif (row == 4):
                ministers = pieces
            elif (row == 5):
                total = pieces

    # Plot attack detection results #
    locs = list(range(len(projects)))
    romans = ['I', 'II', 'III']
    results = [residents, tourists, ministers, total]
    plt.clf()
    fig, ax1 = plt.subplots(figsize = figsize)
    #ax1.set_ylim(80, 113)
    #ax1.set_yticks(range(80, 100 + 1, 5))
    ax1.set_xticks(locs, [romans[pos] + ": Project " + str(proj) for pos, proj in enumerate(projects)])
    
    [ax1.bar([mote + offset[i] for mote in locs], results[i], bar_width, color='w', edgecolor='k', linestyle='-', hatch=markers[i], \
              label=r'\textbf{' + IDSes[i] + r'}') for i in range(len(results))]
    [[ax1.text(locs[j] + offset[i], results[i][j] + 10,\
              '\\textbf{{{:.2f}}}'.format(results[i][j]), \
              ha = 'center', va = 'bottom', rotation = 90, fontsize=23) for j in range(len(results[i]))] for i in range(len(results))]
    
    ax1.set_ylabel('Weighted vote count', fontsize=30)
    ax1.set_xlabel('Selection order', fontsize=30)

    ax1.set_ylim(ax1.get_ylim()[0], 1.65 * ax1.get_ylim()[1])
    
    handles, labels = ax1.get_legend_handles_labels()

    legends = ax1.legend(
        #itertools.chain(*[handles[i::ncols] for i in range(ncols)]), itertools.chain(*[labels[i::ncols] for i in range(ncols)]), \
        loc=loc, \
                         #title="IDS Host Execution Scheme", 
                         handleheight=handleheight, handlelength=handlelength, ncols = ncols, fontsize=22, columnspacing=columnspacing, title_fontsize=20)
    
    legends.get_title().set_multialignment('center')
    fig.tight_layout()
    plt.savefig(os.path.join(store, file_name + ".pdf"), bbox_inches='tight')
    plt.close()



#   display_utilization   #
def display_utilization(load, store, file_name):

    figsize = (6, 6)
    handleheight, handlelength = 2, 1
    loc = "upper center"
    markers = [".", "x", "x", "+", "xx||", "++", 'o', '.', '+', 'x', '*', 's']
    #offset = [bar_width * val for val in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]
    IDSes = ["Honest", "Dishonest"]
    n_bars = len(IDSes)
    bar_width = 0.7
    ncols = n_bars
    columnspacing = 1
    offset = [bar_width * val for val in list(np.arange(-(n_bars - 1)/2, (n_bars - 1)/2 + 0.5, 1))]
    #data_backup = '{0}'.join(present_working_directory.split(os.sep)[:-1] + ["WSL-Backup", "DataBackup"]).format(os.sep)

    residents, tourists, ministers = [], [], []
    with open(load + file_name + ".txt", "r") as fp:
        
        for row, line in enumerate(fp):
        
            pieces = [convert(stat) for stat in line.strip().split()]
            if (row == 0):
                honest = pieces
            elif (row == 1):
                dishonest = pieces

    # Plot attack detection results #
    locs = list(range(len(honest)))[::2]
    #romans = ['1000', '1250', '1500', '1750', '2000']
    romans = ['1000', '1500', '2000']
    honest, dishonest = honest[::2], dishonest[::2]
    results = [honest, dishonest]
    plt.clf()
    fig, ax1 = plt.subplots(figsize = figsize)
    #ax1.set_ylim(80, 113)
    #ax1.set_yticks(range(80, 100 + 1, 5))
    ax1.set_xticks(locs, romans)
    
    [ax1.bar([mote + offset[i] for mote in locs], results[i], bar_width, color='w', edgecolor='k', linestyle='-', hatch=markers[i], \
              label=IDSes[i]) for i in range(len(results))]
    [[ax1.text(locs[j] + offset[i], results[i][j] + 2,\
              '\\textbf{{{:.2f}}}'.format(results[i][j]), \
              ha = 'center', va = 'bottom', rotation = 90, fontsize=23) for j in range(len(results[i]))] for i in range(len(results))]
    
    ax1.set_ylabel('Budget utilization (\%)', fontsize=30)
    ax1.set_xlabel('Total participants', fontsize=30)

    ax1.set_ylim(ax1.get_ylim()[0], 1.60 * ax1.get_ylim()[1])
    
    handles, labels = ax1.get_legend_handles_labels()

    legends = ax1.legend(
        #itertools.chain(*[handles[i::ncols] for i in range(ncols)]), itertools.chain(*[labels[i::ncols] for i in range(ncols)]), \
        loc=loc, \
                         #title="IDS Host Execution Scheme", 
                         handleheight=handleheight, handlelength=handlelength, ncols = ncols, fontsize=22, columnspacing=columnspacing, title_fontsize=20)
    
    legends.get_title().set_multialignment('center')
    fig.tight_layout()
    plt.savefig(os.path.join(store, file_name + ".pdf"), bbox_inches='tight')
    plt.close()



#   display_mechanism_comparison   #
def display_mechanism_comparison(load, store, file_name):

    figsize = (10, 6)
    handleheight, handlelength = 2, 1
    loc = "upper center"
    markers = [".", "x", "x", "+", "xx||", "++", 'o', '.', '+', 'x', '*', 's']
    #offset = [bar_width * val for val in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]
    IDSes = ["Existing", "Proposed"]
    n_bars = len(IDSes)
    bar_width = 0.4
    ncols = n_bars
    columnspacing = 1
    offset = [bar_width * val for val in list(np.arange(-(n_bars - 1)/2, (n_bars - 1)/2 + 0.5, 1))]
    #data_backup = '{0}'.join(present_working_directory.split(os.sep)[:-1] + ["WSL-Backup", "DataBackup"]).format(os.sep)

    residents, tourists, ministers = [], [], []
    with open(load + file_name + ".txt", "r") as fp:
        
        for row, line in enumerate(fp):
        
            pieces = [convert(stat) for stat in line.strip().split()]
            if (row == 0):
                existing = pieces
            elif (row == 1):
                proposed = pieces

    # Plot attack detection results #
    locs = list(range(len(existing)))#[::2]
    romans = ['100', '200', '300', '400', '500']
    #romans = ['1000', '1500', '2000']
    #honest, dishonest = honest[::2], dishonest[::2]
    results = [existing, proposed]
    plt.clf()
    fig, ax1 = plt.subplots(figsize = figsize)
    #ax1.set_ylim(80, 113)
    #ax1.set_yticks(range(80, 100 + 1, 5))
    ax1.set_xticks(locs, romans)
    
    [ax1.bar([mote + offset[i] for mote in locs], results[i], bar_width, color='w', edgecolor='k', linestyle='-', hatch=markers[i], \
              label=IDSes[i]) for i in range(len(results))]
    [[ax1.text(locs[j] + offset[i], results[i][j] + 2,\
              '\\textbf{{{:.2f}}}'.format(results[i][j]), \
              ha = 'center', va = 'bottom', rotation = 90, fontsize=23) for j in range(len(results[i]))] for i in range(len(results))]
    
    ax1.set_ylabel('Project selection rate (\%)', fontsize=30)
    ax1.set_xlabel('Total iterations', fontsize=30)

    ax1.set_ylim(ax1.get_ylim()[0], 1.60 * ax1.get_ylim()[1])
    
    handles, labels = ax1.get_legend_handles_labels()

    legends = ax1.legend(
        #itertools.chain(*[handles[i::ncols] for i in range(ncols)]), itertools.chain(*[labels[i::ncols] for i in range(ncols)]), \
        loc=loc, \
                         #title="IDS Host Execution Scheme", 
                         handleheight=handleheight, handlelength=handlelength, ncols = ncols, fontsize=22, columnspacing=columnspacing, title_fontsize=20)
    
    legends.get_title().set_multialignment('center')
    fig.tight_layout()
    plt.savefig(os.path.join(store, file_name + ".pdf"), bbox_inches='tight')
    plt.close()





##  The main function   ##

#   Main    #
def main():

    load, store = DATA_LOAD_LOCATION, DATA_STORE_LOCATION

    r_w = ['r_1', 'r_5']
    m_w = ['m_5', 'm_10']
    m_c = ['m_c_10', 'm_c_50']
    r_c = ['r_500', 'r_1500']
    t_c = ['t_100', 't_500']
    h_d = ['r_hd', 't_hd', 'm_hd']
    t_w = ['t_rw', 't_tw']
    f_f = ['ff_10', 'ff_20']
    f_m = ['fs', 'ms']

    #[display_votes(load, store, file) for file in r_w + m_w + m_c + r_c + t_c]
    #[display_votes(load, store, file) for file in t_w + f_f]
    #[display_utilization(load, store, file) for file in h_d]
    [display_mechanism_comparison(load, store, file) for file in f_m]


##  Call the main function  ##

#   Initiation  #
if __name__=="__main__":

    try:
    
        #   Start logging to file     #        
        print_locked('\n\n\n\n{:.{align}{width}}'.format("Execution Start at: " 
            + str(datetime.datetime.now()), align='<', width=70), end="\n\n")
        
        print_locked("\n\nProgram Name:\n\n" + str(sys.argv[0].split("/")[-1]))
        
        print_locked("\n\nProgram Path:\n\n" + os.path.dirname(sys.argv[0]))
        
        print_locked("\n\nProgram Name With Path:\n\n" + str(sys.argv[0]), end="\n\n\n")
        
        #   Clear the terminal  #
        #os.system("clear")
        
        #   Initiate lock object    #
        #lock = multiprocessing.Lock()

        #   Initiate pool objects   #
        #pool = multiprocessing.Pool(multiprocessing.cpu_count())
        
        #   Call the main program   #
        start = datetime.datetime.now()
        main()
        print_locked("\nProgram execution time:\t\t", datetime.datetime.now() - start, "hours\n")
        
        #   Close Pool object    #
        #pool.close()

    except Exception:
    
        print_locked(traceback.format_exc())


##   End of Code   ##

