#Program:   Tourism Knapsack V1
#Inputs:    None
#Outputs:   Files: (Appends)
#           1. DATA_STORE_LOCATION/Statistics/Per Round.txt
#           2. DATA_STORE_LOCATION/Statistics/Total Cumulative.txt
#           3. DATA_STORE_LOCATION/Statistics/Individual Cumulative.txt
#Author:    Surja Sanyal
#Email:     hi.surja06@gmail.com
#Date:      11 JUL 2024
#Comments:  1. Please create a folder named "Statistics" in the location DATA_STORE_LOCATION. Outputs will be saved there.




##   Start of Code   ##


#   Imports    #

import os
import re
import sys
import copy
import pickle
import random
import datetime
import traceback
import collections
import numpy as np
import multiprocessing



##  Global environment   ##

#   Customize here  #
PRINT_ITERATION         = 0
PROPOSED_MECHANISM      = 2
RESIDENT_WEIGHTAGE      = 1
MINISTER_WEIGHTAGE      = 10                 #Number             #Ministers' vote weightage; = {1 / 3 / 5}
FREQUENT_FLIERS         = 0.1
TOURISTS                = 300               #Number             #Number of tourists; = {100 / 300 / 500}
RESIDENTS               = 1000              #Number             #Number of residents
MINISTERS               = 30
RANDOM_SEED             = 1
BUDGET                  = 1000000
#BUCKETS                 = 10                #Number             #Number of discrete levels of weigtage
MAX_TOURIST_FREQ        = 100               #Number             #Max visits by any tourist
#FACTOR                  = 5                #Number             #percent of (TOURISTS + RESIDENTS) = MINISTERS; = {5 / 6.5 / 10}
CREATE                  = True             #Boolean            #Save option for votes

#   Do not change   #
LOCK                    = multiprocessing.Lock()
DATA_LOAD_LOCATION      = os.path.dirname(sys.argv[0]) + "/"    #Local data load location
DATA_STORE_LOCATION     = os.path.dirname(sys.argv[0]) + "/"    #Local data store location
#DATA_LOAD_LOCATION     = "/content" + "/"                      #Drive data load location
#DATA_STORE_LOCATION    = "/content" + "/"                      #Drive data store location


#random.seed(RANDOM_SEED)


##  Function definitions    ##



#   Print with lock    #
def print_locked(*content, sep=" ", end="\n"):

    if (PRINT_ITERATION == 0):
        return
    store = DATA_STORE_LOCATION
    
    with open(store + "Logs/" + sys.argv[0].split(os.sep)[-1].split('.')[0].replace(" ", "_") + "_Log_File.txt", "a") as log_file:
    
        try:
        
            with LOCK:
            
                print (*content, sep = sep, end = end)
                print (*content, sep = sep, end = end, file=log_file)

        except Exception:
        
            print ("\n==> Failed to print below content to file.\n")
            print (*content, sep = sep, end = end)
            print ("\n==> Content not in log file ends here.\n")


#   Convert read values to int, str or list    #
def convert(some_value):

    try:
    
        return int(some_value)
        
    except ValueError:
    
        if ',' not in some_value:
        
            return some_value[1:-1]
        
        else:
        
            return [int(element) for element in re.split("\[|, |\]", some_value)]


#   Get statistics  #
def get_stats(tourists, residents, ministers, r_w, m_w, mech, seed):

    random.seed(seed)

    B = BUDGET

    load, store, max_t_freq, ff, create = DATA_LOAD_LOCATION, DATA_STORE_LOCATION, MAX_TOURIST_FREQ, FREQUENT_FLIERS, CREATE
    item_price_list = []

    with open(load + "_item_price_list.txt", "r") as fp:
                                        
        for line in fp:

            item_price_list = item_price_list + [[convert(piece) for piece in [x for x in re.split("\[|, |\]", line) if x.strip()]]]

    h_t_votes, h_r_votes, h_m_votes, d_t_votes, d_r_votes, d_m_votes = [], [], [], [], [], []
    items, prices = [it[0] for it in item_price_list], [it[1] for it in item_price_list]

    if(create):

        #   For tourists    #
        h_t_guy, h_r_guy, h_m_guy = [], [], []

        t_visit_freq = [int(value%max_t_freq) + 1 if (int(value%max_t_freq) == 0) else int(value%max_t_freq) \
                        for value in random.choices(pickle.load(open(store + "_tourist_freq.pickle", "rb")), k=tourists)]
        #t_visit_freq[0] = max_t_freq
        #t_visit_bucket = [visit//buckets + 1 for visit in t_visit_freq]
        #t_visit_bucket = [r_w + (m_w - r_w) * val / max(t_visit_freq) for val in t_visit_freq]
        # Frequent Fliers
        #print(t_visit_freq)
        c_ff = 0
        t_visit_bucket = [0.8 * m_w + 0.2 * m_w * val / max(t_visit_freq) for val in t_visit_freq[int(ff * len(t_visit_freq)):]] \
                         + [r_w + (0.8 * m_w - r_w) * val / max(t_visit_freq) for val in t_visit_freq[:int(ff * len(t_visit_freq))]]
        #print(t_visit_freq, t_visit_bucket, sep="\n\n")
        ff_first_prefs = []
        
        for i in range(tourists):

            total, choices, visited = 0, [], []

            while(len(visited) < len(items)):
                
                #print(i, len(visited), len(choices), len(items))
                next_item = random.choice([item for item in items if item not in visited])

                if(next_item not in visited):

                    visited = visited + [next_item]

                    if(total + prices[items.index(next_item)] <= B):
                        
                        choices = choices + [next_item]
                        total += prices[items.index(next_item)]

            h_t_votes = h_t_votes + choices

            if(i == 0):

                h_t_guy = copy.deepcopy(choices)
                d_t_guy = random.sample(choices, k=len(choices))

            if (i < int(ff * len(t_visit_freq))):

                c_ff += len(choices)
                ff_first_prefs += [choices[0]]

        #   For residents    #
        
        for i in range(residents):

            total, choices, visited = 0, [], []

            while(len(visited) < len(items)):
                
                #print(i, len(visited), len(choices), len(items))
                next_item = random.choice([item for item in items if item not in visited])

                if(next_item not in visited):

                    visited = visited + [next_item]

                    if(total + prices[items.index(next_item)] <= B):
                        
                        choices = choices + [next_item]
                        total += prices[items.index(next_item)]

            h_r_votes = h_r_votes + choices

            if(i == 0):

                h_r_guy = copy.deepcopy(choices)
                d_r_guy = random.sample(choices, k=len(choices))

        #   For ministers   #
        
        for i in range(ministers):

            total, choices, visited = 0, [], []

            while(len(visited) < len(items)):
                
                #print(i, len(visited), len(choices), len(items))
                next_item = random.choice([item for item in items if item not in visited])

                if(next_item not in visited):

                    visited = visited + [next_item]

                    if(total + prices[items.index(next_item)] <= B):

                        choices = choices + [next_item]
                        total += prices[items.index(next_item)]

            h_m_votes = h_m_votes + choices

            if(i == 0):

                h_m_guy = copy.deepcopy(choices)
                


        #   Dishonest   #
        d_t_votes, d_r_votes, d_m_votes = copy.deepcopy(h_t_votes), copy.deepcopy(h_r_votes), copy.deepcopy(h_m_votes)
        d_t_guy, d_r_guy, d_m_guy = random.sample(d_t_votes, k=len(h_t_guy)), random.sample(d_r_votes, k=len(h_r_guy)), random.sample(d_m_votes, k=len(h_m_guy))

        #print(d_m_guy)
        d_t_votes[:len(d_t_guy)] = d_t_guy
        d_r_votes[:len(d_r_guy)] = d_r_guy
        d_m_votes[:len(d_m_guy)] = d_m_guy
        #print(d_m_votes)

        h_t_votes, h_r_votes, h_m_votes, d_t_votes, d_r_votes, d_m_votes = \
            h_t_votes[:tourists], h_r_votes[:residents], h_m_votes[:ministers], d_t_votes[:tourists], d_r_votes[:residents], d_m_votes[:ministers]

        #   Save votes  #
        pickle.dump((h_t_votes, h_r_votes, h_m_votes, h_t_guy, h_r_guy, h_m_guy, d_t_votes, d_r_votes, d_m_votes, t_visit_bucket), open(store + "_votes.pickle", "wb"))


    else:

        #   Load votes  #
        h_t_votes, h_r_votes, h_m_votes, h_t_guy, h_r_guy, h_m_guy, d_t_votes, d_r_votes, d_m_votes, t_visit_bucket = pickle.load(open(load + "_votes.pickle", "rb"))

    #[print(len(item)) for item in [h_t_votes, h_r_votes, h_m_votes, d_t_votes, d_r_votes, d_m_votes]]

    h_t_freq, h_r_freq, h_m_freq = collections.Counter(h_t_votes), collections.Counter(h_r_votes), collections.Counter(h_m_votes)

    m_weighted_freq = collections.Counter()
    t_weighted_freq = collections.Counter()
    r_weighted_freq = collections.Counter()

    for i in h_r_freq:
        
        r_weighted_freq[i] = h_r_freq[i] * r_w

    if (mech == 0):

        for i in h_t_freq:
        
            t_weighted_freq[i] = h_t_freq[i] * r_w

    if (mech == 1):

        for i in range(len(h_t_votes)):
        
            t_weighted_freq[h_t_votes[i]] += t_visit_bucket[i]
    
    for i in h_m_freq:
        
        m_weighted_freq[i] = h_m_freq[i] * m_w

    #[print(a, b) for a, b in zip(h_t_votes, t_visit_bucket)]
    #print(h_t_freq, t_weighted_freq)

    results = t_weighted_freq + h_r_freq + m_weighted_freq
    results = sorted(results.items(), key=lambda x:(-x[1], x[0]), reverse=False)
    
    projects, costs, total_cost = [], [], 0
    for item, freq in results:

        index = items.index(item)
        price = prices[index]
        
        if(total_cost + price <= B):

            total_cost += price
            projects = projects + [item]
            costs = costs + [price]

    
    print_locked("\nWeightage given per vote for ministers:", m_w)
    print_locked("\nWeightage given per vote for residents:", r_w)
    print_locked("\nTotal number of tourists for this vote:", tourists)
    print_locked("\nTotal number of residents for this vote:", residents)
    print_locked("\nTotal number of ministers for this vote:", ministers)
    print_locked("\nTotal number of participants for this vote:", residents + tourists + ministers)
    print_locked("\nTotal number of projects selected:", len(projects))

    #print(factor * (tourists + residents)/100)

    print_locked("\nHONEST VOTE RESULTS ==>\n")

    print_locked("\n\nPROJECTS SELECTED AND THEIR VOTE COUNTS:\n")
    #print_locked("Projects Selected -> Tourist Votes -> Minister Votes -> Final Votes\n")
    #print(projects, total_cost)
    #print_range = len(projects)
    print_range = 3

    #print_locked("\n{:20s}".format("Projects Selected"), end="\t")
    print_locked()
    [print_locked(projects[i] + 1, end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Projects Costs"), end="\t")
    print_locked()
    [print_locked(costs[i], end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Resident Votes"), end="\t")
    print_locked()
    [print_locked(r_weighted_freq[projects[i]], end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Tourist Votes"), end="\t")
    print_locked()
    [print_locked("{:.2f}".format(t_weighted_freq[projects[i]]), end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Minister Votes"), end="\t")
    print_locked()
    [print_locked(m_weighted_freq[projects[i]], end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Final Votes"), end="\t")
    print_locked()
    [print_locked("{:.2f}".format(results[i][1]), end = "\t") for i in range(print_range)]
    print_locked()

    t_budget_utilization = 0
    for item in projects:

        if(item in h_t_freq.elements()):

            t_budget_utilization += prices[items.index(item)] * h_t_freq[item]

    r_budget_utilization = 0
    for item in projects:

        if(item in h_r_freq.elements()):

            r_budget_utilization += prices[items.index(item)] * h_r_freq[item]

    m_budget_utilization = 0
    for item in projects:

        if(item in h_m_freq.elements()):

            m_budget_utilization += prices[items.index(item)] * h_m_freq[item]

    h_t_guy_utility = 0
    for item in projects:

        if(item in h_t_guy):

            h_t_guy_utility += prices[items.index(item)]

    h_r_guy_utility = 0
    for item in projects:

        if(item in h_r_guy):

            h_r_guy_utility += prices[items.index(item)]

    h_m_guy_utility = 0
    for item in projects:

        if(item in h_m_guy):

            h_m_guy_utility += prices[items.index(item)]

    t_ff_utility, t_ff_selection = 0, 0
    for item in projects:

        if(item in h_t_votes[:c_ff]):
        #if(item in h_t_guy):

            t_ff_utility += prices[items.index(item)]
            t_ff_selection += 1

    print_locked("\n\nHONEST BUDGET UTILIZATION: ( % )\n")
    #print_locked("Tourists -> Ministers\n")
    #print_locked("Residents: {:.2f}\nTourists: {:.2f}\nMinisters: {:.2f}".format((100 * h_r_guy_utility)/(B), (100 * h_t_guy_utility)/(B), (100 * h_m_guy_utility)/(B)))
    print_locked("{:.2f}\n{:.2f}\n{:.2f}".format((100 * h_r_guy_utility)/(B), (100 * h_t_guy_utility)/(B), (100 * h_m_guy_utility)/(B)))
    print_locked()

    first_pref_index, max_pref_index, freq_first_pref = [], [], collections.Counter(ff_first_prefs)
    #freq_first_pref = sorted(freq_first_pref.items(), key=lambda x:(-x[1], x[0]), reverse=False)
    max_freq = max(freq_first_pref.values())
    max_first_pref = [val for val in freq_first_pref if freq_first_pref[val] == max_freq]
    
    first_pref_index += [100 * (len(projects) - projects.index(val)) / len(projects) if val in projects else 0 for val in ff_first_prefs]
    max_pref_index += [100 * (len(projects) - projects.index(val)) / len(projects) if val in projects else 0 for val in max_first_pref]

    return np.mean(first_pref_index), np.mean(max_pref_index)
    #(100 * t_ff_selection)/int(ff * len(t_visit_freq)), (100 * t_ff_utility)/(B)

    '''
    #h_t_freq, h_m_freq = copy.deepcopy(t_freq), copy.deepcopy(m_freq)
    t_freq, r_freq, m_freq = collections.Counter(d_t_votes), collections.Counter(d_r_votes), collections.Counter(d_m_votes)

    m_weighted_freq = collections.Counter()
    t_weighted_freq = collections.Counter()
    r_weighted_freq = collections.Counter()

    for i in r_freq:
        
        r_weighted_freq[i] = r_freq[i] * r_w

    for i in range(len(d_t_votes)):
        
        t_weighted_freq[d_t_votes[i]] += t_visit_bucket[i]

    for i in m_freq:
        
        m_weighted_freq[i] = m_freq[i] * m_w

    results = t_weighted_freq + r_freq + m_weighted_freq
    results = sorted(results.items(), key=lambda x:(-x[1], x[0]), reverse=False)
    
    projects, costs, total_cost = [], [], 0
    for item, freq in results:

        index = items.index(item)
        price = prices[index]
        
        if(total_cost + price <= B):

            total_cost += price
            projects = projects + [item]
            costs = costs + [price]

    print_locked("\nDISHONEST VOTE RESULTS ==>\n")
##    print_locked("\nWeightage given per vote for ministers:\t", weightage)
##    print_locked("\nTotal number of tourists for this vote:\t", tourists)
##    print_locked("\nTotal number of ministers for this vote:", int(factor * (tourists + residents)/100))
##    print_locked("\nTotal number of projects selected:\t", len(projects))

    #print_range = len(projects)

    print_locked("\n\nPROJECTS SELECTED AND THEIR VOTE COUNTS:\n")
    #print_locked("Projects Selected -> Tourist Votes -> Minister Votes -> Final Votes\n")

    #print_locked("\n{:20s}".format("Projects Selected"), end="\t")
    print_locked()
    [print_locked(projects[i] + 1, end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Projects Costs"), end="\t")
    print_locked()
    [print_locked(costs[i], end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Resident Votes"), end="\t")
    print_locked()
    [print_locked(r_weighted_freq[projects[i]], end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Tourist Votes"), end="\t")
    print_locked()
    [print_locked("{:.2f}".format(t_weighted_freq[projects[i]]), end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Minister Votes"), end="\t")
    print_locked()
    [print_locked(m_weighted_freq[projects[i]], end = "\t") for i in range(print_range)]
    #print_locked("\n{:20s}".format("Final Votes"), end="\t")
    print_locked()
    [print_locked("{:.2f}".format(results[i][1]), end = "\t") for i in range(print_range)]
    print_locked()

    t_budget_utilization = 0
    for item in projects:

        if(t_freq[item] > 0 and h_t_freq[item] > 0):

            t_budget_utilization += prices[items.index(item)] * min(h_t_freq[item], t_freq[item])

    r_budget_utilization = 0
    for item in projects:

        if(t_freq[item] > 0 and h_r_freq[item] > 0):

            r_budget_utilization += prices[items.index(item)] * min(h_r_freq[item], r_freq[item])

    m_budget_utilization = 0
    for item in projects:

        if(m_freq[item] > 0 and h_m_freq[item] > 0):

            m_budget_utilization += prices[items.index(item)] * min(h_m_freq[item], m_freq[item])

    d_t_guy_utility = 0
    for item in projects:

        if(item in h_t_guy):

            d_t_guy_utility += prices[items.index(item)]

    d_r_guy_utility = 0
    for item in projects:

        if(item in h_r_guy):

            d_r_guy_utility += prices[items.index(item)]

    d_m_guy_utility = 0
    for item in projects:

        if(item in h_m_guy):

            d_m_guy_utility += prices[items.index(item)]

    print_locked("\n\nDISHONEST BUDGET UTILIZATION: ( % )\n")
    #print_locked("Tourists -> Ministers\n")
    #print_locked("Residents: {:.2f}\nTourists: {:.2f}\nMinisters: {:.2f}".format((100 * d_r_guy_utility)/(B), (100 * d_t_guy_utility)/(B), (100 * d_m_guy_utility)/(B)))
    print_locked("{:.2f}\n{:.2f}\n{:.2f}".format((100 * d_r_guy_utility)/(B), (100 * d_t_guy_utility)/(B), (100 * d_m_guy_utility)/(B)))
    #print_locked("{:.2f}".format((100 * d_t_guy_utility)/(B)), "{:.2f}".format((100 * d_m_guy_utility)/(B)))
    print_locked()
    '''

    




##  The main function   ##

#   Main    #
def main():

    #   Get volunteer number    #
    tourists, residents, ministers, r_w, m_w, mechanisms = TOURISTS, RESIDENTS, MINISTERS, RESIDENT_WEIGHTAGE, MINISTER_WEIGHTAGE, PROPOSED_MECHANISM

    #   Generate statistics #
    for mech in range(mechanisms):
        
        sims = [100, 200, 300, 400, 500]
        #sims = [1]
        full_first_selection, full_max_selection = [0 for _ in range(len(sims))], [0 for _ in range(len(sims))]
        print("\nMechanism:", "Proposed" if (mech == 1) else "Existing")
        
        printed = -1
        for i in range(sims[-1]):

            if ((printed != int(100 * i / sims[-1])) and (int(100 * i / sims[-1]) % 10 == 0)):
                print("{:3d}%".format(int(100 * i / sims[-1])), end = "\t")
                printed = int(100 * i / sims[-1])
            
            first_selected, max_selected = get_stats(tourists, residents, ministers, r_w, m_w, mech, i)

            #'''
            if (i < 100):
                full_first_selection[0] += first_selected
                full_max_selection[0] += max_selected

            if (i < 200):
                full_first_selection[1] += first_selected
                full_max_selection[1] += max_selected

            if (i < 300):
                full_first_selection[2] += first_selected
                full_max_selection[2] += max_selected

            if (i < 400):
                full_first_selection[3] += first_selected
                full_max_selection[3] += max_selected

            if (i < 500):
                full_first_selection[4] += first_selected
                full_max_selection[4] += max_selected

            #'''


        full_first_selection = [value / category  for value, category in zip(full_first_selection, sims)]
        full_max_selection = [value / category  for value, category in zip(full_max_selection, sims)]

        print("\nFirst selection:")
        [print("{:.2f}".format(val), end="\t") for val in full_first_selection]

        print("\nMax selection:")
        [print("{:.2f}".format(val), end="\t") for val in full_max_selection]

        print()



##  Call the main function  ##

#   Initiation  #
if __name__=="__main__":

    try:
    
        #   Clear the terminal  #
        #os.system("clear")
    
        #   Start logging to file     #        
        print_locked('\n\n\n\n{:.{align}{width}}'.format("Execution Start at: " 
            + str(datetime.datetime.now()), align='<', width=70), end="\n\n")
        
        print_locked("\n\nProgram Name:\n\n" + str(sys.argv[0].split(os.sep)[-1]))
        
        print_locked("\n\nProgram Path:\n\n" + os.path.dirname(sys.argv[0]))
        
        print_locked("\n\nProgram Name With Path:\n\n" + str(sys.argv[0]), end="\n\n\n")
        
        #   Call the main program   #
        start = datetime.datetime.now()
        main()
        print_locked("\nProgram execution time:\t\t", datetime.datetime.now() - start, "hours\n")
        
        #   Notify completion   #
        #notify()
        
    except Exception:
    
        #   Print reason for abortion   #
        print_locked(traceback.format_exc())


##   End of Code   ##

