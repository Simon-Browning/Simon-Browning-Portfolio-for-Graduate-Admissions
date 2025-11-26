# Code by Simon Browning
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import uproot

events_nat = uproot.open("mockdata.nat.Logweighted2.N150000.root:tree_nat_weight")
events_nat.show()
events_syn = uproot.open("mockdata.syn1.5Percent.Logweighted2.N150000.root:tree_syn_weight")
events_syn.show()

nat_pt_gen = events_nat["nat_pt_gen"].array(library="np")
nat_pt_smear = events_nat["nat_pt_smear"].array(library="np")
syn_pt_gen = events_syn["syn_pt_gen"].array(library="np") # closure
syn_pt_smear = events_syn["syn_pt_smear"].array(library="np") # closure
nat_weights = events_nat["nat_pt_weight"].array(library="np")
syn_weights = events_syn["syn_pt_weight"].array(library="np")

N_nat = len(nat_pt_gen)
N_syn = len(syn_pt_gen)
nat_pt_gen = np.array([(nat_pt_gen[i])*1 for i in range(N_nat)])
nat_pt_smear = np.array([(nat_pt_smear[i])*1 for i in range(N_nat)])
syn_pt_gen = np.array([(syn_pt_gen[i])*1 for i in range(N_syn)])
syn_pt_smear = np.array([(syn_pt_smear[i])*1 for i in range(N_syn)])
nat_weights = np.array([(nat_weights[i])*1 for i in range(N_nat)])
syn_weights = np.array([(syn_weights[i])*1 for i in range(N_syn)])

# Create arrays to hold percent differences
gen_diff = np.zeros((4, 10, 5))
gen_diff_func = np.zeros((4, 10, 5))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for k in range(1):
    axes[k].set_title('Percent Differences for Jet $p_{T}$ between '+str((k+2)*25)+' and '+str((k+2)*25 + 25)+' MeV')
    axes[k].set_xlabel('Iteration')
    axes[k].set_ylabel('Percent Difference')
    pass

for j in range(1, 6):
# Loop thru all iterations
    for iter in range(1, 11):
        print("Current iteration: "+str(iter))
        myweights = np.load("Test_Syn1_5Per_Log_"+str(j)+"_"+str(iter)+".npy")
        myweights_gen = myweights[1, :]
        myweights_func = np.load("Weights_Syn1_5Per_Log_"+str(j)+"_"+str(iter)+".npy")
        myweights_gen_func = myweights_func[1, :]

        # Find yields for 25 MeV momentum ranges for no weight function
        pT = 0
        while pT < 100:
            # Browning - get the counts from the histograms
            gen_counts,_,_ = plt.hist(nat_pt_gen, weights = nat_weights*np.exp((nat_pt_gen)**(0.67897134)), bins = np.linspace(pT, pT + 25, 2))
            omni_gen_counts,_,_ = plt.hist(syn_pt_gen, weights = myweights_gen*np.exp((syn_pt_gen)**(0.67897134)), bins = np.linspace(pT, pT + 25, 2))

            gen_counts_tot = sum(gen_counts)
            omni_gen_counts_tot = sum(omni_gen_counts)

            # Find percent difference
            per_gen_diff = ((np.abs(gen_counts_tot - omni_gen_counts_tot))/(gen_counts_tot + omni_gen_counts_tot))*200

            # Save current percent differences to array
            gen_diff[int(pT/25), iter - 1, j - 1] = per_gen_diff

            pT = pT + 25


        # Find yields for 25 MeV momentum ranges with weight function
        pT = 0
        while pT < 100:
            # Browning - get the counts from the histograms
            gen_counts,_,_ = plt.hist(nat_pt_gen, weights = nat_weights*np.exp((nat_pt_gen)**(0.67897134)), bins = np.linspace(pT, pT + 25, 2))
            omni_gen_counts_func,_,_ = plt.hist(syn_pt_gen, weights = myweights_gen_func*np.exp((syn_pt_gen)**(0.67897134)), bins = np.linspace(pT, pT + 25, 2))

            gen_counts_tot = sum(gen_counts)
            omni_gen_counts_tot_func = sum(omni_gen_counts_func)

            # Find percent difference
            per_gen_diff_func = ((np.abs(gen_counts_tot - omni_gen_counts_tot_func))/(gen_counts_tot + omni_gen_counts_tot_func))*200

            # Save current percent differences to array
            gen_diff_func[int(pT/25), iter - 1, j - 1] = per_gen_diff_func

            pT = pT + 25

# Calculate averages and standard deviations
averages = np.zeros((4, 10))
stds = np.zeros((4, 10))
for p in range(4):
    for i in range(10):
        averages[p, i] = np.mean(gen_diff[p, i, :])
        stds[p, i] = np.std(gen_diff[p, i, :])

# repeat for weight function quantitities
averages_func = np.zeros((4, 10))
stds_func = np.zeros((4, 10))
for p in range(4):
    for i in range(10):
        averages_func[p, i] = np.mean(gen_diff_func[p, i, :])
        stds_func[p, i] = np.std(gen_diff_func[p, i, :])

# Create and save plots
for k in range(1):
    axes[k].errorbar(np.linspace(1, 10, num=10), averages[k+2, :], xerr=None, yerr=stds[k+2, :], ecolor='orange',color='orange', marker='o', markersize=5, capsize=5, label='No Weight Function')
    axes[k].errorbar(np.linspace(1, 10, num=10), averages_func[k+2, :], xerr=None, yerr=stds_func[k+2, :], ecolor='blue',color='blue', marker='o', markersize=5, capsize=5, label='With Weight Function')
    if k == 0:
        axes[k].legend(frameon=False)
    pass

plt.savefig("Poster_Results_3.jpg")
fig.show()