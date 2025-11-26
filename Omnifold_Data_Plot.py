from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize

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
#nat_weights = np.ones(len(nat_pt_smear))*1
#syn_weights = np.ones(len(syn_pt_smear))*1

myweights = np.load("Weights_Syn1_5Per_Log_5_10.npy")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hist_syn_pt_smear=axes[0].hist(syn_pt_smear,weights=syn_weights*np.exp((syn_pt_smear)**(0.67897134)),bins=np.linspace(0,100,100),color='blue',alpha=0.5,label="MC, reco")
hist_nat_pt_smear=axes[0].hist(nat_pt_smear,weights=nat_weights*np.exp((nat_pt_smear)**(0.67897134)),bins=np.linspace(0,100,100),color='orange',alpha=0.5,label="Data, reco")
hist_est_pt_smear=axes[0].hist(syn_pt_smear,weights=myweights[0, :]*np.exp((syn_pt_smear)**(0.67897134)), bins=np.linspace(0,100,100),color='black',histtype="step",label="OmniFolded - Single",lw=2)

hist_syn_pt_gen=axes[1].hist(syn_pt_gen,weights=syn_weights*np.exp((syn_pt_gen)**(0.67897134)),bins=np.linspace(0,100,100),color='blue',alpha=0.5,label="MC, true")
hist_nat_pt_gen=axes[1].hist(nat_pt_gen,weights=nat_weights*np.exp((nat_pt_gen)**(0.67897134)),bins=np.linspace(0,100,100),color='orange',alpha=0.5,label="Data, true")
hist_est_pt_gen=axes[1].hist(syn_pt_gen,weights=myweights[1, :]*np.exp((syn_pt_gen)**(0.67897134)),bins=np.linspace(0,100,100),color='black',histtype="step",label="OmniFolded - Single",lw=2)
      
for k in range(2):
    axes[k].set_yscale('log')
    axes[k].set_xlabel("$p_{T}$")
    axes[k].set_ylabel("events")
    axes[0].set_title("Detector Level")
    axes[1].set_title("True Level")
    axes[k].legend(frameon=False)
    axes[k].semilogy()

#print(tot_counts)
#print(sum(tot_counts))
fig.show()
plt.savefig("Steep_Good_Omni_View.jpg")