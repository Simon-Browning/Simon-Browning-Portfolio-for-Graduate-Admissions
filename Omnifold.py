from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import copy

import uproot

from sklearn import datasets
from sklearn import preprocessing
from sklearn import neural_network
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import  make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from scikeras.wrappers import KerasRegressor
from scikeras.wrappers import KerasClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
#from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras


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
#nat_weights = np.array([(nat_weights[i])*1 for i in range(N_nat)])
#syn_weights = np.array([(syn_weights[i])*1 for i in range(N_syn)])
nat_weights = np.ones(len(nat_pt_smear))*1
syn_weights = np.ones(len(syn_pt_smear))*1

# Browning - only keep data samples that are less than 40
'''for i, val in enumerate(nat_pt_gen):
    if val > 25:
        nat_pt_gen[i] = np.nan
        nat_pt_smear[i] = np.nan
        #nat_weights[i] = np.nan
nat_pt_gen = nat_pt_gen[~np.isnan(nat_pt_gen)]
nat_pt_smear = nat_pt_smear[~np.isnan(nat_pt_smear)]
#nat_weights = nat_weights[~np.isnan(nat_weights)]

for i, val in enumerate(syn_pt_gen):
    if val > 25:
        syn_pt_gen[i] = np.nan
        syn_pt_smear[i] = np.nan
        #syn_weights[i] = np.nan
syn_pt_gen = syn_pt_gen[~np.isnan(syn_pt_gen)]
syn_pt_smear = syn_pt_smear[~np.isnan(syn_pt_smear)]
#syn_weights = syn_weights[~np.isnan(syn_weights)]'''

syn_pt = np.stack([syn_pt_gen, syn_pt_smear], axis=1)

# Browning - this model may be optimized using Keras tuner
inputs = Input((1, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model = Model(inputs=inputs, outputs=outputs)

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

j = 1
# Browning - added loop to produce mutiple test plots
while j <= 5:
  iterations = 10
  verbose = 0 # this just tells the model if we want to see the progress

  # this will ultimately hold the weights for each iteration and step
  weights = np.empty(shape=(iterations, 2, len(syn_pt)))
  # shape = (iteration, step, event)

  # labels for the classifier
  labels_s = np.zeros(len(syn_pt)) # the synthetic label is 0
  labels_n = np.ones(len(nat_pt_smear)) # the natural label is 1


  # Put together the features and labels for Step 1
  #   xvals_1 holds the pT for the smeared synthetic followed by the smeared natural
  #   yvals_1 holds the labels -- synthetic are 0 and natural are 1
  
  #xvals_1 = np.concatenate((syn_pt_smear, nat_pt_smear)).reshape(-1, 1)
  #yvals_1 = np.concatenate((labels_s, labels_n)).reshape(-1, 1)
  xvals_1 = np.concatenate((syn_pt_smear, nat_pt_smear)).reshape(-1, 1)
  yvals_1 = np.concatenate((np.zeros(len(syn_pt_smear)), np.ones(len(nat_pt_smear)))).reshape(-1, 1)

  xvals_2 = np.concatenate((syn_pt_gen, syn_pt_gen)).reshape(-1, 1)
  yvals_2 = np.concatenate((np.zeros(len(syn_pt_gen)), np.ones(len(syn_pt_gen)))).reshape(-1, 1)

  # Put together the features and labels for Step 2
  #   xvals_2 holds the pT for the generated synthetic followed by another round
  #       of generated synthetic. The latter values will be reweighted with the
  #       results from Step 1
  #   yvals_2 holds the labels -- unweighted generated synthetic will be 0 and
  #       weighted generated synthetic will be 1
  #xvals_2 = np.concatenate((syn_pt_gen, syn_pt_gen)).reshape(-1, 1)
  #yvals_2 = np.concatenate((labels_s, labels_n)).reshape(-1, 1)

  # initial iterative weights are ones
  weights_pull = np.ones(len(syn_pt_smear))
  #weights_push = np.ones(len(syn_pt_smear))
  weights_push = copy.deepcopy(syn_weights)

  for i in range(iterations):
    # Browning - following line added to display iteration number

    print("Current iteration: "+str(i + 1))
    if (verbose>0):
      print("\nITERATION: {}\n".format(i + 1))
      pass

    # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
    # weights reweighted Sim. --> Data

    if (verbose>0):
      print("STEP 1\n")
      pass

    # here we take the initial (push) weights and tack on a bunch of 1s to be reweighted
    weights_1 = np.concatenate((weights_push, nat_weights)).reshape(-1, 1)
    #weights_1 = np.concatenate((weights_push, np.ones(len(nat_pt_smear)))).reshape(-1, 1)

    # Browning - custom weight function
    weights_1_new = weights_1*np.exp((xvals_1)**(0.67897134))

    # this is where we split the data into training and testing (validation) samples
    X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1_new)

    callback = EarlyStopping(monitor='binary_crossentropy', patience=30, verbose=1, restore_best_weights=True)

    # compile the NN model
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  weighted_metrics=['binary_crossentropy'])
    # Use the model to fit the smeared samples: synthetic = 0; natural = 1
    model.fit(X_train_1,
              Y_train_1,
              sample_weight=w_train_1,
              epochs=1000,
              batch_size=10000,
              validation_data=(X_test_1, Y_test_1, w_test_1),
              verbose=verbose,
              callbacks=[callback])
    
    results_1 = model.evaluate(X_test_1, Y_test_1, batch_size=2000)
    #print("test loss, test acc:", results_1)

    #print('Step 1 Score',i,':',model.evaluate(X_test_1, Y_test_1))
    # have the NN predict the weights for the smeared synthetic
    #   take these and scale the previous weights
    weights_pull = weights_push * reweight(syn_pt_smear,model)
    # this inserts the pulled weights into the first step for that iteration
    weights[i, :1, :] = weights_pull

    # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
    # weights Gen. --> reweighted Gen.

    if (verbose>0):
      print("\nSTEP 2\n")
      pass

    # we load up 1s for weights of the (unweighted) generated synthetic
    #   we use the pulled weights for the weighted generated synthetic
    #weights_2 = np.concatenate((np.ones(len(syn_pt_gen)), weights_pull)).reshape(-1, 1)
    weights_2 = np.concatenate((syn_weights, weights_pull)).reshape(-1, 1)
    # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

    # Browning - custom weight function
    weights_2_new = weights_2*np.exp((xvals_2)**(0.67897134))

    # Again, we'll split a validation sample
    X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2_new)

    # Compile the NN model
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  weighted_metrics=['binary_crossentropy'])
    # Use the model to fit the generated samples: unweighted synthetic = 0; weighted = 1
    model.fit(X_train_2,
              Y_train_2, 
              sample_weight=w_train_2,
              epochs=1000,
              batch_size=2000,
              validation_data=(X_test_2, Y_test_2, w_test_2),
              verbose=verbose,
              callbacks=[callback])
    
    results_2 = model.evaluate(X_test_2, Y_test_2, batch_size=2000)
    #print("test loss, test acc:", results_2)

    #print('Step 2 Score',i,':',model.evaluate(X_test_2, Y_test_2))
    # Have the NN predict the weights for the generated synthetic
    #   We will use these weights to start the next iteration or
    #   estimate the true distribution
    #weights_push = reweight(syn_pt_gen,model)
    weights_push = reweight(syn_pt_gen,model)*syn_weights
    # insert the weights into the second step for that iteration
    weights[i, 1:2, :] = weights_push
    pass

    # Browning - may have to undo following indent
    #return weights
    #myweights = of.omnifold(theta0,theta_unknown_S,2,model)
    myweights = weights

    # Browning - save weights array
    np.save("Steep_"+str(j)+"_"+str(i + 1)+".npy", myweights[i, :, :])
  
    if i == 9:
    #myweights_df = pd.DataFrame(myweights[i, :, :])
    #myweights_df.to_csv("Syn2_First_25_"+str(i + 1)+"_Iter.csv", index = False)

      fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

      '''hist_syn_pt_smear=axes[0].hist(syn_pt_smear,weights=syn_weights*np.exp((syn_pt_smear)**(0.67897134)),bins=np.linspace(0,100,100),color='blue',alpha=0.5,label="MC, reco")
      hist_nat_pt_smear=axes[0].hist(nat_pt_smear,weights=nat_weights*np.exp((nat_pt_smear)**(0.67897134)),bins=np.linspace(0,100,100),color='orange',alpha=0.5,label="Data, reco")
      hist_est_pt_smear=axes[0].hist(syn_pt_smear,weights=myweights[i, 0, :]*np.exp((syn_pt_smear)**(0.67897134)), bins=np.linspace(0,100,100),color='black',histtype="step",label="OmniFolded - Single",lw=2)

      hist_syn_pt_gen=axes[1].hist(syn_pt_gen,weights=syn_weights*np.exp((syn_pt_gen)**(0.67897134)),bins=np.linspace(0,100,100),color='blue',alpha=0.5,label="MC, true")
      hist_nat_pt_gen=axes[1].hist(nat_pt_gen,weights=nat_weights*np.exp((nat_pt_gen)**(0.67897134)),bins=np.linspace(0,100,100),color='orange',alpha=0.5,label="Data, true")
      hist_est_pt_gen=axes[1].hist(syn_pt_gen,weights=myweights[i, 1, :]*np.exp((syn_pt_gen)**(0.67897134)), bins=np.linspace(0,100,100),color='black',histtype="step",label="OmniFolded - Single",lw=2)
      '''
      hist_syn_pt_smear=axes[0].hist(syn_pt_smear,weights=syn_weights,bins=np.linspace(0,100,100),color='blue',alpha=0.5,label="MC, reco")
      hist_nat_pt_smear=axes[0].hist(nat_pt_smear,weights=nat_weights,bins=np.linspace(0,100,100),color='orange',alpha=0.5,label="Data, reco")
      hist_est_pt_smear=axes[0].hist(syn_pt_smear,weights=myweights[i, 0, :], bins=np.linspace(0,100,100),color='black',histtype="step",label="OmniFolded - Single",lw=2)

      hist_syn_pt_gen=axes[1].hist(syn_pt_gen,weights=syn_weights,bins=np.linspace(0,100,100),color='blue',alpha=0.5,label="MC, true")
      hist_nat_pt_gen=axes[1].hist(nat_pt_gen,weights=nat_weights,bins=np.linspace(0,100,100),color='orange',alpha=0.5,label="Data, true")
      hist_est_pt_gen=axes[1].hist(syn_pt_gen,weights=myweights[i, 1, :], bins=np.linspace(0,100,100),color='black',histtype="step",label="OmniFolded - Single",lw=2)
      
      for k in range(2):
          # Browning - bake accuracy into plots for easy record-keeping
          axes[0].set_title('Detector-Level Data')
          axes[1].set_title('Particle-Level Data')

          axes[k].set_xlabel("jet $p_{T}$")
          axes[k].set_ylabel("events")
          axes[k].legend(frameon=False)
          axes[k].semilogy()
          pass

      fig2.show()
      plt.savefig("Steep_"+str(j)+".jpg")
    # End of potential indent

  j = j + 1