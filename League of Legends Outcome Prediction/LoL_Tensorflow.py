import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier

def loadDataAndTargets():
    """Reads and loads the data csv file"""

    #df = pd.read_csv("/content/final_data.csv") -- used for Google Colab
    df = pd.read_csv("Final_data_w.o_base_stats2.csv")
    df.columns.values[0] = "index"
    df = df.drop(['index'], axis = 1)
    #y = df["win"].to_numpy()
    return df

# Eliminate rows that have NaN
def remove_bad_rows(df_tier):
    """Removes rows with NaN values"""

    data = df_tier.to_numpy()[:, 1:]  # Convert to NumPy array if not already one
    num_rows, num_cols = data.shape

    i = 0

    while i < num_rows:
      # Convert to float to handle non-numeric types
        if np.isnan(data[i].astype(float)).any():
            # Delete row if it contains NaN
            data = np.delete(data, i, axis = 0)
            num_rows -= 1
        else:
            i += 1

    return data

def get_tier_data(df, tier): # tier is a string
    """Keeps only the specified player tier in data"""

    df_tier = df[df["tier"] == tier]
    print('The size of our unclean data array is', np.shape(df_tier.to_numpy()[:, 1:]))

    data = remove_bad_rows(df_tier)
    num_data_rows, num_data_cols = np.shape(data)
    feature_clean = data[:, 1:(num_data_cols - 1)]
    target_clean = data[:, num_data_cols - 1].astype(int)
    print('The size of our cleaned data array is', np.shape(data))

    return feature_clean, target_clean

def tt_split(feature_clean, target_clean):
    """Executes a typical test train split"""
    feature_train, feature_test, target_train, target_test = train_test_split(feature_clean, target_clean, stratify = target_clean, random_state = 0)

    return feature_train, feature_test, target_train, target_test

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape = (349, ))) #2019
    model.add(tf.keras.layers.Dense(50, activation = 'relu'))
    model.add(tf.keras.layers.Dense(25, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation = 'tanh'))
    model.add(tf.keras.layers.Dense(2))


    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])

    return model




df = loadDataAndTargets()

iron_feature, iron_target = get_tier_data(df, 'IRON')

iron_X_train, iron_X_test, iron_y_train, iron_y_test = tt_split(iron_feature, iron_target)

scaler = StandardScaler()
scaler.fit(iron_X_train)

iron_X_train = tf.convert_to_tensor(iron_X_train, dtype = tf.float32)
iron_y_train = tf.convert_to_tensor(iron_y_train, dtype = tf.float32)
iron_X_test = tf.convert_to_tensor(iron_X_test, dtype = tf.float32)
iron_y_test = tf.convert_to_tensor(iron_y_test, dtype = tf.float32)

clf = KerasClassifier(build_fn = create_model, 
                                 epochs = 25,  
                                 verbose = 2)

pipe = make_pipeline(StandardScaler(), clf)

pipe.fit(iron_X_train, iron_y_train)
print('Train Score: ', pipe.score(iron_X_train, iron_y_train))
print('Test Score: ', pipe.score(iron_X_test, iron_y_test))