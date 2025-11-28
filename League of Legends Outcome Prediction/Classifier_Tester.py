import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # for making palettes

# Common utilities
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def loadDataAndTargets():
    """Reads and loads the data csv file"""

    df = pd.read_csv("Final_data_w.o_base_stats2.csv")
    df.columns.values[0] = "index"
    df = df.drop(['index'], axis = 1)
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

def plot_1par_grid_search(cv_results, grid_param_1, name_param_1, show_train = 1):
    """Plot GridCV results for one parameter searches"""

    val_scores = np.array(cv_results['mean_test_score'])
    train_scores = np.array(cv_results['mean_train_score'])

    _, ax = plt.subplots(1, 1, figsize = (10, 4))
    cmap = plt.get_cmap("tab10")

    ax.plot(grid_param_1, val_scores, '-o', label = f"Validation")
    if show_train: ax.plot(grid_param_1, train_scores, '-s',label = f"Train")

    ax.set_title("Grid Search Scores")
    ax.set_xlabel(name_param_1)
    ax.set_ylabel('CV Average Score')
    ax.legend(loc = 'lower right')
    ax.grid('on')

    return

def opt_depth_finder(lowest_depth, highest_depth, feature_clean, target_clean): # This function takes around 30 mins to run
    """Determines the optimal depth for RandomForestClassifier"""

    estimators = [('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state = 0)) ]
    pipe = Pipeline(estimators)

    # Grid Search Parameters
    DEPTHS = np.arange(lowest_depth, highest_depth)
    params = {'clf__max_depth':DEPTHS}

    grid_clf = GridSearchCV(pipe, param_grid = params, return_train_score = True)
    grid_clf.fit(feature_clean, target_clean)

    plot_1par_grid_search(grid_clf.cv_results_, DEPTHS, 'max_depth')  # only works for 1 parameter
    plt.axvline(DEPTHS[grid_clf.best_index_], color = 'black', lw = 2, alpha = 0.5) # draw black line behind best params

    print('Best parameters:', grid_clf.best_params_)
    print(f"Best Validation Score:\t{grid_clf.best_score_:0.3f}")

    return

# For K Nearest Neighbors -- find optimal k value
def plot_2par_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, show_train = 1):
    """Plot GridCV results for two parameter searches"""

    val_scores = np.array(cv_results['mean_test_score']).reshape(len(grid_param_1), len(grid_param_2)) # FIXED A BUG HERE
    train_scores = np.array(cv_results['mean_train_score']).reshape(len(grid_param_1), len(grid_param_2))

    _, ax = plt.subplots(1, 1, figsize = (10, 4))
    cmap = plt.get_cmap("tab10")

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
      ax.plot(grid_param_1, val_scores[:, idx], '-o',  color = cmap(idx), label = f"Validation: {name_param_2} = {str(val)}") # FIXED
      if show_train: ax.plot(grid_param_1, train_scores[:, idx], ':.', color = cmap(idx), label = f"Train: {name_param_2} = {str(val)}")

    ax.set_title("Grid Search Scores")
    ax.set_xlabel(name_param_1)
    ax.set_ylabel('CV Average Score')
    ax.legend(loc = 'upper right')
    ax.grid('on')

    return

def opt_k_finder(lowest_k, highest_k, feature_clean, target_clean):
    """Finds the optimal k value for KNearestNeighborsClassifier"""
    estimators = [('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]
    pipe = Pipeline(estimators)

    # Grid Search Parameters
    KLIST = np.arange(lowest_k, highest_k)
    WLIST = ['uniform', 'distance']
    params = {'clf__n_neighbors':KLIST, 'clf__weights':WLIST}

    grid_clf = GridSearchCV(pipe, param_grid=params, return_train_score = True)
    grid_clf.fit(feature_clean, target_clean)

    plot_2par_grid_search(grid_clf.cv_results_, KLIST, WLIST, 'n_neighbors', 'weight')  # only works for 2 parameters

    print(grid_clf.best_params_)
    print(f"Best Validation Score:\t{grid_clf.best_score_:0.3f}")

    return

def tt_split(feature_clean, target_clean):
    """Executes a typical test train split"""
    feature_train, feature_test, target_train, target_test = train_test_split(feature_clean, target_clean, stratify = target_clean, random_state = 0)

    return feature_train, feature_test, target_train, target_test



df = loadDataAndTargets()

iron_feature, iron_target = get_tier_data(df, 'IRON')

iron_X_train, iron_X_test, iron_y_train, iron_y_test = tt_split(iron_feature, iron_target)

scaler = StandardScaler()
scaler.fit(iron_X_train)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth = 8, random_state = 0))
#pipe = make_pipeline(StandardScaler(), neural_network.MLPClassifier(hidden_layer_sizes = ((50, 50)), max_iter = 500, random_state = 5))
#pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 9, weights = 'distance'))
#pipe = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators = 1000, learning_rate = 1.0, max_depth = 50, random_state = 0, tol = 1e-10))
pipe.fit(iron_X_train, iron_y_train)
print('Train Score: ', pipe.score(iron_X_train, iron_y_train))
print('Test Score: ', pipe.score(iron_X_test, iron_y_test))

iron_y_pred = pipe.predict(iron_X_test)
print(classification_report(iron_y_test, iron_y_pred))




# Tensor flow stuff
#def build_model(hp):
#    model = keras.Sequential()
#    model.add(keras.layers.Dense(
#        hp.Choice('units', [8, 16, 32]),
#        activation = 'relu'))
#    model.add(keras.layers.Dense(1, activation = 'relu'))
#    model.compile(loss = 'mse', metrics = ['accuracy'])

#    return model

# Convert arrays to tensors
#iron_X_train = tf.convert_to_tensor(iron_X_train, dtype = tf.float32)
#iron_y_train = tf.convert_to_tensor(iron_y_train, dtype = tf.float32)
#iron_X_test = tf.convert_to_tensor(iron_X_test, dtype = tf.float32)
#iron_y_test = tf.convert_to_tensor(iron_y_test, dtype = tf.float32)

#tuner = keras_tuner.RandomSearch(build_model, objective = 'val_accuracy', max_trials = 50)
#tuner.search(iron_X_train, iron_y_train, epochs = 10, validation_data = (iron_X_test, iron_y_test))
#best_model = tuner.get_best_models()[0]

#print(tuner.get_best_hyperparameters(num_trials = 1)[0])