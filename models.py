#!/usr/bin/env python

"""This file generates, trains, and runs a keras neural network on GDELT data"""

import pandas as pd
import tensorflow as tf
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import sklearn.preprocessing as pre
from keras.utils import to_categorical

# constants
dir = os.path.dirname(__file__)
prediction_results_filename = os.path.join(dir, '..','data','prediction_results','prediction_results_nn.csv')
model_h5 = os.path.join(dir, 'model.h5')
dir = os.path.dirname(__file__)
training_data_filename = os.path.join(dir, '..','data','gdelt_abbrv.csv')

batch_size = 512
epochs = 300
cols=['id','Date','Source','Target','CAMEOCode','NumEvents','NumArts','QuadClass','Goldstein','SourceGeoType',
      'SourceGeoLat','SourceGeoLong','TargetGeoType','TargetGeoLat','TargetGeoLong','ActionGeoType','ActionGeoLat',
      'ActionGeoLong']


# clean dat
def data_prep(self, df):
    return df.fillna(0)


# Load and compile a NN model.
def loadCompileModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(16, )))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.50))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.50))
    model.add(tf.keras.layers.Dense(211, activation='softmax'))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

  # This version of fit takes a validation set. Use in tandem with split method.
def fitModel(x_train, x_val, y_train, y_val):
    model = loadCompileModel()

    history =model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return model

# Split a given data set into both training and testing sets. Useful when tinkering with model.
def splitTraining(training_data):

    # Get label col for classifier
    Y = training_data["CAMEOCode"]
    # hot = to_categorical(np.array(Y))
    # drop label col so we don't train on it.
    training_data = training_data.drop('CAMEOCode', axis=1)
    # normalize
    x = training_data.values  # returns a numpy array
    min_max_scaler = pre.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    training_data = pd.DataFrame(x_scaled)
    # Get list of features.
    features = list(training_data.columns)
    X = training_data[features]
    (x_train, x_val, y_train, y_val) = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)
    return x_train, x_val, y_train, y_val


def generate_model_split_nn(training_data):
    training_data.columns = cols
    training_data = training_data.drop('id', 1)
    X, x, Y, y = splitTraining(training_data)
    return fitModel(X, x, Y, y), X, x, Y, y

def generate_model_split(training_data=None):
    # Get label col for classifier
    Y = training_data["CAMEOCode"]

    # drop label col so we don't train on it.
    training_data = training_data.drop('CAMEOCode', axis=1)

    # Get list of features.
    features = list(training_data.columns)
    #  print("* features:", features, sep="\n")
    X = training_data[features]
    dt = dt_construct()
    # use sklearn's train_test_split for accuracy testing.
    # I tested this out a number of different ways. Gini always provided the best results.
    # We may want to lower max depth to 3 in practice to avoid overfitting, but this is the model
    # that gets me the best accuracy with this data.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
    # fit and predict with accuracy report.
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return y_pred.ravel(), y_test

def generate_decision_tree(self, new_tree=True, show=True, validate=True, training_data=None,
                           verbose=False, grid=False):
    """Create a decision tree using sklearn and data_prep method."""
    if not new_tree:
        return self.load_pickle_model()

    if training_data is None:
        training_data = self.loadFile(training_data_filename)

    training_data = self.data_prep(training_data)
    # Get label col for classifier
    Y = training_data["Label"]
    print("Value counts: " + str(Y.value_counts()))
    # drop label col so we don't train on it.
    training_data = training_data.drop('Label', axis=1)

    # Get list of features.
    features = list(training_data.columns)
    # print("* features:", features, sep="\n")
    X = training_data[features]
    dt = self.dt_construct()
    if grid:
        self.grid_search(X, Y)
        return
    elif validate:
        # use sklearn's train_test_split for accuracy testing.
        # I tested this out a number of different ways. Gini always provided the best results.
        # We may want to lower max depth to 3 in practice to avoid overfitting, but this is the model
        # that gets me the best accuracy with this data.

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=66)
        X_train = pd.DataFrame(X_train, columns=training_data.columns)
        if verbose:
            eval_set = [(X_train, y_train), (X_test, y_test)]
            eval_metric = ["auc", "error"]
            dt.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
            plt.plot(dt.evals_result()['validation_0']['error'])
            plt.ylabel('Error')
            plt.show()
        else:
            dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)

    else:
        dt.fit(X, Y)

    # Show me the tree
    if show:
        # plot single tree
        plot_tree(dt)
        plt.show()
    self.pickle_model(dt)
    return dt

def generate_model_direct(self, X, Y):
    dt = self.dt_construct()
    # sm = SMOTE(random_state=12, ratio=1)
    # X, Y = sm.fit_sample(X, Y)
    dt.fit(X, Y)
    self.pickle_model(dt)
    return dt

def grid_search(self, X, Y):
    params = {
        'min_child_weight': [1, 5, 10],
        'learning_rate': [.1, .3, .5, .6, .7, .8],
        'gamma': [0, 1, 5],
        'subsample': [0.5, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 150, 200, 250, 300]
    }
    boost = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                        silent=True, nthread=1)
    folds = 3
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1009)
    grid = GridSearchCV(estimator=boost, param_grid=params, n_jobs=-4, verbose=3)
    random_search = grid.fit(X, Y)
    #random_search = RandomizedSearchCV(boost, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
    #                                   n_jobs=4, cv=skf.split(X, Y), verbose=3, random_state=1009)

    # Here we go
    random_search.fit(X, Y)
    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-grid-search-results-01.csv', index=False)

def dt_construct():
    return xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
          colsample_bynode=1, colsample_bytree=0.5, gamma=5,
          learning_rate=0.1, max_delta_step=0, max_depth=3,
          min_child_weight=10, missing=None, n_estimators=100, n_jobs=1,
          nthread=1, random_state=0,
          reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
          silent=True, subsample=0.6, verbosity=1)

def xgboost():
    gdelt_df = pd.read_csv('data/gdelt_encoded.csv', index_col=0)
    label_df = gdelt_df['CAMEOCode']
    pl_xgb = Pipeline(steps=
                      [('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
    scores = cross_val_score(pl_xgb, gdelt_df, label_df, cv=2)
    print('Accuracy for XGBoost Classifier : ', scores.mean())

def get_abbrv():
    df = pd.read_csv('data/gdelt.csv', delim_whitespace=True)
    df_abbrv = df.head(100000)
    df_abbrv.to_csv('data/gdelt_abbrv.csv')

def normalize_and_encode(filename):
    le = preprocessing.LabelEncoder()
    # FIT AND TRANSFORM
    # use df.apply() to apply le.fit_transform to all columns
    for chunk in pd.read_csv(filename, delim_whitespace=True, chunksize=1000000):
        try:
            X_2 = chunk.apply(le.fit_transform)
            pd.DataFrame(X_2).to_csv('data/gdelt_encoded_full.csv',  mode='a', header=False)
        except TypeError:
            pass

def main():
    normalize_and_encode('data/gdelt.csv')
    generate_model_split_nn(pd.read_csv('data/gdelt_encoded_full.csv',low_memory=False).head(1000000))
    # generate_model_split(pd.read_csv('data/gdelt_encoded.csv', index_col=0))


if __name__ == "__main__": main()