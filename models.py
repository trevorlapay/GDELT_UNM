#!/usr/bin/env python

"""This file generates, trains, and runs a keras neural network on GDELT data"""

import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import sklearn.preprocessing as preprocessing
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import sklearn.preprocessing as pre
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import keras as keras
from keras.models import Sequential
from keras.utils import to_categorical

# constants
dir = os.path.dirname(__file__)
prediction_results_filename = os.path.join(dir, '..','data','prediction_results','prediction_results_nn.csv')
model_h5 = os.path.join(dir, 'model.h5')
dir = os.path.dirname(__file__)
training_data_filename = os.path.join(dir, '..','data','gdelt_abbrv.csv')

look_back = 50
batch_size = 32896*2*2*2
epochs = 20
cols=['Source','Target','CAMEOCode','NumEvents','NumArts','SourceGeoType',
      'TargetGeoType', 'ActionGeoType','SourceGeoLatInt','TargetGeoLatInt', 'ActionGeoLatInt','Month']
# 1 = statement_positive

def data_prep(df):
    '''remove fields that directly correlate to label'''
    try:
        df = df.drop('QuadClass', axis=1)
    except:
        pass
    try:
        df = df.drop('Goldstein', axis=1)
    except:
        pass
    return df


def loadCompileModel():
    '''Load and compile a NN model.'''
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(11  , )))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.50))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.50))
    model.add(tf.keras.layers.Dense(23, activation='softmax'))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


def fitModel(x_train, x_val, y_train, y_val):
    '''This version of fit takes a validation set. Use in tandem with split method.'''
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
    serialize(model)
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
    X, x, Y, y = splitTraining(training_data)
    return fitModel(X, x, Y, y), X, x, Y, y


def  generate_model_split(training_data=None):
    # Get label col for classifier
    Y = training_data["CAMEOCode"]

    # drop label col so we don't train on it.
    training_data = training_data.drop('CAMEOCode', axis=1)

    # Get list of features.
    features = list(training_data.columns)
    #  print("* features:", features, sep="\n")
    X = training_data[features]
    X = data_prep(X)
    dt = dt_construct()
    # use sklearn's train_test_split for accuracy testing.
    # I tested this out a number of different ways. Gini always provided the best results.
    # We may want to lower max depth to 3 in practice to avoid overfitting, but this is the model
    # that gets me the best accuracy with this data.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
    # fit and predict with accuracy report.
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return y_pred.ravel(), y_test


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
    gdelt_df = load_file().head(50000000)
    label_df = gdelt_df['CAMEOCode']
    gdelt_df.drop(['CAMEOCode'], 1)
    pl_xgb = Pipeline(steps=
                      [('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
    scores = cross_val_score(pl_xgb, gdelt_df, label_df, cv=2)
    print('Accuracy for XGBoost Classifier : ', scores.mean())


def get_abbrv():
    df = pd.read_csv('data/gdelt.csv', delim_whitespace=True,
                     low_memory=False, dtype={0: 'str', 1: 'str', 2: 'str',
                                              3: 'str', 4: 'int32', 5: 'int8',
                                              6: 'float64', 7: 'float64', 8: 'str',
                                              9: 'float64', 10: 'float64', 11: 'float64',
                                              12: 'float64', 13: 'float64', 14: 'float64',
                                              15: 'float64', 16: 'float64', 17: 'float64'})
    df_abbrv = df.head(100000)
    df_abbrv.to_csv('data/gdelt_abbrv.csv', index=False)


def encode():
    df = pd.read_csv('data/gdelt.csv',  chunksize=20000000, delim_whitespace=True,
                     low_memory=False, dtype={0: 'str', 1: 'str', 2: 'str',
                                              3: 'str', 4: 'int32', 5: 'int8',
                                              6: 'float64', 7: 'float64', 8: 'str',
                                              9: 'float64', 10: 'float64', 11: 'float64',
                                              12: 'float64', 13: 'float64', 14: 'float64',
                                              15: 'float64', 16: 'float64'})
    df_transform = pd.DataFrame()
    le = preprocessing.LabelEncoder()

    for chunk in df:
        # We do a little feature swapping. Note that 42 = 20 (there are no examples in 20)
        # We do this to spread out the values in the 4-level label, since it is unbalanced
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('042', '20')
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('043', '21')
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('046', '22')
        # There are not enough values in the dataset to justify these, so I move them to the end of the
        # feature set. THey get removed on read.
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('044', '51')
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('041', '52')
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('045', '53')
        chunk['CAMEOCode'] = chunk['CAMEOCode'].replace('020', '54')
        chunk['CAMEOCode'] = chunk['CAMEOCode'].astype(str).str[:2]
        chunk['CAMEOCode'] = pd.to_numeric(chunk.CAMEOCode, errors='coerce', downcast='integer').fillna(0).astype(int)
        chunk['QuadClass'] = pd.to_numeric(chunk.QuadClass, errors='coerce', downcast='integer')
        chunk['Goldstein'] = pd.to_numeric(chunk.Goldstein, errors='coerce', downcast='integer')
        chunk['SourceGeoType'] = pd.to_numeric(chunk.SourceGeoType, downcast='integer').fillna(0).astype(int)
        chunk['TargetGeoType'] = pd.to_numeric(chunk.TargetGeoType, downcast='integer').fillna(0).astype(int)
        chunk['ActionGeoType'] = pd.to_numeric(chunk.ActionGeoType, downcast='integer').fillna(0).astype(int)
        chunk['SourceGeoLatInt'] = pd.to_numeric(chunk.SourceGeoLat, downcast='integer').fillna(0).astype(int)
        chunk['TargetGeoLatInt'] = pd.to_numeric(chunk.TargetGeoLat, downcast='integer').fillna(0).astype(int)
        chunk['ActionGeoLatInt'] = pd.to_numeric(chunk.ActionGeoLat, downcast='integer').fillna(0).astype(int)
        chunk['Month'] = chunk['Date'].astype(str).str[4:6].astype(int)
        chunk['Year'] = chunk['Date'].astype(str).str[0:4].astype(int)
        chunk['Source'] = le.fit_transform(chunk['Source'])
        chunk['Target'] = le.fit_transform(chunk['Target'])

        df_transform = pd.concat([df_transform, chunk])
    pd.DataFrame(df_transform).to_csv('data/gdelt_encoded_full.csv', index=False)


# Serialize weights.
def serialize(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved to disk")

def plot_selectbest(df):
    y = df['CAMEOCode']
    X = df.drop(['CAMEOCode'], 1)
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(X, y)
    x_val = X.columns

    plt.bar(x_val, selector.scores_, color='b')
    plt.xticks(x_val, x_val, rotation='vertical')
    plt.tight_layout()
    plt.show()


def plot_cameo_hist(df):
    # matplotlib histogram
    plt.hist(df['CAMEOCode'], color='blue', edgecolor='black',
             bins=np.arange(24) - 0.5, label=range(1, 23, 1))
    # Add label
    plt.title('Label Histogram')
    plt.xlabel('Cameo Code')
    plt.ylabel('# of Examples')
    plt.xticks(np.arange(1, 23, 1))
    plt.xlim([-1, 23])
    plt.ticklabel_format(style='plain')
    plt.show()


def plot_correlation(df):
    plt.matshow(df.corr())
    plt.show()

def print_unique_label(df):
    print(str(df.CAMEOCode.unique()))

def load_file():
    '''return a manageable encoded dataframe from file'''
    mylist = []

    for chunk in pd.read_csv('data/gdelt_encoded_full.csv', usecols=cols, sep=',', chunksize=20000, dtype={0: 'int16',
                                                                            1: 'int16', 2: 'int16',
                                                                            3: 'int16', 4: 'int16', 5: 'float64',
                                                                            6:'int16', 7:'int8', 8:'int8', 9:'int8', 10: 'int16'}):
        mylist.append(chunk)
    big_data = pd.concat(mylist, axis=0)
    # Drop fishy cameocodes that are out of bounds.
    indexNames = big_data[big_data['CAMEOCode'] > 22].index
    big_data.drop(indexNames, inplace=True)
    return data_prep(big_data)

def lstm_data_build():
    df = load_file()
    df_transform = pd.DataFrame()
    # 11263 = USA
    df_transform = df.loc[df['Source'] == 11263]
    df_transform.to_csv('gdelt_usa.csv')
    print(str(len(df_transform)))
    features_set = []
    labels = []
    for i in range(look_back, len(df_transform)):
        features_set.append(df_transform[i - look_back:i].values)
        labels.append(df_transform.iloc[[i]]['CAMEOCode'].values)
    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 12))
    np.save('np_lstm_feat.npy', features_set)
    np.save('np_lstm_labels.npy', labels)

def lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 12)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(units=50,  return_sequences=False))
    model.add(Dropout(0.5))

    # model.add(Dense(units = 1))
    model.add(Dense(23, activation='softmax'))
    # Compile the model
    model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def lstm_fit():
    model = lstm_model()
    features_set = np.load('np_lstm_feat.npy')
    labels = np.load('np_lstm_labels.npy')

    X_test = features_set[-300:]
    y_test = labels[-300:]
    history = model.fit(features_set, labels, epochs=200, batch_size=32)
# Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    serialize(model)
    return model

def main():
    # encode()
    #lstm_data_build()
    lstm_fit()
    # plot_cameo_hist(df)
    # plot_selectbest(df)
    # plot_correlation(df)
    # xgboost()
    # generate_model_split(df.head(40000000))


if __name__ == "__main__": main()