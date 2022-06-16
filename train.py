# train.py
# Errol Mamani 2022
import os
import sys;print("Python", sys.version)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from imblearn.under_sampling import RandomUnderSampler # under sampling module
from sklearn.model_selection import train_test_split # function to splid dataset
from sklearn.ensemble import RandomForestClassifier # randon forest function sklearn

from sklearn.metrics import accuracy_score  # METRIC accuracy
from sklearn.metrics import classification_report  # METRIC f1 score imbaled data
from sklearn.svm import SVC # cuadrati svm
from sklearn.svm import LinearSVC # linear cudratic
from joblib import dump # save model
import pickle

# apply the maximum absolute scaling in Pandas using the .abs() and .max() methods
def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled

def train():
    # load , read and normaliza dataset
    dataset_link = "./datase.csv"
    df = pd.read_csv(dataset_link)
    print("size of data: {}".format(df.shape))
    # print(df.head(5) )

    #creating numerical id for store_id
    df["oder_id_numerical"] = df["store_id"].rank(method="dense").astype(int)
    # print(df.head(5) )
    # creating numerical data from timestamp
    df["created_at"]= pd.to_datetime( df["created_at"] ) # cast obj--> datatime
    # df["month"] = df["created_at"].dt.month.fillna(0)
    df["day"] = df["created_at"].dt.day.fillna(0)
    df["hour"] = df["created_at"].dt.hour.fillna(0)
    # print(df.head())

    # selecting util data to feed into the model
    X = df[["to_user_distance","to_user_elevation", "total_earning","oder_id_numerical","hour"]]
    print(X.head())

    # DATA NORMALIZATION and creat X, y
    X = maximum_absolute_scaling(X).to_numpy()
    y = df["taken"].to_numpy()
    print(X[:5])
    print(y[:5] )

    # UNDER_SAMPLING DATASET
    rus = RandomUnderSampler(random_state=42) # instance under_sampling
    X_resampled, y_resampled = rus.fit_resample(X, y)
    #show re sampled dataset
    print("X_res: {}  | y_res: {}".format(X_resampled.shape, y_resampled.shape))

    # split dataset
    X = X_resampled
    y = y_resampled
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)
    print("x_train: {} | y_train: {}".format(X_train.shape, y_train.shape))

    # TRAINING RANDON FOREST
    # #inicializar
    rfc = RandomForestClassifier(random_state=0)
    # entrenar el modelo
    rfc.fit(X_train, y_train)
    # predecir el resultado
    y_pred = rfc.predict(X_test)

    # svc=SVC() #Default hyperparameters
    # svc.fit(X_train,y_train)
    # y_pred=svc.predict(X_test)
    print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    # mostrar metrica mas precisa para classificacion binaria
    target_names = ['class 0', 'class 1']

    print(classification_report(y_test, y_pred, target_names=target_names))

    # SAVE MODEL
    # dump(rfc, 'Inference_rfc.joblib')
    filename = 'Inference_rfc.joblib'
    pickle.dump(rfc, open(filename, 'wb'))


if __name__== "__main__":
    train()
