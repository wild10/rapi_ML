# errol mamani : 2022
# rappi picker model

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

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

def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled

def inference():

    path = os.getcwd()
    print("path location : {} \n".format(path) )
    output_path = os.path.join(path,'output.csv')
    print(output_path,"\n")

    # Load , read test data
    test_link = "./orders_test.csv"
    df = pd.read_csv(test_link)
    print("size of data: {}".format(df.shape))



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

    # DATA NORMALIZATION and creat X
    X = maximum_absolute_scaling(X).to_numpy()

    # LOAD & RUN MODEL
    # rfc = load('Inference_rfc.joblib')
    filename = 'Inference_rfc.joblib'
    rfc = pickle.load(open(filename, 'rb'))

    print("Random forest Classifier")
    y_pred = rfc.predict(X)
    print("sample of first 30: {} \n".format( y_pred[:30]))
    # save test in a file
    print(output_path)
    # X = pd.DataFrame(X)
    df["predict"] = pd.DataFrame(y_pred)
    print(df.head())
    df.to_csv('./output.csv')

if __name__=='__main__':
    inference()
