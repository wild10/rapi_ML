
# errol mamani
# " 2022"
from sklearn.externals import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier # randon forest function sklear

from flask import Flask, request, jsonify # flas to do REST API
from sklearn.externals import joblib # for load the model

import traceback

application = Flask(__name__)

@application.route('/prediction', methods=['POST'])


#POST FUCTION
def predict():

    if lr:

        try:
            json_ = request.json
            print(json_)
            # df = pd.get_dummies(pd.DataFrame(json_))
            df = pd.DataFrame(json_)
            # print(df)
            print("here")
            df = df.reindex(columns=["order_id","store_id","to_user_distance","to_user_elevation", "total_earning", "created_at"], fill_value=0)

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
            df_scaled = X.copy()
            # apply maximum absolute scaling
            for column in df_scaled.columns:
                df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()

            X = df_scaled
            # print(query)
            predict = list(lr.predict(X))
            # predict = list([1])

            return jsonify({'prediction': str(predict)})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:

        print ('Model not good')
        return ('Model is not good')



if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    lr = joblib.load("Inference_rfc.pkl")
    print('Model loaded')

    application.run(port=port, debug=True)


#  SIMPLE EXAMPLE TO TEST REST API
# RUN POSTMAN query
# http://127.0.0.1:12345/prediction
# run python app.py
# simple input POSTMANT raw --> JSON
# [
# {"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5500,"created_at":"2017-09-17T14:36:51Z"}
# ]

#largers batch
# [
# {"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5600,"created_at":"2017-09-17T14:36:51Z"},
# {"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5500,"created_at":"2017-09-17T14:36:51Z"},
# {"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5500,"created_at":"2017-09-17T14:36:51Z"}
#
# ]
