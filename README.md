# rapi_ML
this is a repo from the ML model developed with test purpose with sample data. this repo contains:
- train.py:  which is the training and experimental model using random forest and SVM
- inference.py: which is the inferencial way to reproduce the trained model for testing porpuse
- dockerfil: the necessary setting of the docker image for usability
- inference_rfc.joblib : trained weigths of the model for inferences and API 
- app.py: REST API source code for deploy purpose we are using mainly FLASK
- dataset(csv files):
  - datase (train data for develop)
  - orders_test (test data for develp)
  
## instalation dependencies
* we are using virtual env fron anaconda with ubuntu 20.4 LTS
* and mainly python 3.7
* other specific libs may you require
*
```
numpy 1.21.6
pandas 0.25.0
sklearn 0.21.2
joblib 0.13.2
pickl4 4.0
imblearn 0.5.0
```
## run
* train (in your virtual env)

```
python train.py
```

* inferences (in your virtual env)

```
python inference.py
```

* api (first for testing you should install [postmant](https://www.postman.com/) and run app.py)
* first run
```
python app.py
```

-  in the POST : http: //127.0.0.1:12345 /prediction

-  body --> raw (JSON)
-  postman sample input
```
[
{"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5600,"created_at":"2017-09-17T14:36:51Z"},
{"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5500,"created_at":"2017-09-17T14:36:51Z"},
{"order_id":"14924129", "store_id":"900007722","to_user_distance":2.188069 ,"to_user_elevation":43.430176,"total_earning":5500,"created_at":"2017-09-17T14:36:51Z"}
]
```
- postman output
```
{
  "prediction": "[1, 1, 1]"
}
```

* docker image (ubuntu 20.4)

```
sudo docker run docker-rcl-model python3 inference.py
```
 
