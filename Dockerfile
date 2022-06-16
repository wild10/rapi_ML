FROM jupyter/scipy-notebook

RUN pip install joblib
RUN conda install -c conda-forge imbalanced-learn
RUN pip install pickle4


COPY datase.csv ./datase.csv
COPY orders_test.csv ./orders_test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py
