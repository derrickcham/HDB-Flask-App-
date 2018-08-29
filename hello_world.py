from flask import Flask, request, render_template
import pickle
from sklearn.ensemble import RandomForestRegressor
import os
import hello_world
import json
import pandas as pd
import numpy as np
import requests
import sys


colnames = ['floor_area_sqm', 'flat_age', '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']

def model_predict(sqm_s,agey,flat_type_catty):
    with open(os.getcwd()+ '/rf_pickled_model.pkl','rb') as file_handler:
        fitted_model = pickle.load(file_handler)
    flat_type_discretize = str(flat_type_catty)
    dictionary = {'floor_area_sqm' : [float(sqm_s)], 'flat_age': [int(agey)], flat_type_discretize: [1]}
    diff = np.setdiff1d(colnames,list(dictionary.keys()))
    df = pd.DataFrame.from_dict(dictionary)
    for col in diff:
        df.loc[0,col] = int(0)
        print(df)
    df = df[colnames]
    prediction = fitted_model.predict(df.loc[0].values.reshape(1,-1))
    return(prediction[0])

