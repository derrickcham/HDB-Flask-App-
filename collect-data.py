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

app = Flask(__name__)

@app.route('/')
def flask_homepage():
	return(render_template('test.html'))

@app.route('/collect_data', methods = ['POST'])
def collect_data():
	error_messages_list = []

	try: 
		fa_sqm = float(request.form['fa_sqm'])
		print('FLoor area sqm = ' + str(fa_sqm))

	except 	ValueError:
		fa_sqm = None
		error_messages_list.append('Floor Area needs to be numeric')

	try:
		flat_age= int(request.form['flat_age'])
		print('Flat Age = ' + str(flat_age))

	except ValueError:
		flat_age = None
		error_messages_list.append('Flat age needs to be numeric')

	flat_type = request.form['flat_type']

	if len(error_messages_list) != 0:
		for error in error_messages_list:
			return(error)
	
	else: 

		prediction = hello_world.model_predict(fa_sqm, flat_age, flat_type)
		print('resale_value' + str(prediction))
		return(render_template('index.html', value = prediction))

if __name__ == '__main__':
	app.debug = True
	port = int(os.environ.get("PORT", 5000))
	app.run(host='0.0.0.0', port=port)