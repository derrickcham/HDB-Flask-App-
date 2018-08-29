#import packages

import pandas as pd
import numpy as np

#read csv files 
first = pd.read_csv('C:/Users/Derrick/Desktop/HDB/resale-flat-prices-based-on-approval-date-1990-1999.csv')
second = pd.read_csv('C:/Users/Derrick/Desktop/HDB/resale-flat-prices-based-on-approval-date-2000-feb-2012.csv')
third = pd.read_csv('C:/Users/Derrick/Desktop/HDB/resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv')
fourth = pd.read_csv('C:/Users/Derrick/Desktop/HDB/resale-flat-prices-based-on-registration-date-from-jan-2015-onwards.csv')

#combine the datasets together
listy= [second, third, fourth]
combined = first.append(listy)
combined.head()

#create flat age
combined['month'] = pd.to_datetime(combined['month'])
combined['lease_commence_date'] = pd.to_datetime(combined['lease_commence_date'])
combined['flat_age'] = combined['month'] - combined['lease_commence_date']
combined['flat_age'] = combined['flat_age'].dt.days/365

#more clean up. In flat_type, there are two labels for the same category 'MULTI-GENERATION'. They have to be standardised.
combined['flat_type'] = combined['flat_type'].replace({'MULTI GENERATION':'MULTI-GENERATION'})

#dropping unnecessary variables
combined.drop(['block', 'street_name', 'remaining_lease'], axis = 1, inplace= True)

#we are selecting only numerical columns
numerical_only  = combined._get_numeric_data().columns
X = combined[numerical_only]
Y = X['resale_price']
X.drop('resale_price', axis = 1, inplace = True)

#Chosen set, combined 
select = pd.DataFrame() 
select['floor_area_sqm'] = 31
select['flat_age'] = 25
select.set_value(0, 'floor_area_sqm', 31)
select.set_value(0, 'flat_age', 25)

#creating the model 
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=0, verbose = 3,  n_estimators =50)
print(X)
print(Y)

#fit the model 
regr.fit(X, Y) 

#predict on the selection
y_pred = regr.predict(select)

print(y_pred)