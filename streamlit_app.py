import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import streamlit as st

st.set_page_config(layout='wide')

st.title('Gold Price Prediction')
st.write('This app predicts the price of Gold')

gold_data = pd.read_csv('gld_price_data.csv')
gold_data['Date'] = pd.to_datetime(gold_data['Date']) 

# Calculate correlations, excluding non-numeric columns
correlation = gold_data.corr(numeric_only=True)
st.subheader('Correlation')
fig = plt.figure(figsize= (8,8))
sns.heatmap(correlation,cbar=True, square= True , fmt = '.2f', annot = True, annot_kws={'size':8}, cmap = 'Blues')
sns.displot(gold_data['GLD'], color = 'red') 
st.pyplot(fig)

X = gold_data.drop(['Date','GLD'], axis = 1)
Y = gold_data['GLD']

X_train, X_test, Y_train , Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_train,Y_train)

test_data_prediciton = regressor.predict(X_test)

error_score = metrics.r2_score(Y_test, test_data_prediciton)

Y_test = list(Y_test)

fig = plt.figure(figsize= (15,10))
plt.plot(Y_test, color = 'blue' , label = 'Actual Value')
plt.plot(test_data_prediciton, color = 'green', label = 'Predicted Value')
plt.title('Actual Price Vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
st.pyplot(fig)


