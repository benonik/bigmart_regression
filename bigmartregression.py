# Import libraries
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error,accuracy_score

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# import data
train_data = pd.read_csv('big_mart_train.csv')
test_data = pd.read_csv('big_mart_test.csv')


"""Deal With Missing Data
The missingno library provides a neat way to showcase which variables have
missing data. This is done below using a bar chart. I will then proceed to use
Pandas fillna method to fill the two columns that have missing data (Item_Weight, Outlet_Size)
"""
msno.bar(train_data)
msno.bar(test_data)

train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(),inplace = True)
test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(),inplace = True)

outlet_size_tr = train_data['Outlet_Size']
outlet_size_ts = test_data['Outlet_Size']
imputer1 = CategoricalImputer()
outlet_size_tr = imputer1.fit_transform(outlet_size_tr)
outlet_size_ts = imputer1.fit_transform(outlet_size_ts)

train_data = train_data.drop(['Outlet_Size'], axis = 1)
train_data.insert(8, 'Outlet_Size', outlet_size_tr)


test_data = test_data.drop(['Outlet_Size'], axis = 1)
test_data.insert(8, 'Outlet_Size', outlet_size_ts)


# Let's see if there are any columns we can drop

cor = train_data.corr()
cor["Item_Outlet_Sales"].sort_values(ascending = False)

# The year that an outlet was established has a very low correlation figure

train_data = train_data.drop(['Outlet_Establishment_Year'], axis = 1)
test_data = test_data.drop(['Outlet_Establishment_Year'], axis = 1)

train_data = train_data.replace(to_replace="LF", value = "Low Fat")
test_data = test_data.replace(to_replace="LF", value = "Low Fat")
train_data = train_data.replace(to_replace="reg", value = "Regular")
test_data = test_data.replace(to_replace="reg", value = "Regular")

""" Separating the training and testing datasets into their respective independent
    and dependent variables """

X_train = train_data.iloc[:, [1,2,3,4,5,7,8,9]].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:,  [1,2,3,4,5,7,8,9]].values

""" Now let's deal with categorical data """


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,3,5,6,7])], remainder='passthrough')
X_train = ct.fit_transform(X_train).toarray()
X_test = ct.fit_transform(X_test).toarray()

""" Feature Scaling """

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


""" PCA """
# how many PCA components do we need?
pca = PCA(n_components = 25)
pca.fit_transform(X_train)
pca.fit(X_test)


""" Trying out the Various Models """
regressor_xgb=RandomForestRegressor(n_estimators = 300)
regressor_xgb.fit(X_train,y_train)
y_pred=regressor_xgb.predict(X_test)

prediction = pd.DataFrame(y_pred,columns=['Item_Outlet_Sales'])
results = test_data.iloc[:,[0,6]]
results = pd.concat([results, prediction],axis=1)
results.to_csv('randomforest.csv',index=False)








