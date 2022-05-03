import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from datetime import datetime
import sys

df = pd.read_csv('test_googl.csv', header = None)
df = df.drop(df.index[0])
df.columns = ['Sales', 'EPS', 'Div', 'Flow', 'Date',  'Price', 'Change', 'ChangeP',  'Open', 
				'High', 'Low', 'VWAP', 'ShortInt', 'Short', 'Vol30', 'Vol90', 'Vol90A', 'Vol30A', 
				'CVol', 'PE', 'T50MA', 'T100MA', 'T100_50Lcross', 'T100_50Prox', 'NextCP']

df2 = pd.read_csv('test_amzn.csv', header = None)
df2 = df2.drop(df.index[0])
df2.columns = ['Sales', 'EPS', 'Div', 'Flow', 'Date',  'Price', 'Change', 'ChangeP',  'Open', 
				'High', 'Low', 'VWAP', 'ShortInt', 'Short', 'Vol30', 'Vol90', 'Vol90A', 'Vol30A', 
				'CVol', 'PE', 'T50MA', 'T100MA', 'T100_50Lcross', 'T100_50Prox', 'NextCP']

df3 = pd.read_csv('test_msft.csv', header = None)
df3 = df3.drop(df.index[0])
df3.columns = ['Sales', 'EPS', 'Div', 'Flow', 'Date',  'Price', 'Change', 'ChangeP',  'Open', 
				'High', 'Low', 'VWAP', 'ShortInt', 'Short', 'Vol30', 'Vol90', 'Vol90A', 'Vol30A', 
				'CVol', 'PE', 'T50MA', 'T100MA', 'T100_50Lcross', 'T100_50Prox', 'NextCP']


df4 = pd.DataFrame(np.concatenate([df, df2, df3]), columns = df.columns)

df4.sort_values(by = ['Date'], inplace = True)
df4 = df4[df4.Sales != 'Sales']

value_list = []

# for value in df4['NextCP']:
# 	value_list.append(float(value)**3)

# df4['NextCP'] = value_list


#df2.to_csv('test.csv')


x = df4.drop(['NextCP'], axis = 1)
y = df4.NextCP

x = x.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors= 'coerce')

x.fillna(0, inplace = True)
y.fillna(0, inplace = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 888, shuffle = False)

clf = RandomForestRegressor(n_estimators = 100)
clf.fit(x_train, y_train)

pred = clf.predict(x_test)

predicted = []
predicted = clf.predict(x)
df4['Predicted'] = predicted
# value_list = []
# for value in df4['NextCP']:
# 	if value < 0:
# 		temp = -(-value) ** (1. / 3)
# 		value_list.append(temp)
# 	else:
# 		temp = value ** (1. / 3)
# 		value_list.append(temp)
# df4['NextCP'] = value_list

# value_list = []
# for value in df4['Predicted']:
# 	if value < 0:
# 		temp = -(-value) ** (1. / 3)
# 		value_list.append(temp)
# 	else:
# 		temp = value ** (1. / 3)
# 		value_list.append(temp)
# df4['Predicted'] = value_list	

df4.to_csv('test2.csv')

'''
Predicts specfic date
'''
# df_2 = df[(df['Date'] == '12/05/2020')]
# x_2 = df_2.drop(['Next'], axis = 1)
# x_2 = x_2.apply(pd.to_numeric, errors='coerce')
# x_2.fillna(0, inplace = True)
# print(x_2)
# print(clf.predict(x_2))


print(f'\n Decision Tree model coefficients:')
print(f'R^2 score for decision tree model: {r2_score(y_test, pred):.4f}')
print(f'Mean square error (MSE) score for decision tree model: {mean_squared_error(y_test, pred):.4f}')
print(f'Mean absolute error (MAE) score for decision tree model: {mean_absolute_error(y_test, pred):.4f}')

# # # print(clf.feature_importances_)



importances = list(zip(x.columns, clf.feature_importances_))
print('Feature importances/weighting for Decision Tree model:\n')
for i in importances:
    print(f'Feature {i[0]} , Weighting Score: {i[1]:.4f}')

# plt.bar([x - 0.1 for x in range(len(importances))], importances, width = 0.2, label = 'Tree')
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], df.columns[:-1], fontsize = 9)
# plt.xlabel('Feature', fontsize = 15)
# plt.ylabel('Weighting', fontsize = 15)
# plt.show()