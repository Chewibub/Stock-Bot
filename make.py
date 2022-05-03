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
'''
Prints initial data to excel (offline/speed)
'''
# msft = yf.Ticker("MSFT")
# print(msft)

# df = msft.history(period="max")
# df.to_csv('msft.csv')

df3 = pd.read_csv('googl.csv', header = None)
df3 = df3.rename(columns = {0:'Date', 1:'Open', 2:'High', 3:'Low', 4:'Close'})

df = pd.read_csv('PE_googl.csv', header = None)
df = df.rename(columns = {0: 'Sales', 1: 'EPS', 2: 'Div', 3: 'Flow', 4: 'Date'})


df2 = pd.DataFrame(np.repeat(df.iloc[[0]].values,1,axis=0))

x = 0
i = 1
curr = 0

for date in df.Date:
	if x == len(df['Date']) - 1:
		i = len(df3['Date']) - len(df2)
		df_temp = pd.DataFrame(np.repeat(df.iloc[[x]].values, i, axis=0))
		df2 = df2.append(df_temp, ignore_index=True)
		break
	test = datetime.strptime(df['Date'].loc[x + 1], '%b \'%y')
	while test > datetime.strptime(df3['Date'].loc[i], '%d/%m/%Y'):
		i += 1
		curr += 1
	df_temp = pd.DataFrame(np.repeat(df.iloc[[x]].values, curr, axis=0))
	df2 = df2.append(df_temp, ignore_index=True)
	curr = 0
	x += 1

df2 = df2.join(df3)
df2.columns = ['Sales', 'EPS', 'Div', 'Flow', 'Date2', 'Date',  'Price', 'Change', 'ChangeP',  'Open', 'High', 'Low', 'VWAP', 'ShortInt', 'Short', 'Vol30', 'Vol90', 'Vol90A', 'Vol30A', 'CVol']
df2 = df2.rename(index = {0: 'DropIT'})
df2 = df2.drop(columns = ['Date2'])
df2 = df2.drop(index = ['DropIT'])

date_list = []

for date in df2['Date']:
	date_temp = datetime.strptime(date, '%d/%m/%Y')
	date_temp_num = date_temp.year * 10000 + date_temp.month * 100 + date_temp.day
	date_list.append(date_temp_num)

df2['Date'] = date_list


x = 1

VWAP_list = []
roll_close_xvol = 0
roll_vol = 0
df2 = df2.apply(pd.to_numeric, errors = 'coerce')

for VWAP in df2['VWAP']:
	if np.isnan(df2['VWAP'].loc[x]) == False:
		VWAP_list.append(df2['VWAP'].loc[x])
		x += 1
		continue
	if x > 7:
		roll_close_xvol -= df2['CVol'].loc[x - 7] * df2['Price'].loc[x - 7]
		roll_vol -= df2['CVol'].loc[x - 7]

	roll_close_xvol += df2['CVol'].loc[x] * df2['Price'].loc[x]
	roll_vol += df2['CVol'].loc[x]
	VWAP_temp = roll_close_xvol / roll_vol
	VWAP_list.append(VWAP_temp)
	x += 1

df2['VWAP'] = VWAP_list

EPS_Year = []
EPS_roll = 0
df2['EPS'].fillna(0, inplace = True)

x = 1
prev = df2['Date'].loc[1]
prev_x = x

for EPS in df2['EPS']:
	divisor = x - prev_x + 1
	curr = df2['Date'].loc[x]
	if curr - prev > 10000:
		EPS_roll -= df2['EPS'].loc[prev_x]
		prev_x += 1
	EPS_roll += EPS
	EPS_temp = EPS_roll / divisor * 4
	EPS_Year.append(EPS_temp)
	x += 1

df2['EPS'] = EPS_Year

x = 1

PE_list = []
for EPS in df2['EPS']:
	if EPS == 0:
		PE = 0
	else:
		PE = df2['Price'].loc[x] / EPS
	PE_list.append(PE)
	x += 1

df2['PE'] = PE_list

T50MA_list = []
T50MA_roll = 0

x = 1

for price in df2['Price']:
	divisor = x
	if x > 50:
		T50MA_roll -= df2['Price'].iloc[x - 50]
		divisor = 50
	T50MA_roll += price
	T50MA = T50MA_roll / divisor
	T50MA_list.append(T50MA)
	x += 1

df2['T50MA'] = T50MA_list

T100MA_list = []
T100MA_roll = 0

x = 1

for price in df2['Price']:
	divisor = x
	if x > 100:
		T100MA_roll -= df2['Price'].iloc[x - 100]
		divisor = 100
	T100MA_roll += price
	T100MA = T100MA_roll / divisor
	T100MA_list.append(T100MA)
	x += 1	

df2['T100MA'] = T100MA_list

dayssince_list = []
days_since_crossed = 0

for price50, price100 in zip(df2['T50MA'], df2['T100MA']):
	if price50 >= price100:
		if days_since_crossed < 0:
			days_since_crossed = 0
		else:
			days_since_crossed += 1
	elif price100 > price50:
		if days_since_crossed > 0:
			days_since_crossed = 0
		else:
			days_since_crossed -= 1
	dayssince_list.append(days_since_crossed)

df2['T100_50Lcross'] = dayssince_list

relative_close_list = []
temp_close = 0

for price50, price100 in zip(df2['T50MA'], df2['T100MA']):
	 if price50 >= price100:
	 	temp_close = price50 / price100
	 else:
	 	temp_close = price100 / price50
	 relative_close_list.append(temp_close)

df2['T100_50Prox'] = relative_close_list

check_day = {'Mon': 0, 'Tue': 0, 'Wed': 0, 'Thur': 0, 'Fri': 0}

zero_data = np.zeros(shape=(1, len(df2.columns)))
df_empty = pd.DataFrame(zero_data, columns=df2.columns)

prev = datetime.strptime(str(df2['Date'].loc[1]), '%Y%m%d')
first = 1

df_dict = {}
df_dict['Mon'] = df_empty
df_dict['Tue'] = df_empty
df_dict['Wed'] = df_empty
df_dict['Thur'] = df_empty
df_dict['Fri'] = df_empty


total = 1

for date in df2['Date']:
	date_temp = datetime.strptime(str(date), '%Y%m%d')
	if first == 1:
		if check_day['Fri'] == 1:
			first = 2
		if check_day['Thur'] == 1 and date_temp.strftime('%A') == "Monday":
			first = 2

	if (date_temp - prev).days > 5 or first == 2:
		for day in check_day:
			if check_day[day] == 0:
				df_dict[day] = df_dict[day].append(df_empty)
		prev = date_temp
		check_day = check_day.fromkeys(check_day, 0)
		total += 1
		first = 0
	if date_temp.strftime('%A') == "Monday":
		df_dict['Mon'] = df_dict['Mon'].append(df2[(df2['Date'] == date)], ignore_index = True)
		check_day['Mon'] = 1
	elif date_temp.strftime('%A') == "Tuesday":
		df_dict['Tue'] = df_dict['Tue'].append(df2[(df2['Date'] == date)], ignore_index = True) 
		check_day['Tue'] = 1
	elif date_temp.strftime('%A') == "Wednesday":
		df_dict['Wed'] = df_dict['Wed'].append(df2[(df2['Date'] == date)], ignore_index = True)
		check_day['Wed'] = 1
	elif date_temp.strftime('%A') == "Thursday":
		df_dict['Thur'] = df_dict['Thur'].append(df2[(df2['Date'] == date)], ignore_index = True)
		check_day['Thur'] = 1
	elif date_temp.strftime('%A') == "Friday":
		df_dict['Fri'] = df_dict['Fri'].append(df2[(df2['Date'] == date)], ignore_index = True)
		check_day['Fri'] = 1

df5 = pd.concat(df_dict, axis = 1)
df5 = df5.drop(df5.index[0])
x = 0
next_list = []
for x in range(len(df5['Fri']['Price']) - 1):
	x += 1
	if df5['Fri']['Price'].loc[x] == 0:
		price = df5['Thur']['Price'].loc[x]
	else:
		price = df5['Fri']['Price'].loc[x]
	if df5['Fri']['Price'].loc[x + 1] == 0:
		change = df5['Thur']['Price'].loc[x + 1] / price - 1
	else:
		change = df5['Fri']['Price'].loc[x + 1] / price - 1
	print(change)
	next_list.append(change)


#df5.to_csv('test.csv')
'''
What you want to predict
'''
# next_list = []
# for i in range(len(df2['ChangeP']) - 1):
# 	next_list.append(df2['ChangeP'].loc[i + 2])

next_list.append(0)
df5['NextCP'] = next_list

# df4 = pd.read_csv('test_amzn.csv', header = None)
# df4.columns = ['Sales', 'EPS', 'Div', 'Flow', 'Date',  'Price', 'Change', 'ChangeP',  'Open', 'High', 'Low', 'VWAP', 'ShortInt', 'Short', 'Vol30', 'Vol90', 'Vol90A', 'Vol30A', 'CVol', 'PE', 'T50MA', 'T100MA', 'NextCP']
# df4 = df4.drop(df4.index[0])
# df2 = pd.concat([df2, df4], ignore_index = True) 

print(df5)
# df5.to_csv('test.csv')