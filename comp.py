import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import datetime
import algo


def back_tester():

	# Testing

	# msft = yf.Ticker("MSFT")
	# hist = msft.history()
	# print(hist.Close['2020-09-17'])

	a = datetime.datetime(2017, 1, 1)
	numdays = 1368
	dateList = []
	for x in range (0, numdays):
	    dateList.append(a + datetime.timedelta(days = x))

	portfolio = {}
	stocks = {}
	portfolio['Money'] = 1000000
	portfolio['Worth'] = 1000000

	wealth_history = {}
	wealth = 0

	for temp_date in dateList:
		day_order = algos(temp_date)
		if day_order == None:		
			wealth_history[temp_date] = wealth
			continue

		ticker = yf.Ticker(day_order['Code'].upper())
		price_data = ticker.history(temp_date)
		cost = price_data.Close[temp_date] * day_order['NoUnits']

		if cost > portfolio['Money']:
			print('Im too poor')
			wealth_history[temp_date] = wealth
			continue

		if stocks[day_order['Code'].upper()] == None:
			stocks[day_order['Code'].upper()] += day_order['NoUnits']
		else:
			stocks[day_order['Code'].upper()] += day_order['NoUnits']

		portfolio['Money'] -= cost

		wealth = 0

		for stock in stocks:
			ticker = yf.Ticker(stock)
			price = ticker.history(temp_date).Close[temp_date]
			wealth += price * stocks[stock]
		wealth_history[temp_date] = wealth
		
	plt.plot(x = dateList, y = wealth_history.values())
	plt.show()

back_tester()