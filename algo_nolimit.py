"""
0. Take in a date (this is to determine what was the previous month)
1. Get price history of stocks for the previous month
2. Find out std deviation for that month
3. Starting on the next week, we check percent change at close from previous month close
4. We buy if - from deviation, + from deviation we sell if we have any, squarely units of std deviation
    4a) e.g, deviation is 2, we see amzn go up 2%, and we're holding 5 units, we sell 1 unit
    4b) deviation is still 2, we see msft go down 4%, we buy 4 units
4.5.) Choose the highest deviation each day
5. if over 2 std deviations, we don't touch (natural variance, instability, etc.)
6. We send orders each week by 'CODE', 'No. Units', 'buy/sell' order is assumed processed at date close (sent above)
"""

"""
period: data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval: data interval (intraday data cannot extend last 60 days) Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
start: If not using period - Download start date string (YYYY-MM-DD) or datetime.
end: If not using period - Download end date string (YYYY-MM-DD) or datetime.
prepost: Include Pre and Post market data in results? (Default is False)
auto_adjust: Adjust all OHLC automatically? (Default is True)
actions: Download stock dividends and stock splits events? (Default is True)
"""

import yfinance as yf
import calendar
import datetime
import math
import time
import pandas as pd
import matplotlib.pyplot as plt

portfolio = {}
portfolio['Money'] = 1000000
market_cap_history = {'MSFT' : 1660000000000, 'AMZN' : 1639000000000, 'AAPL' : 2036000000000, 'GOOG' : 1070000000000, 'NVDA' : 341260000000, 'CRM' : 235280000000, 'TSLA' : 395380000000}
stock_codes = ['MSFT', 'AMZN', 'AAPL', 'GOOG', 'NVDA', 'CRM']

# market_cap_history = {'SPY' : 1}
# stock_codes = ['SPY']

history_dict = {}
wealth_history = {}

def wealth_calculator(dt):
    wealth = 0
    for stock in portfolio:
        if stock != 'Money':
            if stock != 'Wealth':
                current_price = history_dict[stock][dt]
                wealth += current_price * portfolio[stock]
        elif stock == 'Money':
            wealth += portfolio[stock]
    return wealth

def grab_history(trade_start, trade_end):
    for ticker in stock_codes:
        history_dict[ticker] = yf.Ticker(ticker).history(start=trade_start, end=trade_end).Close

def algo(dt):



    if dt not in history_dict[stock_codes[0]]:
        print(" ------ ")
        print('Cannot trade today.')
        return

    wealth_history[dt] = wealth_calculator(dt)
    print("Current wealth : " + str(wealth_history[dt]))

    start_dt = dt - datetime.timedelta(31)

    stock_dict = {}

    total_capped_mean = 0
    total_market_cap = 0
    weighted_variance = 0
    weighted_std_deviation = 0

    dateList = []

    for x in range(0, 31):
        dateList.append(start_dt + datetime.timedelta(days=x))

    for ticker in stock_codes:
        # print(ticker)
        stock = yf.Ticker(ticker)
        total = 0
        count = 0
        prev = -1

        for day in dateList:
            if day in history_dict[ticker]:
                stock_price = history_dict[ticker][day]
                # Previous
                if prev == -1:
                    prev = stock_price
                    continue
                # Running total pct change += Current / Previous - 1
                total += abs((stock_price / prev) - 1)
                # Running count
                count += 1
                prev = stock_price
            else:
                continue
        # mean = runnin total / running count
        mean = total / count
        market_cap = market_cap_history[ticker]
        capped_mean = mean * market_cap
        total_capped_mean += capped_mean
        total_market_cap += market_cap
        # print("Mean:" + str(mean))
        # print("Market Cap: " + str(stock.info['marketCap']))
        # print('')

    weighted_mean = total_capped_mean / total_market_cap

    for ticker in stock_codes:
        stock = yf.Ticker(ticker)
        count = 0
        prev = -1
        total_variance = 0
        for day in dateList:
            if day in history_dict[ticker]:
                stock_price = history_dict[ticker][day]
                if prev == -1:
                    prev = stock_price
                    continue
                current_pct_change = abs((stock_price / prev) - 1)
                total_variance += (current_pct_change - weighted_mean) ** 2
                count += 1
                prev = stock_price
            else:
                continue
        mean = total_variance / count
        market_cap = market_cap_history[ticker]
        weighted_variance += mean * market_cap

    weighted_std_deviation = math.sqrt(weighted_variance / total_market_cap)

    trading_leeway = datetime.timedelta(days=5)
    max_daily_change = 0
    moveable_units = {}
    best_ticker = None

    for ticker in stock_codes:

        for i in range(1, 5):
            if dt - datetime.timedelta(days=i) in history_dict[ticker]:
                dt_prev = dt - datetime.timedelta(days=i)
                break

        today_price = history_dict[ticker][dt]
        prev_price = history_dict[ticker][dt_prev]

        daily_change = (today_price / prev_price) - 1
        # print("Today: " + str(history[len(history) - 1]))
        # print("Last business day: " + str(history[len(history) - 2]))
        # print("daily change: " + str(daily_change))
        # print('')

        deviations_changed = (daily_change - weighted_mean) / weighted_std_deviation
        # print(" ------ ")
        # print("Stock is: " + str(ticker))
        # print("deviations change: " + str(deviations_changed))
        # print("Stock moved: " + str(daily_change))
        # print("Mean is: " + str(weighted_mean))
        # print("Std is: " + str(weighted_std_deviation))

        if not ticker in portfolio:
            portfolio[ticker] = 0

        one_unit = 1000
        multiplier = 3

        if deviations_changed > 0:
            if portfolio[ticker] < math.floor(((abs(deviations_changed) ** multiplier) * one_unit) / today_price):
                moveable_units[ticker] = -(portfolio[ticker])
            else:
                moveable_units[ticker] = -math.floor((((abs(deviations_changed) ** multiplier) * one_unit) / today_price))
        else:
            if portfolio['Money'] < math.floor(((abs(deviations_changed) ** multiplier) * one_unit)):
                moveable_units[ticker] = math.floor(portfolio['Money'] / today_price)
            else:
                moveable_units[ticker] = math.floor(((abs(deviations_changed) ** multiplier) * one_unit) / today_price)

        if moveable_units[ticker] != 0:
            portfolio['Money'] -= today_price * moveable_units[ticker]
            portfolio[ticker] += moveable_units[ticker]
            print("maximum moveable units: " + str(moveable_units[ticker]))
            print("ticker to buy/sell: " + str(ticker))
            print("spending: " + str(moveable_units[ticker] * history_dict[ticker][dt]))
            print("price today: " + str(today_price))




    print(" ------ ")
    if best_ticker == None:
        print("No buying on this day")
        return

    if math.isnan(history_dict[best_ticker][dt]):
        print("*******")
        print("*******")
        print("*******")
        print("*******")
        print("*******")
        print("*******")
        print('wtf, nan?')
        print(dt)
        print("*******")
        print("*******")
        print("*******")
        print("*******")
        print("*******")
        exit()


    # print("weighted_std_deviation: " + str(weighted_std_deviation))
    # print("weighted_mean: " + str(weighted_mean))

    return

a = datetime.datetime(2017, 1, 1)
numdays = 1368
dateList = []
for x in range (0, numdays):
    dateList.append(a + datetime.timedelta(days = x))

b = dateList[len(dateList) - 1]

start_time = time.time()

grab_history(a - datetime.timedelta(days=31), a + datetime.timedelta(days=numdays))

for date in dateList:
    algo(date)

portfolio['Wealth'] = 0

print("-----  Results  -----")

for stock in portfolio:
    if stock != 'Money':
        if stock != 'Wealth':
            current_price = history_dict[stock][-1]
            portfolio['Wealth'] += current_price * portfolio[stock]
            print("Holding " + str(stock))
            print("Units : " + str(portfolio[stock]))
            print("Value : " + str(current_price * portfolio[stock]))
    elif stock == 'Money':
        portfolio['Wealth'] += portfolio[stock]
        print("Holding " + str(portfolio[stock]), end = ' ')
        print("in cash") 

elapsed_time = time.time() - start_time

df = pd.DataFrame(data = wealth_history, index = ['Wealth'])

df = (df.T)

df.to_csv('something1.csv')

print("Total wealth at trading end is: " + str(portfolio['Wealth']))

print("elapsed_time : " + str(elapsed_time))