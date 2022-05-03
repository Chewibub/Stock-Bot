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

def big_algo():
    return

portfolio = {}
portfolio['Money'] = 1000000

def algo(dt):


    start_dt = dt - datetime.timedelta(31)

    end_dt = dt - datetime.timedelta(1)

    print(start_dt)
    print(end_dt)

    stock_codes = ['MSFT', 'AMZN', 'AAPL', 'GOOG', 'NVDA', 'CRM']
    market_cap_history = {'MSFT' : 1660000000000, 'AMZN' : 1639000000000, 'AAPL' : 2036000000000, 'GOOG' : 1070000000000, 'NVDA' : 341260000000, 'CRM' : 235280000000}

    total_capped_mean = 0
    total_market_cap = 0
    weighted_variance = 0
    weighted_std_deviation = 0

    for ticker in stock_codes:
        # print(ticker)
        stock = yf.Ticker(ticker)
        total = 0
        count = 0
        prev = -1
        for stock_price in stock.history(start=start_dt, end=end_dt).Close:
            # Previous
            print("count is : "+ str(count), end = ' ')
            print("stk price : " + str(stock_price))
            if prev == -1:
                prev = stock_price
                continue
            # Running total pct change += Current / Previous - 1
            total += abs((stock_price / prev) - 1)
            # Running count
            count += 1
            prev = stock_price
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
        for stock_price in stock.history(start=start_dt, end=end_dt).Close:
            if prev == -1:
                prev = stock_price
                continue
            current_pct_change = abs((stock_price / prev) - 1)
            total_variance += (current_pct_change - weighted_mean) ** 2
            count += 1
            prev = stock_price
        mean = total_variance / count
        market_cap = market_cap_history[ticker]
        weighted_variance += mean * market_cap

    weighted_std_deviation = math.sqrt(weighted_variance / total_market_cap)
    print("weighted_std_deviation: " + str(weighted_std_deviation))
    # print("weighted_mean: " + str(weighted_mean))


    trading_leeway = datetime.timedelta(days=7
        )
    max_daily_change = 0
    moveable_units = {}
    best_ticker = None

    for ticker in stock_codes:
        stock = yf.Ticker(ticker)
        history = stock.history(start=dt-trading_leeway, end=dt).Close
        today_price = history[len(history) - 1]
        prev_price = history[len(history) - 2]
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

        if deviations_changed == math.nan:
            continue

        if not ticker in portfolio:
            portfolio[ticker] = 0

        if deviations_changed > 0:
            if portfolio[ticker] < math.floor(((abs(deviations_changed) ** 2) * 5000) / today_price):
                moveable_units[ticker] = -math.floor(portfolio[ticker])
            else:
                moveable_units[ticker] = -math.floor((((abs(deviations_changed) ** 2) * 5000) / today_price))
        else:
            if portfolio['Money'] < math.floor(((abs(deviations_changed) ** 2) * 5000)):
                moveable_units[ticker] = math.floor(portfolio['Money'] / today_price)
            else:
                moveable_units[ticker] = math.floor(((abs(deviations_changed) ** 2) * 5000) / today_price)

        if moveable_units[ticker] != 0:
            if deviations_changed > -2:
                if abs(moveable_units[ticker] * today_price) > max_daily_change:
                    max_daily_change = moveable_units[ticker] * today_price
                    best_ticker = ticker

    print(" ------ ")
    print("Date : " + str(dt))
    if best_ticker == None:
        print("No buying on this day")
        return
    print("maximum moveable units: " + str(moveable_units[best_ticker]))
    print("best ticker to buy/sell: " + str(best_ticker))

    if moveable_units[best_ticker] != 0:
        history = yf.Ticker(best_ticker).history(start=dt-trading_leeway, end=dt).Close
        portfolio['Money'] -= history[len(history) - 1] * moveable_units[best_ticker]
        portfolio[best_ticker] += moveable_units[best_ticker]
        print("spending: " + str(moveable_units[best_ticker] * history[len(history) - 1]))
        print("stock_price :" +str(history[len(history) - 1]))

    return

a = datetime.datetime(2017, 3, 8)
numdays = 2
dateList = []
for x in range (0, numdays):
    dateList.append(a + datetime.timedelta(days = x))

start_time = time.time()

for date in dateList:
    algo(date)

portfolio['Wealth'] = 0

print("-----  Results  -----")

for stock in portfolio:
    if stock != 'Money':
        if stock != 'Wealth':
            history = yf.Ticker(stock).history(start=a-datetime.timedelta(5), end=a).Close
            current_price = history[len(history) - 1]
            portfolio['Wealth'] += current_price * portfolio[stock]
            print("Holding " + str(stock))
            print("Units : " + str(portfolio[stock]))
            print("Value : " + str(current_price * portfolio[stock]))
    elif stock == 'Money':
        portfolio['Wealth'] += portfolio[stock]
        print("Holding " + str(portfolio[stock]), end = ' ')
        print("in cash") 

elapsed_time = time.time() - start_time

print("Total wealth at trading end is: " + str(portfolio['Wealth']))

print("elapsed_time : " + str(elapsed_time))