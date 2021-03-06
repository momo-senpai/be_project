import os
import csv
import talib
import pandas
import array as arr
import yfinance as yf
import json
from flask import Flask, request, render_template
from patterns import candlestick_patterns, price_action_patterns
from datetime import date
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.signal import argrelextrema
# from statsmodels.nonparametric.kernel_regression import KernelReg
# from fbprophet import Prophet
from pattern_finder import find_extrema, find_patterns, plot_window

app = Flask(__name__)


@app.route('/getjson')
def getjson():
    ticker = request.args.get('symbol', None)
    if ticker is None:
        return {
            "error": "No symbol found."
        }
    else:
        ticker = ticker.upper()
        path = os.path.join(os.getcwd(), "data", "stocks", ticker+".csv")
        return csv_to_json(path)


@app.route('/chatbot')
def chatbot():
    from newspaper import Article
    import random
    import string
    import nltk
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    nltk.download('punkt', quiet=True)

# Get the article
# article = Article('https://www.investopedia.com/articles/investing/082614/how-stock-market-works.asp')
    article = Article(
        'https://www.investopedia.com/terms/t/technicalanalysis.asp')
    article.download()
    article.parse()
    article.nlp()
    corpus = article.text

# Print the article data

    text = corpus
    sentence_list = nltk.sent_tokenize(text)  # A list a sentence

    print(sentence_list)

    def greeting_response(text):
        text = text.lower()
   # Bots greeting response
        bot_greetings = ['hello', 'hi', 'hey', 'hi there']
   # User greetings
        user_greetings = ['hi', 'heya', 'hello', 'hola']

        for word in text.split():
            if word in user_greetings:
                return random.choice(bot_greetings)

    def index_sort(list_var):
        length = len(list_var)
        list_index = list(range(0, length))

        x = list_var
        for i in range(length):
            for j in range(length):
                if x[list_index[i]] > x[list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp

                    return list_index

    def bot_response(user_input):
        user_input = user_input.lower()
        sentence_list.append(user_input)
        bot_response = ''
        cm = CountVectorizer().fit_transform(sentence_list)
        similarity_scores = cosine_similarity(cm[-1], cm)
        similarity_scores_list = similarity_scores.flatten()
        index = index_sort(similarity_scores_list)
        index = index[1:]
        response_flag = 0

        j = 0
        for i in range(len(index)):
            if similarity_scores_list[index[i]] > 0.0:
                bot_response = bot_response+' '+sentence_list[index[i]]
                response_flag = 1
                j = j+1
                if j > 2:
                    break
                if response_flag ==0:
                    bot_response = bot_response+' '+"I apologize, I dont understand."

                    sentence_list.remove(user_input)
                    return bot_response

                    print("Bot: Hi! I am your ChatBot.")

                    exit_list = ['exit', 'see you later', 'bye', 'quit']
                while(True):
                    user_input = input()
                    if user_input.lower() in exit_list:
                        print("Bot: Bye Bye!")
                        break
                    else:
                        if greeting_response(user_input) != None:
                            print("Bot: "+greeting_response(user_input))
                        else:
                            print('Bot: '+bot_response(user_input))


@app.route('/scanner')
def scanner():
    pattern = request.args.get('pattern', False)
    fet = request.args.get('fet')
    stock_list = request.args.get('state', state())
    stocks = {}

    with open('data/symbols.csv') as f:
        for row in csv.reader(f):
            stocks[row[0]] = {'company': row[1]}

    if pattern:
        for filename in os.listdir('data/stocks'):
            df = pandas.read_csv('data/stocks/{}'.format(filename))
            pattern_function = getattr(talib, pattern)
            symbol = filename.split('.')[0]

            try:
                results = pattern_function(
                    df['Open'], df['High'], df['Low'], df['Close'])
                results.iloc[::-1]

                last = results.tail(50).values[0]
                i = 0
                for j in reversed(results):
                    if last == 0:
                        last = int(j)
                    if last != 0:
                        break

                    if i == 50:
                        break

                    i += 1

                if last > 0:
                    stocks[symbol][pattern] = 'BULLISH'
                    stocks[symbol]["day"] = i
                    print(i)
                elif last < 0:
                    stocks[symbol][pattern] = 'BEARISH'
                    stocks[symbol]["day"] = i
                    print(i)
                else:
                    stocks[symbol][pattern] = None
            except Exception as e:
                print('Failed on File: ', filename, e)
    if fet:
        fetch_data()

    return render_template('scanner.html',
                           candlestick_patterns=candlestick_patterns,
                           stocks=stocks, pattern=pattern, fet=fet,
                           state=stock_list, active='scanner')


@app.route('/news', methods=['GET', "POST"])
def news():
    from newsapi import NewsApiClient
    newsapi = NewsApiClient(api_key='ca28357f195f40a9b89c153b4f569361')
    if request.method == 'POST':
        term = request.form.get('name')
        all_articles = newsapi.get_everything(q=term,
                                              sources='google-news-in,the-hindu,the-times-of-india',
                                              domains='www.thehindu.com,timesofindia.indiatimes.com,news.google.com',
                                              from_param='2021-09-28',
                                              to='2021-09-30',
                                              language='en',
                                              sort_by='relevancy',
                                              page=2)

        return render_template('news.html', articles=all_articles['articles'])


    return render_template('news.html', articles=None)

@app.route('/prediction', methods=['GET', "POST"])
def prediction():
    if request.method == 'POST':
        try:
            ticker = yf.Ticker('AAPL')
            data = ticker.history(period="1d")
            df = pandas.DataFrame(data).reset_index()
            df = df[["Date", "Close"]]
            df = df.rename(columns = {"Date": "ds","Close":"y"})
            fbp = Prophet(daily_seasonality= True)
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=3650)
            forecast = fbp.predict(fut)
            print(forecast)

        except Exception as e:
            print("Failed to get required data.", e)
    return render_template('prediction.html')




@app.route('/finder', methods=['GET', "POST"])
def finder():
    if request.method == 'POST':
        symbol = request.form.get('name')
        try:
            googl = yf.download(symbol + '.NS', start='2020-01-01', end='2020-01-31')
            googl.drop(['Adj Close'], axis=1, inplace=True)
            prices, extrema, smooth_prices, smooth_extrema = find_extrema(googl['Close'], bw=[1.5])
            patterns = find_patterns(extrema)

            return render_template('finder.html',symbol=symbol, items=patterns.items(), pap=price_action_patterns)

        except Exception as e:
            print("Failed to get required data.", e)

    return render_template('finder.html')

    

@app.route('/about')
def about():
    return render_template('about.html', active="about")


@app.route('/patterns')
def patterns():
    return render_template('patterns.html', active="pattern")


@app.route('/fetch_data')
def fetch_data():
    current_date = str(date.today())
    with open('data/symbols.csv') as f:
        for line in f:
            if "," not in line:
                continue
            symbol = line.split(",")[0]
            # print(symbol)
            data = yf.download(symbol + '.NS', start="2020-01-01", end="{}".format(current_date))
            data.to_csv('data/stocks/{}.csv'.format(symbol))

    return {
        "code": "success"
    }


@app.route('/')
def index():
    return render_template('index.html', active='home')


@app.route('/state')
def state():
    consolidate_stock = arr.array('b', [])  # a1
    consolidate_stock = "Consolidating:\n"
    breakout_stock = arr.array('b', [])  # a2
    breakout_stock = "Breaking Out:\n"

    def consolidating(df, percentage=2):
        recent_candlesticks = df[-5:]

        max_close = recent_candlesticks['Close'].max()
        min_close = recent_candlesticks['Close'].min()

        threshold = 1 - (percentage / 100)
        if min_close > (max_close * threshold):
            return True

        return False

    def breaking_out(df, percentage=2.5):
        last_close = df[-1:]['Close'].values[0]

        if consolidating(df[:-1], percentage=percentage):
            recent_closes = df[-6:-1]

            if last_close > recent_closes['Close'].max():
                return True

        return False

    for filename in os.listdir('data/stocks/'):
        df = pandas.read_csv('data/stocks/{}'.format(filename))

        if consolidating(df, percentage=2.5):
            a1 = "{}".format(filename.strip('.csv')) + "\n"
            consolidate_stock += a1

        if breaking_out(df):
            a2 = "{}".format(filename.strip('.csv')) + "\n"
            breakout_stock += a2

        stock_list = consolidate_stock + "\n" + breakout_stock

    return stock_list


def csv_to_json(csvFilePath):
    jsonArray = []

    # read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf)

        # convert each csv row into python dict
        for row in csvReader:
            # add this python dict to json array
            jsonArray.append(row)

    # convert python jsonArray to JSON String and write to file
    jsonString = json.dumps(jsonArray, indent=4)
    return jsonString


if __name__ == "__main__":
    app.run(debug=True)
