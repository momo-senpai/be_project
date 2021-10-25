if request.method == 'POST':
        try:
            googl = yf.download('AAPL', start='2020-01-01', end='2020-01-31')
            googl.drop(['Adj Close'], axis=1, inplace=True)
            prices, extrema, smooth_prices, smooth_extrema = find_extrema(googl['Close'], bw=[1.5])
            patterns = find_patterns(extrema)

            for name, pattern_periods in patterns.items():
                print(f"{name}: {len(pattern_periods)} occurences")

            print(patterns.items)
            for name, pattern_periods in patterns.items():
                if name=='TBOT':
                    print(name)
                for name, pattern_periods in patterns.items():
                    print(f"{name}: {len(pattern_periods)} occurences")
            return render_template('finder.html', items=pattern.items())
        except Exception as e:
            print("Failed to get required data.", e)



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

<div>
                        {% for name,pattern_periods in items %}
                            <div>   <h2>{% print(f"{name}: {len(pattern_periods)} occurences") %}</h2></div>
                            {% endfor %}
                        </div>