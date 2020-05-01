from flask import Flask, request
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100


app = Flask(__name__)
data = pd.read_csv('data/sentiment.tsv', sep='\t', names=['id', 'label_dataset','train_dataset'])

def get_train_set(input_txt):
    input_txt = input_txt.split("\t")
    return input_txt



data['tidy_tweet'] = np.vectorize(remove_pattern)(data['train_dataset'], "<br >")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100


data['train_len'] = data['train_dataset'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['train_dataset'].apply(lambda x: count_punct(x))

X = data['tidy_tweet'][:200]
y = data['label_dataset'][:200]

cv = CountVectorizer()
X = cv.fit_transform(X)
X = pd.concat([data['train_len'][:200], data['punct%'][:200], pd.DataFrame(X.toarray())], axis=1)

clf = LogisticRegression(C=1.0, class_weight=None, dual=False,
                         fit_intercept=True,
                         intercept_scaling=1, l1_ratio=None,
                         max_iter=100, multi_class='auto',
                         n_jobs=None, penalty='l2',
                         random_state=None, solver='lbfgs',
                         tol=0.0001, verbose=0,
                         warm_start=False)

clf.fit(X, y)


@app.route('/')
def home():
    return "hello"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.json
        data = [json_data["message"]]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len, punct, vect], axis=1)
        my_prediction = clf.predict(total_data)

    return str(my_prediction[0]);


if __name__ == '__main__':
    app.run(port=4000)
