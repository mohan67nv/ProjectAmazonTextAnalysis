from collections import Counter

import pandas as pd
import gzip
import pickle
import time
import re
from string import punctuation
import nltk
# nltk.download()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_training_data(path):
    """
    Get all usable data
    :param path: path to compressed data
    :return: panda data frame
    """
    i = 0
    df = {}
    for d in parse(path):
        i += 1
        if i <= 1400000:
            df[i] = d
        else:
            break
        if (i + 1) % 1000 == 0:
            print("Step:", i + 1)
    return pd.DataFrame.from_dict(df, orient='index')


def generate_pickles(path='D:/amazon_data/reviews_Electronics_5.json.gz'):
    df_data = get_training_data(path)
    pickle.dump(df_data['reviewText'].values, open("./training_data.pickle", "wb"))
    pickle.dump(df_data['overall'].values, open("./training_labels.pickle", "wb"))
    return df_data['reviewText'].values, df_data['overall'].values


def get_meaningful_words(text):
    stops = set(stopwords.words("english"))
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    return [w for w in words if not w in stops]


def generate_bag_of_words(reviews, n_top_words=50000, save=False):
    without_stp = Counter()
    for i, review in enumerate(reviews):
        if i % 1000 == 0:
            print("Review nr:", i)
        meaningful_words = get_meaningful_words(review)
        without_stp.update(w for w in meaningful_words)

    print("Generating top bag of words...")
    top_words = [y[0] for y in without_stp.most_common(n_top_words)]
    top_words.sort()

    if save:
        print("Saving...")
        pickle.dump(top_words, open("./bag_of_words.pickle", "wb"))
    return top_words


    # def get_test_data(path):


#     """
#     Do not call this before the real test!!!!
#     """
#     pass
#     i = 0
#     df = {}
#     for d in parse(path):
#         i += 1
#         if i > 1400000:
#             df[i] = d
#     return pd.DataFrame.from_dict(df, orient='index')

def load_training_data():
    training_data = pickle.load(open("training_data.pickle", "rb"))
    training_labels = pickle.load(open("training_labels.pickle", "rb"))
    return training_data, training_labels


def load_bag_of_words(w_freq=False):
    if w_freq:
        return pickle.load(open("bag_of_words_w_freq.pickle", "rb"))
    return pickle.load(open("bag_of_words.pickle", "rb"))


if __name__ == '__main__':
    start_time = time.time()
    # generate_pickles('D:/amazon_data/reviews_Electronics_5.json.gz')

    training_data = pickle.load(open("training_data.pickle", "rb"))
    # training_labels = pickle.load(open("training_labels.pickle", "rb"))

    bag_of_words = generate_bag_of_words(training_data, save=True)

    print("Time :", time.time() - start_time)
