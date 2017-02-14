from collections import Counter
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import re
from string import punctuation
import nltk
# nltk.download()
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix

STOP_WORDS = set(stopwords.words("english"))


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


# Fixing names on the dataframe
def fix_dataframe(df):
    y = df['overall'].values
    X = df['reviewText']
    df = pd.DataFrame(np.column_stack((X, y)), columns=['text', 'review_labels'])
    return df


# Checking class distribution.
def class_dist(df):
    value_counts = df.groupby('review_labels').count()
    value_counts['distribution'] = value_counts.text.apply(lambda x: x / value_counts.text.sum())
    return value_counts


# Class balancing
def class_balancing(df):
    values = {}
    for i in range(1, 6):
        values[str(i)] = df[df.review_labels == i].sample(class_dist(df).text.min())
    balanced_df = pd.concat([values['1'], values['2'], values['3'], values['4'], values['5']], axis=0)
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df


def generate_pickles(path='D:/amazon_data/reviews_Electronics_5.json.gz'):
    df_data = get_training_data(path)
    df_data = fix_dataframe(df_data)
    df_data = class_balancing(df_data)
    pickle.dump(df_data['text'].values, open("./training_data_balanced.pickle", "wb"))
    pickle.dump(df_data['review_labels'].values, open("./training_labels_balanced.pickle", "wb"))
    return df_data['text'].values, df_data['review_labels'].values


def get_meaningful_words(text, no_stopwords=True):
    words = re.sub("[^a-zA-Z]", " ", text).lower().split()
    if no_stopwords:
        return [w for w in words if not w in STOP_WORDS]
    else:
        return words


def generate_bag_of_words(reviews, n_top_words=50000, remove_n_frequent_words=0, no_stopwords=True, save=False):
    without_stp = Counter()
    for i, review in enumerate(reviews):
        if i % 1000 == 0:
            print("Review nr:", i)
        meaningful_words = get_meaningful_words(review, no_stopwords=no_stopwords)
        without_stp.update(w for w in meaningful_words)

    print("Total number of unique words:", len(without_stp))
    print("Generating top bag of words...")
    top_words = [y[0] for y in without_stp.most_common(n_top_words + remove_n_frequent_words)]
    top_words.sort()
    top_words = top_words[:n_top_words]

    if save:
        print("Saving...")
        pickle.dump(top_words, open("./bag_of_words_balanced.pickle", "wb"))
    return top_words


    # def get_test_data(path):


#     """
#     Do not call this before the real test!!!!
#     """
#     pass
#     df = {}
#     i = 0
#     for d in parse(path):
#         i += 1
#         if i > 1400000:
#             df[i] = d
#     return pd.DataFrame.from_dict(df, orient='index')

def load_training_data():
    training_data = pickle.load(open("training_data.pickle", "rb"))
    training_labels = pickle.load(open("training_labels.pickle", "rb"))
    return training_data, training_labels


def load_bag_of_words(no_stopwords=True, w_freq=False):
    if no_stopwords:
        if w_freq:
            return pickle.load(open("bag_of_words_w_freq.pickle", "rb"))
        return pickle.load(open("bag_of_words.pickle", "rb"))
    else:
        return pickle.load(open("bag_of_words_w_stopwords.pickle", "rb"))


def draw_heatmap(res_true, res_pred):
    classes = ['1', '2', '3', '4', '5']
    cm = np.array(confusion_matrix(res_true, res_pred))
    df_cm = pd.DataFrame(cm, index=['True' + ' ' + i for i in classes],
                         columns=['Predicted' + ' ' + i for i in classes])
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.show()


def generate_index_representation(X_data, Y_data, save_to_pickle=False, max_data=None):
    X_representations = []
    Y_one_hots = []

    one_hot_template = np.array(pickle.load(open("bag_of_words.pickle", "rb")))

    for i, review in enumerate(X_data):
        if (i + 1) % 1000 == 0:
            print(i)

        words = get_meaningful_words(review)
        words_idxs = []
        for word in words:
            if word in one_hot_template:
                idx = np.where(one_hot_template == word)
                words_idxs.append(int(idx[0]))
        X_representations.append(words_idxs)

        if Y_data is not None:
            label = Y_data[i] - 1.  # 0-4, not 1-5 stars
            one_hot_label = np.zeros(5)
            one_hot_label[int(label)] = 1.
            Y_one_hots.append(one_hot_label)

        if max_data is not None and len(X_representations) >= max_data:
            break
    X_representations = np.array(X_representations)
    Y_one_hots = np.array(Y_one_hots)
    if save_to_pickle:
        pickle.dump(X_representations, open("./train_data_ann_representation_balanced.pickle", "wb"))
        pickle.dump(Y_one_hots, open("./train_labels_one_hots_balanced.pickle", "wb"))
    return X_representations, Y_one_hots


if __name__ == '__main__':
    start_time = time.time()
    # generate_pickles('D:/amazon_data/reviews_Electronics_5.json.gz')

    training_data = pickle.load(open("training_data_balanced.pickle", "rb"))
    training_labels = pickle.load(open("training_labels_balanced.pickle", "rb"))

    generate_index_representation(training_data, training_labels, save_to_pickle=True)

    # bag_of_words = generate_bag_of_words(training_data, n_top_words=50000, save=True)

    print("Time :", time.time() - start_time)
