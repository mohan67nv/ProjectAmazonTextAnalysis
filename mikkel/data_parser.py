import pandas as pd
import gzip
import pickle
import time


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

def load_pickles():
    training_data = pickle.load(open("training_data.pickle", "rb"))
    training_labels = pickle.load(open("training_labels.pickle", "rb"))
    return training_data, training_labels

if __name__ == '__main__':
    start_time = time.time()
    # generate_pickles('D:/amazon_data/reviews_Electronics_5.json.gz')

    training_data = pickle.load(open("training_data.pickle", "rb"))
    # training_labels = pickle.load(open("training_labels.pickle", "rb"))

    print("Time :", time.time() - start_time)
