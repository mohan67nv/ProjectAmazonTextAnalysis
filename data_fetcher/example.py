import pandas as pd
import gzip

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


start_time = time.time()
df = get_training_data('D:/amazon_data/reviews_Electronics_5.json.gz')

print("Time :", time.time() - start_time)
