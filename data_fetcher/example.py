import pandas as pd
import gzip

import time
import numpy as np


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



# Fixing names on the dataframe
def fix_dataframe(df = df):
    y = df['overall'].values
    X = df['reviewText']
    df = pd.DataFrame(np.column_stack((X,y)), columns = ['text', 'review_labels'])
    return df
df = fix_dataframe(df)

# Checking class distribution.
def class_dist(df = df):
    value_counts = df.groupby('review_labels').count()
    value_counts['distribution'] = value_counts.text.apply(lambda x: x/value_counts.text.sum())
    return value_counts

# Class balancing
def class_balancing(df = df):
    values = {}
    for i in range(1,6):
        values[str(i)] = df[df.review_labels == i].sample(class_dist(df).text.min())
    balanced_df = pd.concat([values['1'], values['2'], values['3'], values['4'], values['5']], axis = 0)
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df
