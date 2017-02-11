# coding: utf-8

# In[2]:

import os

path = 'C:\\users\johannes\ProjectAmazonTextAnalysis\johannes'
os.chdir(path)
import pickle

import pandas as pd
import numpy as np

import sklearn

print(sklearn.__version__)
from sklearn.model_selection import train_test_split

from collections import Counter
import gzip

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import ngrams
from nltk.corpus import stopwords

import time


# In[3]:

# import spacy
# nlp = spacy.load('en')


# In[31]:

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


sample_size = 1000


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
        if i <= sample_size:
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
df = get_training_data('reviews_Electronics_5.json.gz')

print("Time :", time.time() - start_time)

df_1 = df


# In[32]:

def fix_dataframe(df=df_1):
    y = df['overall'].values
    X = df['reviewText']
    df = pd.DataFrame(np.column_stack((X, y)), columns=['text', 'labels'])
    return df


df = fix_dataframe(df_1)


# In[33]:

def split_data(df=df):
    train_df, test_df = train_test_split(df)
    # print(train_df.head())
    # return pd.DataFrame(train_df, columns=['text', 'labels']), pd.DataFrame(test_df, columns=['text', 'labels'])
    return train_df, test_df


train_df, test_df = split_data(df)
train_df.head()
print(train_df.shape)


# In[34]:

# Using the standard stopwords given by nltk. Can also do feature relevance according to word frequency limits.
# Could also test a limit for word appearance in a given percentage of texts.

def find_words(df=train_df, stopword=False, word_frequency=[sample_size, np.log(sample_size)]):
    # stemmer = SnowballStemmer('english')

    texts = df['text'].values
    # dictionary = np.unique([word.lower() for text in texts for word in word_tokenize(text)])
    # word_count = Counter([word.lower() for text in texts for word in word_tokenize(text)])

    if stopword == False:
        word_count = Counter([word.lower() for text in texts
                              for word in word_tokenize(text)])
        if word_frequency != None:
            word_count = {word: count for word, count in word_count.items()
                          if count < word_frequency[0] and count > word_frequency[1]}
    elif stopword == True:
        word_count = Counter([word.lower()
                              for text in texts
                              for word in word_tokenize(text)
                              if word not in stopwords.words('english')])
    else:
        raise ValueError('stopword argument needs to be True/False')

    dictionary = [word for word, count in word_count.items()]
    word_count = sorted([(word, count) for word, count in word_count.items()], key=lambda x: -x[1])
    return word_count, dictionary


word_freq, dictionary = find_words()
print(word_freq[:100])


# In[35]:

def find_bigrams(words):
    return zip(words, words[1:])


# In[36]:

def get_bigrams(df=train_df, lower_limit=np.log(sample_size)):
    texts = df['text'].values
    # lower case text
    texts_lower = [[word.lower() for word in word_tokenize(text)] for text in texts]
    # bigrams from the lower case text
    bigrams = [gram for text in texts_lower for gram in find_bigrams(text)]
    # Count of bigrams sorted
    bigram_count = Counter(bigrams)
    bigrams = [bigram for bigram, count in bigram_count.items() if count > lower_limit]
    sorted_bigrams = sorted([(bigram, count)
                             for bigram, count in bigram_count.items()
                             if bigram in bigrams],
                            key=lambda x: -x[1])

    return bigrams, sorted_bigrams, texts_lower
    # (',', 'but) , ('do', "n't"), ('the', 'price'), ('.', 'if'), ('but', 'it'), ('did', "n't"), 


bigrams, bigram_count, texts_lower = get_bigrams()
# print(bigrams)
print(bigram_count[:100])


# In[10]:

def word_dataframe(texts=train_df.text.values,
                   words=dictionary):
    word_occurances = []
    for text in texts:
        text_occurences = np.zeros(len(words))
        for word in word_tokenize(text):
            word = word.lower()
            if word in words:
                index = words.index(word)
                text_occurences[index] += 1
        word_occurances.append(text_occurences)

    X_words = pd.DataFrame(np.array(word_occurances), columns=words)
    return X_words


# In[ ]:

def word_dataframe(texts=train_df.text.values,
                   words=dictionary):
    word_occurances = []
    texts_lower = [[word.lower() for word in word_tokenize(text)] for text in texts]

    for text in texts_lower:
        text_occurences = np.zeros(len(words))
        for word in text:
            if word in words:
                index = words.index(word)
                text_occurences[index] += 1
        word_occurances.append(text_occurences)

    X_words = pd.DataFrame(np.array(word_occurances), columns=words)
    return X_words


# In[30]:

def bigram_dataframe(texts=train_df.text.values,
                     bigrams=bigrams):
    bigram_occurances = []
    for text in texts:
        text_occurences = np.zeros(len(bigrams))
        text_words = [word.lower() for word in word_tokenize(text)]
        bigrams_in_text = [gram for gram in find_bigrams(text_words)]
        for gram in bigrams_in_text:
            if gram in bigrams:
                index = bigrams.index(gram)
                text_occurences[index] += 1
        bigram_occurances.append(text_occurences)

    cols = [str(gram) for gram in bigrams]
    # print(cols)
    X_bigrams = pd.DataFrame(np.array(bigram_occurances), columns=cols)
    return X_bigrams


_ = bigram_dataframe()
_.head()

# In[12]:

# Dataframes of bigrams/unigrams in the train and test datasets.
train_bigrams = bigram_dataframe(texts=train_df.text.values)
test_bigrams = bigram_dataframe(texts=test_df.text.values)

train_unigrams = word_dataframe(texts=train_df.text.values)
test_unigrams = word_dataframe(texts=test_df.text.values)


# In[24]:

# print(train_unigrams.head())
# print(test_unigrams.head())
# print(20*'#')
# print(train_unigrams.shape)
# print(test_unigrams.shape)


# In[25]:

# Combining the bigrams and unigrams dataframes to one dataframe. 
def merge_grams(unigrams, bigrams):
    combined = pd.concat([bigrams, unigrams], axis=1)
    return combined


# In[31]:

X_train = merge_grams(train_bigrams, train_unigrams)
X_test = merge_grams(test_bigrams, test_unigrams)
y_train = train_df['labels']
y_test = test_df['labels']

# In[32]:

X_train.to_pickle('X_train.pkl')
X_test.to_pickle('X_test.pkl')
y_train.to_pickle('y_train.pkl')
y_test.to_pickle('y_test.pkl')


# In[ ]:
