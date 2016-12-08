import pandas as pd
import gzip

import time


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if (i+1) % 1000 == 0:
            print("Step:", i)
    return pd.DataFrame.from_dict(df, orient='index')

start_time = time.time()
df = getDF('D:/amazon_data/reviews_Electronics_5.json.gz')

print("Time used:", time.time()-start_time)
