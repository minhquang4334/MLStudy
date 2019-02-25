import pandas as pd
import re
from nltk import ngrams

df = pd.read_csv("./truyen_kieu_data.txt",sep="/", names=["row"]).dropna()

def transform_row(row):
    row = re.sub(r"^[0-9\.]+", "", row)

    row = re.sub(r"[\.,\?]+$", "", row)

    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("\"", " ") \
        .replace(":", " ").replace("\"", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ")

    row = row.strip()
    return row

def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]


df["row"] = df.row.apply(transform_row)

df["1gram"] = df.row.apply(lambda t: kieu_ngram(t, 1))
df["2gram"] = df.row.apply(lambda t: kieu_ngram(t, 2))
df["context"] = df["1gram"] + df["2gram"]
train_data = df.context.tolist()
print len(train_data)
print df.head(10)


