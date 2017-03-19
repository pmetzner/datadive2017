#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import gzip
import joblib
import time
import pandas as pd
import random
import re

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from gensim import corpora, models, similarities

start = time.time()

home = os.environ['DATADIVE']
documents = '%s/texts/termcounts-min-2/' % home
export = '%s/texts/termcounts-min-2_export' % home 

# Load corpus and lookup dictionary
print('Loading corpus and dictionary.')
corpus = pickle.load(gzip.open('./corpus.p.gz'))
lookup = pickle.load(open('./lookup.p','rb'))

# Train model
print('To TF-IDF')
tfidf = models.tfidfmodel.TfidfModel(corpus, id2word=lookup)
corpus_tfidf = tfidf[corpus]

print('Fit LDA.')
n_topics = 25
model = models.LdaModel(corpus_tfidf, id2word=lookup, num_topics=n_topics)
pickle.dump(model, gzip.open('lda.p.gz', 'wb', compresslevel=9))

# Get top terms for each topic
topic_terms = []
for i in range(n_topics):
    temp = model.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term)
    topic_terms.append(terms)
    print("Top 10 terms for topic #%s: %s" % (str(i), ", ".join([t[0] for t in terms])))

# Build word cloud for one topic
def terms_to_wordcounts(terms, multiplier=1000):
    expanded = [int(multiplier*t[1]) * [t[0]] for t in terms]
    expanded = [" ".join(t) for t in expanded]
    expanded = " ".join(expanded)
    return(expanded)

wordcloud = (WordCloud(background_color="black")
             .generate(terms_to_wordcounts(topic_terms[0])))

plt.imshow(wordcloud)
plt.savefig('./figures/wordcloud_01.png')


print('Calculate similarities.')
index = similarities.MatrixSimilarity(model[corpus_tfidf])
vec_lda = model[corpus_tfidf]
sims = index[vec_lda]

# Get documents
docs = os.listdir(documents)
docs = [i.replace('termCounts-', '') for i in docs]
docs = [re.sub('txt$', 'pdf', i) for i in docs]

features = pd.read_csv('%s/features/features.csv' % home)
features = features.dropna()

thisdoc = random.choice(features.filename.tolist())
if thisdoc in docs:
    idx = docs.index(thisdoc)
    loadings = model.get_document_topics(corpus_tfidf[idx])
    model.show_topic(loadings[0][0], 10)

stop = time.time()
print('Script executed in %s seconds' % str(int(stop-start)))