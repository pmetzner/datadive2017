#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import gzip
from gensim import corpora, models, similarities

# Load corpus and lookup dictionary
corpus = pickle.load(gzip.open('./corpus.p.gz', 'rb'))
lookup = pickle.load(open('./lookup.p','rb'))

# Train model
tfidf = models.tfidfmodel.TfidfModel(corpus, id2word=lookup)
corpus_tfidf = tfidf[corpus]

model = models.LdaModel(corpus_tfidf, id2word=lookup, num_topics=10)
pickle.dump(model, gzip.open('model.p.gz', 'wb', compresslevel=9))
    
