#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import pickle
import time
import gzip
from tqdm import *

start = time.time()

# Set file locations
home = os.environ['DATADIVE']
documents = '%s/termcounts-min-2/' % home
export = '%s/termcounts-min-2_export' % home
os.system('mkdir -p export')

# Read document word counts
corpus_texts = os.listdir(documents)

# Identify terms that appear in at least two documents
corpus = []
freqs = {}
useful = {}

for text in tqdm(corpus_texts, desc='Reading Files'):
    with open(documents + text, 'r') as f:
        thisdoc = []
        for row in f.readlines():
            row = row.split(',')
            if len(row)==2:  # No rows with more than two items
                idx = row[0].strip().replace('"', '')  # Strip quotes from term
                if re.match('\D', idx) and len(idx) > 2:  # No digits
                    val = int(row[1].strip())  # Get count within document
                    thisdoc.append((idx, val))
                    if idx in freqs:
                        freqs[idx] += 1  # Increase frequency
                        if freqs[idx] == 5:
                            useful[idx] = None  # Once 
                    else:
                        freqs[idx] = 1  # Start frequency count for new terms
        corpus.append(thisdoc)  # Add document to corpus

# Generate lookup table from useful terms
lookup = dict(zip(useful.keys(), range(1, len(useful)+1)))

# Prune corpus to useful terms (i.e., appears in at least five documents)
pruned = []
for doc in tqdm(corpus, desc='Thinning corpus'):
    td = []
    for i in doc:
        if i[0] not in useful:
            del i
        else:
            i = list(i)
            i[0] = lookup[i[0]]
            td.append(tuple(i))
    pruned.append(td)

print('Dumping corpus to disk.')
pickle.dump(pruned, gzip.open('corpus.p.gz', 'wb', compresslevel=9))

print('Saving dictionary.')
lookup = {v: k for (k,v) in lookup.items()}
pickle.dump(lookup, open('lookup.p', 'wb'))

stop = time.time()

print('Documents processed in %s seconds.' % str(int(stop)-int(start)))

counts = {}
for doc in pruned:
    for word in doc:
        if word[0] in counts:
            counts[word[0]] += word[1]
        else:
            counts[word[0]] = word[1]