#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import pickle
import time
import joblib
from tqdm import *

start = time.time()

# Set file locations
home = os.environ['DATADIVE']
documents = '%s/texts/termcounts-min-2/' % home
export = '%s/texts/termcounts-min-2_export' % home
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
        t_search = re.compile (r'^[a-zA-Z]+',re.IGNORECASE)
        d_search = re.compile (r'\.')
        for row in f.readlines():
            row = row.split(',')
            if len(row)==2:  # No rows with more than two items
                idx = row[0].strip().replace('"', '').replace('-','').replace('_','')  # Strip quotes and replace _,- with blanks in term
                if t_search.match(idx) and not d_search.findall(idx) and len(idx) > 2:  # No digits
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
joblib.dump(pruned, 'corpus.p.gz')

print('Saving dictionary.')
lookup = {v: k for (k,v) in lookup.items()}
pickle.dump(lookup, open('lookup.p', 'wb'))

stop = time.time()

print('Script executed in %s seconds.' % str(int(stop-start)))

