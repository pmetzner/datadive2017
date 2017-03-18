#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

sims = pickle.load(open('similarities.p', 'rb'))
# model = pickle.load(open('model.p', 'rb'))

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
X = model.fit_transform(sims)

plt.plot(sims)