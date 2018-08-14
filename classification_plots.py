#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import csv
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from processing_functions import compute_tfidf
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def create_arrays(real_pred):
	gir = real_pred
	gir_cols_drop = []
	mont = real_pred
	mont_cols_drop = []
	for i, index in enumerate(real_pred.index.values):
		if (real_pred['Real classification'].iloc[i] == 1) or (real_pred['Predicted'].iloc[i] == 1):
			gir_cols_drop.append(i)
		if (real_pred['Real classification'].iloc[i] == 0) or (real_pred['Predicted'].iloc[i] == 0):
			mont_cols_drop.append(i)
	gir = gir.drop(gir_cols_drop)
	mont = mont.drop(mont_cols_drop)
	return ([gir, mont])


def plot_cdf(gir_array, mont_array):
	gir_probs = gir_array['Prob 0']
	mont_probs = mont_array['Prob 1']

	num_bins = 50
	"""counts, bin_edges = np.histogram(gir_probs, bins = num_bins)
	cdf = np.cumsum(counts)
	plt.plot(bin_edges[1:], cdf/cdf[-1])
	plt.savefig('gir_values.png')"""


	#fig, ax = plt.subplots(figsize=(8,4))
	
	plt.hist(gir_probs, bins = num_bins, histtype = 'step', density = True, cumulative = True, label = 'Gir')
	plt.hist(mont_probs, bins = num_bins, histtype = 'step', density = True, cumulative = True, label = 'Mont')
	plt.legend(loc = 'upper left')
	plt.show()
	plt.savefig('classification_values.png')


if __name__ == '__main__':
	import sys
	#real_pred = pickle.load(open("real_pred_lr.pickle", "rb"))
	real_pred = pickle.load(open("real_pred_xgb.pickle", "rb"))
	gir_array, mont_array = create_arrays(real_pred)
	plot_cdf(gir_array, mont_array)