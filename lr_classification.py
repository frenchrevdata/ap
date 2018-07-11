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
from sklearn import metrics, cross_validation, preprocessing
from processing_functions import compute_tfidf
from xgboost import XGBClassifier

def run_train_classification(speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches):
	train, train_classification = data_clean(speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches)
	
	writer = pd.ExcelWriter("training_set.xlsx")
	train.to_excel(writer, 'Sheet1')
	writer.save()

	### Logistic Regression
	model = LogisticRegression()
	model.fit(train.get_values(), train_classification)
	predicted = cross_validation.cross_val_predict(model, train.get_values(), train_classification, cv = 10)

	"""### xgboost
	model = XGBClassifier()
	model.fit(train.get_values(), train_classification)
	predicted = cross_validation.cross_val_predict(model, train.get_values(), train_classification, cv = 10)"""

	print ("Training CV Score: " + str(metrics.accuracy_score(train_classification, predicted)))

	return [model, train]
	#return logreg

def run_test_classification(model, train, speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches):
	test, test_classification = data_clean(speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches)

	"""test_pred = model.predict(test.get_values())
	accuracy = metrics.accuracy_score(test_classification, test_pred)"""

	test = test[train.columns]

	print "Test CV Score: " + str(model.score(test.get_values(), test_classification))


def data_clean(speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches):
	classification = []
	data_set = []
	for speechid in bigram_speeches:
		speaker = speechid_to_speaker[speechid]
		if speakers_to_analyze.loc[speaker, "Party"] == "Girondins":
			classification.append(0)
		else:
			classification.append(1)
		# add some doc freq cutoff here
		bigram_input = {k:v for k,v in bigram_speeches[speechid].items() if (bigram_freq[k] >= 10)}
		unigram_input = {k:v for k,v in unigram_speeches[speechid].items() if (unigram_freq[k] >= 55)}
		
		bigram_scores = compute_tfidf(bigram_input, num_speeches, bigram_doc_freq)
		unigram_scores = compute_tfidf(unigram_input, num_speeches, unigram_doc_freq)
		
		merge_scores = bigram_scores.copy()
		merge_scores.update(unigram_scores)
		
		data_set.append(merge_scores)

	data = pd.DataFrame(data_set)
	data = data.fillna(0)

	return([data, classification])