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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from processing_functions import compute_tfidf
from xgboost import XGBClassifier

def run_train_classification(speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches):
	iteration = "train"
	train, train_classification, speeches, speakers = data_clean(iteration, speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches)

	"""### Remove procedural and other generic language
	train.columns = train.columns.map(str)

	columns_to_drop = ["(u'salut',)", "(u'veux',)", "(u'``',)", "(u'voici',)", "(u'surtout',)", "(u'sais',)", "(u'quil',)", "(u'pu',)", "(u'peut-etre',)", "(u'lui-meme',)", "(u'lesquels',)", "(u'elle-meme',)", "(u'ae',)"]
	for column in columns_to_drop:
		test.drop(column, axis = 1)

	train = train.drop(columns = columns_to_drop, axis = 1)"""

	print len(train.index)


	### Logistic Regression
	model = LogisticRegression()
	model.fit(train.get_values(), train_classification)
	predicted = cross_validation.cross_val_predict(model, train.get_values(), train_classification, cv = 10)

	"""### xgboost
	model = XGBClassifier()
	model.fit(train.get_values(), train_classification)
	predicted = cross_validation.cross_val_predict(model, train.get_values(), train_classification, cv = 10)"""


	print ("Training CV Score: " + str(metrics.accuracy_score(train_classification, predicted)))
	#print ("Test CV Score: " + str(model.score(x_test, y_test)))

	train_classification = pd.DataFrame(train_classification)
	speeches = pd.DataFrame(speeches, columns = ['Speechid'])
	speakers = pd.DataFrame(speakers, columns = ['Speaker'])
	
	"""train_total = train.copy()
	train_total.join(train_classification)
	train_total.join(speeches)"""
	train_total = pd.concat([train, train_classification, speeches, speakers], axis = 1)
	writer = pd.ExcelWriter("training_data.xlsx")
	train_total.to_excel(writer, 'Sheet1')
	writer.save()


	return [model, train]
	#return logreg

def run_test_classification(model, train, speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches):
	iteration = "test"
	test, test_classification, speeches, speakers = data_clean(iteration, speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches)

	"""test_pred = model.predict(test.get_values())
	accuracy = metrics.accuracy_score(test_classification, test_pred)"""


	#test.columns = test.columns.map(str)
	print len(test.index)

	# Remove any columns not in the training set
	cols_to_keep = []
	for column in test.columns:
		if column in train.columns:
			cols_to_keep.append(column)
	test = test[cols_to_keep]

	# Add columns of zero for features not in the test set but in the training set
	"""np.zeros
	nrows is number rows in the DataFrame
	ncols
	convert to dataframe and pass in names of columns
	concatenate to other dataframe"""

	col_names = []
	for col in train.columns:
		if col not in test.columns:
			test[col] = 0
			#col_names.append(col)

	"""numrows = test.shape[0]
	numcols = len(col_names)
	print col_names
	zero_array = np.zeros((numrows, numcols))
	zero_array = pd.DataFrame(zero_array, columns = col_names)

	#test.reset_index(drop = True)
	test.append(zero_array, ignore_index = True)
	
	#test_mod = pd.concat([test, zero_array], ignore_index = True)

	#test.merge(zero_array, right_index = True).reset_index()"""

	test = test[train.columns]

	print "Test CV Score: " + str(model.score(test.get_values(), test_classification))

	test_classification_df = pd.DataFrame(test_classification, columns = ['Real classification'])
	speeches = pd.DataFrame(speeches, columns = ['Speechid'])
	speakers = pd.DataFrame(speakers, columns = ['Speaker'])

	predictions = model.predict(test.get_values())
	predicted_values = pd.DataFrame(predictions, columns = ['Predicted'])

	print confusion_matrix(test_classification, predicted_values)

	predict_prob = pd.DataFrame(model.predict_proba(test.get_values()), columns = ['Prob 0', 'Prob 1'])
	
	real_pred = pd.concat([test_classification_df, predicted_values, predict_prob, speeches, speakers], axis = 1)

	write_to = pd.ExcelWriter("predictions.xlsx")
	real_pred.to_excel(write_to, 'Sheet1')
	write_to.save()

	
	#test.join(test_classification)
	#test.join(speeches)
	test_total = pd.concat([test, test_classification_df, speeches, speakers], axis = 1)
	writer = pd.ExcelWriter("test_data.xlsx")
	#test_total.to_excel(writer, 'Sheet1')
	test_total.to_excel(writer, 'Sheet1')
	writer.save()

	return real_pred

### If doing train and test separately
def data_clean(iteration, speechid_to_speaker, speakers_to_analyze, bigram_speeches, unigram_speeches, bigram_freq, unigram_freq, bigram_doc_freq, unigram_doc_freq, num_speeches):
	classification = []
	data_set = []
	speeches = []
	speakers = []
	### Should I do this once for all the data and then split it into test and train? That way all the data is based on the same bigrams. Or is that
	### bad because then the training data is connected to the test data via the tfidf calculations?
	for speechid in bigram_speeches:
		speaker = speechid_to_speaker[speechid]

		#create a vector of speechids in correct order to reverse engineer and check which speeches were/were not correctly classified
		speeches.append(speechid)
		speakers.append(speaker)

		if speakers_to_analyze.loc[speaker, "Party"] == "Girondins":
			classification.append(0)
		else:
			classification.append(1)
		# add some doc freq cutoff here
		if iteration == "train":
			bigram_input = {k:v for k,v in bigram_speeches[speechid].items() if (bigram_freq[k] >= 7)}
			unigram_input = {k:v for k,v in unigram_speeches[speechid].items() if (unigram_freq[k] >= 50)}
			
			bigram_scores = compute_tfidf(bigram_input, num_speeches, bigram_doc_freq)
			unigram_scores = compute_tfidf(unigram_input, num_speeches, unigram_doc_freq)
		else:
			bigram_scores = compute_tfidf(bigram_speeches[speechid], num_speeches, bigram_doc_freq)
			unigram_scores = compute_tfidf(unigram_speeches[speechid], num_speeches, unigram_doc_freq)

		
		merge_scores = bigram_scores.copy()
		merge_scores.update(unigram_scores)
		
		data_set.append(merge_scores)

	data = pd.DataFrame(data_set)
	data = data.fillna(0)

	return([data, classification, speeches, speakers])