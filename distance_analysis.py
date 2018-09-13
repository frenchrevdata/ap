#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pickle
import pandas as pd
from pandas import *
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import regex as re
from make_ngrams import compute_ngrams
import math
from collections import defaultdict
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, write_to_excel
from scipy import spatial

global num_speeches
doc_freq = pickle.load(open("bigram_doc_freq.pickle", "rb"))

# Compute distances to Gir and Mont tfidf vectors
# Need to import excel files
# Need to compute ngrams and then tfidf for each grouping
# Then compute cosine similarity/difference
# In freq_map write the tfidf dicts and the diff dict to memory to import in this file


def distance_analysis():

	gir_tfidf = process_excel('girondins_tfidf_allbigrams.xlsx')
	mont_tfidf = process_excel("montagnards_tfidf_allbigrams.xlsx")

	gir_dict = convert_keys_to_string(gir_tfidf)
	mont_dict = convert_keys_to_string(mont_tfidf)
	gir_mont_diff = compute_difference(gir_dict, mont_dict)

	by_month = pd.read_excel("By_Month.xlsx")
	by_date = pd.read_excel("By_Date.xlsx")

	by_month = create_tfidf_vectors(by_month)
	by_month_dist = compute_distances(by_month, 'month',  gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_month_dist, 'by_month_distances.xlsx')

	by_date = create_tfidf_vectors(by_date)
	by_period = aggregate_by_period(by_date)

	by_period_dist = compute_distances(by_period, 'period', gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_period_dist, "by_period_distances.xlsx")

	by_date_dist = compute_distances(by_date, 'date',  gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_date_dist, 'by_date_distances.xlsx')

	
def gen_bigrams(text):
	compute_ngrams(text, 2)

def gen_tfidf(vector):
	compute_tfidf(vector, num_speeches, doc_freq)

def aggregate_by_period(dataframe):
	before_convention = Counter()
	convention = Counter()
	after_convention = Counter()
	for i, time in enumerate(dataframe['Full Date']):
		time = str(time)
		if (time >= "1792-6-10") and (time <= "1792-8-10"):
			before_convention = before_convention + dataframe['ngrams'].iloc[i]
		if (time >= "1792-9-20") and (time < "1793-6-2"):
			convention = convention + dataframe['ngrams'].iloc[i]
		if (time >= "1793-6-2") and (time <= "1793-8-2"):
			after_convention = after_convention + dataframe['ngrams'].iloc[i]

	before_convention_tfidf = compute_tfidf(before_convention, num_speeches, doc_freq)
	convention_tfidf = compute_tfidf(convention, num_speeches, doc_freq)
	after_convention_tfidf = compute_tfidf(after_convention, num_speeches, doc_freq)

	before_convention_df = pd.DataFrame.from_dict(before_convention_tfidf, orient = "index")
	convention_df = pd.DataFrame.from_dict(convention_tfidf, orient = "index")
	after_convention_df = pd.DataFrame.from_dict(after_convention_tfidf, orient = "index")

	#period_df = pd.DataFrame([before_convention, convention, after_convention])
	#write_to_excel(period_df, 'periods.xlsx')

	period_df = [before_convention_tfidf, convention_tfidf, after_convention_tfidf]
	return period_df
	

def create_tfidf_vectors(dataframe):
	speeches = dataframe['concat_speeches'].tolist()
	ngrams = []
	for unit in speeches:
		ngrams.append(compute_ngrams(unit, 2))
	ngrams_to_add = pd.Series(ngrams)
	dataframe['ngrams'] = ngrams_to_add.values
	tfidf = []
	for element in ngrams:
		tfidf.append(compute_tfidf(element, num_speeches, doc_freq))
	tfidf_to_add = pd.Series(tfidf)
	dataframe['tfidf'] = tfidf_to_add.values
	return dataframe

def compute_distances(dataframe, period, gir_dict, mont_dict, gir_mont_diff):
	period_vector = pd.Series()
	if period == 'month':
		period_vector = dataframe['Year-Month'].tolist()
		period_vector = pd.Series(period_vector)
		tfidf_scores = dataframe['tfidf'].tolist()
	elif period == 'date':
		period_vector = dataframe['Date'].tolist()
		period_vector = pd.Series(period_vector)
		tfidf_scores = dataframe['tfidf'].tolist()
	else:
		periods = ["Before convention", "Convention", "After convention"]
		period_vector = pd.Series(periods)
		tfidf_scores = dataframe

	#tfidf_scores = dataframe['tfidf'].to_dict()
	gir_dist = []
	mont_dist = []
	gir_mont_diff_dist = []
	for counter in tfidf_scores:
	#for counter in tfidf_scores.items():
		# The way that extraction of a column from a dataframe works, it returns a tuple and the data is the second entry
		#to_compare = counter[1]
		to_compare = counter
		to_compare = convert_keys_to_string(to_compare)
		gir_dist.append(1 - cosine_similarity(gir_dict, to_compare))
		mont_dist.append(1 - cosine_similarity(mont_dict, to_compare))
		gir_mont_diff_dist.append(cosine_similarity(gir_mont_diff, to_compare))

	gir_dist = pd.Series(gir_dist)
	mont_dist = pd.Series(mont_dist)
	gir_mont_diff_dist = pd.Series(gir_mont_diff_dist)
	comp_df = pd.DataFrame([period_vector, gir_dist, mont_dist, gir_mont_diff_dist])
	comp_df = comp_df.transpose()
	comp_df.columns = [period, 'distance to gir', 'distance to mont', 'distance to diff']
	return comp_df

# Need this function because the keys for one of the dictionaries were not strings so the match check
# in cosine similarity was not working
def convert_keys_to_string(dictionary):
	to_return = {}
	for k in dictionary:
		to_return[str(k)] = dictionary[k]
	return to_return

def compute_difference(dict1, dict2):
	diff = {}
	for k in dict1:
		diff[k] = dict1[k] - dict2[k]
	return diff 


def cosine_similarity(dict1, dict2):
	numerator = 0
	denom1 = 0
	denom2 = 0
	for k in dict2:
		if k in dict1:
			numerator += dict1[k]*dict2[k]
		denom2 += dict2[k]*dict2[k]
	for v in dict1.values():
		denom1 += v*v
	cos_sim = numerator/math.sqrt(denom1*denom2)
	return cos_sim

if __name__ == '__main__':
    import sys
    file = open('num_speeches.txt', 'r')
    num_speeches = int(file.read())

    distance_analysis()