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
	by_month = pd.read_excel("By_Month.xlsx")
    by_date = pd.read_excel("By_Date.xlsx")

    by_month = create_tfidf_vectors(by_month)
    by_month_dist = compute_distances(by_month, 'month')

    by_date = create_tfidf_vectors(by_date)
    by_date_dist = compute_distances(by_date, 'date')

    by_month_convention = pd.read_excel("By_Month_Convention.xlsx")
    by_date_convention = pd.read_excel("By_Date_Convention.xlsx")

    by_month_convention = create_tfidf_vectors(by_month_convention)
    by_month_convention_dist = compute_distances(by_month_convention, 'month')

    by_date_convention = create_tfidf_vectors(by_date_convention)
    by_date_convention_dist = compute_distances(by_date_convention, 'date')


def gen_bigrams(text):
	compute_ngrams(text, 2)

def gen_tfidf(vector):
	compute_tfidf(vector, num_speeches, doc_freq)

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

def compute_distances(dataframe, period):
	period_vector = pd.Series()
	if period == 'month':
		period_vector = dataframe['Month']
	if period == 'date':
		period_vector = dataframe['Date'].tolist()

	gir_dict = pickle.load(open("gir_dict.pickle", "rb"))
    mont_dict = pickle.load(open("mont_dict.pickle", "rb"))
    gir_mont_diff = pickle.load(open("gir_mont_diff.pickle", "rb"))

    tfidf_scores = dataframe['tfidf'].tolist()
    for counter in tfidf_scores:
    	gir_dist.append(1 - cosine_similarity(gir_dict, tfidf_scores))
    	mont_dist.append(1 - cosine_similarity(mont_dict, tfidf_scores))
    	gir_mont_diff_dist.append(cosine_similarity(gir_mont_diff, tfidf_scores))

    gir_dist = pd.Series(gir_dist)
    mont_dist = pd.Series(mont_dist)
    gir_mont_diff_dist = pd.Series(gir_mont_diff_dist)
    comp_df = pd.DateFrame([period_vector, gir_dist, mont_dist, gir_mont_diff_dist])
    return comp_df


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