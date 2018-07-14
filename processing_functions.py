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


def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def print_to_csv(dictionary, filename):
	output_file = filename
	with open(filename, mode='w') as f:
		f.write('Bigrams|freq\n')
		for bigram, count in dictionary.items():
			f.write('{}|{}\n'.format(bigram, count))

def print_to_excel(dict1, dict2, filename):
	df = pd.DataFrame([dict1, dict2])
	df = df.transpose()
	df.columns = ["Girondins", "Montagnards"]
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()

def load_list(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name = 'Sheet1')
	pd_list = pd_list.set_index('Name')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'AP Speaker Authority List xlsx')
	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

def process_excel(filename):
	xls = ExcelFile(filename)
	first = xls.parse(xls.sheet_names[0])
	first = first.set_index('Bigrams')
	first = first.fillna(0)
	second = first.to_dict()
	for entry in second:
		third = second[entry]
	for item in third:
		third[item] = int(third[item])
	return(third)

def compute_tfidf(dictionary, num_speeches, doc_freq):
	tfidf = {}
	for ngram in dictionary:
		if ngram in doc_freq:
			df = doc_freq[ngram]
		else:
			df = 1
		idf = math.log10(num_speeches) - math.log10(df)
		tf = dictionary[ngram]
		tfidf[ngram] = (1+math.log10(tf))*idf
	return tfidf

def normalize_dicts(first_dict, second_dict):
	# Normalize counts
	dict1 = first_dict
	dict2 = second_dict
	all_sum = 0
	all_sum = all_sum + sum(dict1.values()) + sum(dict2.values())
	
	for key in dict1:
		dict1[key] = float(dict1[key])/all_sum

	for key in dict2:
		dict2[key] = float(dict2[key])/all_sum

	"""print_to_excel(Girondins, Montagnards, 'combined_normalized.xlsx')
	print_to_csv(Girondins, "Girondins_counts_normalized.csv")
	print_to_csv(Montagnards, "Montagnards_counts_normalized.csv")"""

	return([dict1, dict2])