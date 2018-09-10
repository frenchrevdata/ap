#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
This file is a collection of functions used multiple times throughout other scripts.
It avoids a repetition of data.
"""

import pickle
import unicodedata
import pandas as pd
from pandas import *
import numpy as np
import math


# Removes accents from the input parameter
def remove_diacritic(input):
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

"""
def print_to_excel(dict1, dict2, filename):
	df = pd.DataFrame([dict1, dict2])
	df = df.transpose()
	df.columns = ["Girondins", "Montagnards"]
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()
"""

# Writes the given dataframe to the provided filename
# Assumes that the df parameter is a pandas dataframe
def write_to_excel(df, filename):
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()

# Loads an unspecified list of speakers into a pandas list from Excel
def load_list(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name = 'Sheet1')
	pd_list = pd_list.set_index('Name')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

# Loads the comprehensive speaker list into a pandas dataframe from Excel
def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'AP Speaker Authority List xlsx')
	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

# Imports a given Excel file for use in the Girondins/Montagnards analysis
# Make sure that the header of the first column in "Bigrams"
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

# Computes the tf-idf score for given parameters
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

# Normalizes dictionaries for use in computing the distance between the two vectors
# Returns an array of the two normalized dictionaries
def normalize_dicts(first_dict, second_dict):
	dict1 = first_dict
	dict2 = second_dict
	all_sum = 0
	all_sum = all_sum + sum(dict1.values()) + sum(dict2.values())
	
	for key in dict1:
		dict1[key] = float(dict1[key])/all_sum

	for key in dict2:
		dict2[key] = float(dict2[key])/all_sum

	return([dict1, dict2])