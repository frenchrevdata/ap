#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
import csv
import pickle
import regex as re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import gzip
from make_ngrams import compute_ngrams
import math


def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def aggregate_by_group(speakers_to_analyze, Girondins, Montagnards):
	files = os.listdir("../Speakers/")
	for filename in files:
		with open('../Speakers/' + filename, "r") as f:
			speaker_data = pickle.load(f)
		#speaker_data = pickle.load(open('../Speakers/' + filename, "r"))
		speaker = re.findall(r'([a-zA-Z\- \']+)_ngrams.pickle', filename)[0]
		party = speakers_to_analyze.loc[speaker, "Party"]
		if party == "Girondins":
			try:
				Girondins = Girondins + speaker_data
			except NameError:
				Girondins = speaker_data
		else:
			try:
				Montagnards = Montagnards + speaker_data
			except NameError:
				Montagnards = speaker_data

	Gir_output_file = "Girondins_counts.csv"
	with open(Gir_output_file, mode='w') as gf:
		gf.write('Bigrams|freq\n')
		for bigram, count in Girondins.items():
			if count >= 3:
				gf.write('{}|{}\n'.format(bigram, count))
	Mont_output_file = "Montagnards_counts.csv"
	with open(Mont_output_file, mode='w') as mf:
		mf.write('Bigrams|freq\n')
		for bigram, count in Montagnards.items():
			if count >= 3:
				mf.write('{}|{}\n'.format(bigram, count))

	compute_distance(Girondins, Montagnards)
	
def compute_distance(Girondins, Montagnards):
	diff_counter = {}

	# Normalize counts
	all_sum = 0
	for key in Girondins:
		all_sum = all_sum + Girondins[key]
	
	for key in Montagnards:
		all_sum = all_sum + Montagnards[key]


	for key in Girondins:
		if Girondins[key] >= 3:
			Girondins[key] = float(Girondins[key])/all_sum
		else:
			Girondins[key] = 0

	for key in Montagnards:
		if Montagnards[key] >= 3:
			Montagnards[key] = float(Montagnards[key])/all_sum
		else:
			Montagnards[key] = 0


	# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	"""for bigram in Girondins:
		if bigram in Montagnards:
			diff_counter[bigram] = Girondins[bigram] - Montagnards[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	print(euclidean_distance)"""

	## When every bigram accounted for
	for bigram in Montagnards:
		if bigram in Girondins:
			Montagnards[bigram] = Girondins[bigram] - Montagnards[bigram]
	for bigram in Girondins:
		if bigram not in Montagnards:
			Montagnards[bigram] = Girondins[bigram]

	sum_of_squares = 0
	for entry in Montagnards:
		sum_of_squares = sum_of_squares + math.pow(Montagnards[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	print(euclidean_distance)

def load_list(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'Sheet1')
	pd_list = pd_list.set_index('Name')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

if __name__ == '__main__':
    import sys
    Girondins = Counter()
    Montagnards = Counter()
    speakers_to_analyze = load_list("Girondins and Montagnards.xlsx")
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate_by_group(speakers_to_analyze, Girondins, Montagnards)
