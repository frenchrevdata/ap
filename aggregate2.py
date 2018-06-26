#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
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
import gzip
from make_ngrams import compute_ngrams
import math
from collections import defaultdict

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'
doc_freq = defaultdict(lambda: 0)

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker, Girondins, Montagnards):
	speaker_names = set()
	speakers_to_consider = []
	gir_num_speeches = 0
	mont_num_speeches = 0
	# Keeps track of which speeches contain the given bigram
	bigrams_speeches = collections.defaultdict()
	gir_doc_freq = collections.defaultdict()
	mont_doc_freq = collections.defaultdict()
	speeches = collections.defaultdict(dict)
	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))
	for speaker_name in speakers_to_consider:
		print speaker_name
		party = speakers_to_analyze.loc[speaker_name, "Party"]
		speech = Counter()
		for identity in raw_speeches:
			date = re.findall(date_regex, str(identity))[0]
			if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name == speechid_to_speaker[identity]):
				indv_speech_ngram = compute_ngrams(raw_speeches[identity])
				for bigram in indv_speech_ngram:
					if bigram in bigrams_speeches:
						bigrams_speeches[bigram].append(identity)
					else:
						bigrams_speeches[bigram] = []
						bigrams_speeches[bigram].append(identity)
					if bigram in doc_freq:
						doc_freq[bigram] = doc_freq[bigram] + 1
					else:
						doc_freq[bigram] = 1

				speeches[identity] = indv_speech_ngram
				"""if party == "Girondins":
					gir_num_speeches += 1
					gir_doc_freq = check_num_speakers(indv_speech_ngram, speaker_name, gir_doc_freq)
					try:
						Girondins = Girondins + indv_speech_ngram
					except NameError:
						Girondins = indv_speech_ngram
				else:
					mont_num_speeches += 1
					mont_doc_freq = check_num_speakers(indv_speech_ngram, speaker_name, mont_doc_freq)
					try:
						Montagnards = Montagnards + indv_speech_ngram
					except NameError:
						Montagnards = indv_speech_ngram"""
				#speech = speech + indv_speech_ngram
		#speaker_ngrams = compute_ngrams(speech)
		"""pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
		with open(pickle_filename, 'wb') as handle:
			pickle.dump(speech, handle, protocol = 0)"""

	classification = pd.Series()
	training_set = pd.DataFrame()
	for speechid in speeches:
		speaker = speechid_to_speaker[speechid]
		if speakers_to_analyze.loc[speaker, "Party"] == "Girondins":
			classification.append(0)
		else:
			classification.append(1)
		scores = compute_tfidf(dict(speeches[speechid]))
		training_set = pd.DateFrame([training_set, scores])
	print_to_csv(training_set, 'training_data.csv')









	"""with open('bigrams_to_speeches.csv', 'wb') as outfile:
		writer = csv.writer(outfile)
		for key, val in bigrams_speeches.items():
			writer.writerow([key, val])

	Girondins = {k:v for k,v in Girondins.items() if (v >= 3)} #and (len(gir_doc_freq[k]) > 1)}
	print_to_csv(Girondins, "Girondins_counts.csv")

	Montagnards = {k:v for k,v in Montagnards.items() if (v >= 3)} #and (len(mont_doc_freq[k]) > 1)}
	print_to_csv(Montagnards, "Montagnards_counts.csv")

	print_to_excel(Girondins, Montagnards, 'combined_frequency.xlsx')

	num_speeches = gir_num_speeches + mont_num_speeches
	gir_tfidf = compute_tfidf(Girondins, num_speeches)
	mont_tfidf = compute_tfidf(Montagnards, num_speeches)

	#compute_distance(gir_tfidf, mont_tfidf)

	print_to_csv(gir_tfidf, 'gir_tfidf.csv')
	print_to_csv(mont_tfidf, 'mont_tfidf.csv')
	print_to_excel(gir_tfidf, mont_tfidf, 'combined_tfidf.xlsx')
	
	normalized = normalize_dicts(Girondins, Montagnards)
	compute_distance(normalized[0], normalized[1])"""


def check_num_speakers(speech_data, speaker, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram].add(speaker)
		else:
			party_dict[bigram] = set()
			party_dict[bigram].add(speaker)
	return party_dict

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
	

def compute_distance(Girondins, Montagnards):
	diff_counter = {}
	
	# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	for bigram in Girondins:
		if bigram in Montagnards:
			diff_counter[bigram] = Girondins[bigram] - Montagnards[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	#print(euclidean_distance)
	#print("---------")

	## When every bigram accounted for
	diff_counter = {}
	for bigram in Montagnards:
		if bigram in Girondins:
			diff_counter[bigram] = Girondins[bigram] - Montagnards[bigram]
	for bigram in Girondins:
		if bigram not in Montagnards:
			diff_counter[bigram] = Girondins[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	#print(euclidean_distance)

def normalize_dicts(Girondins, Montagnards):
	# Normalize counts
	all_sum = 0
	all_sum = all_sum + sum(Girondins.values()) + sum(Montagnards.values())
	
	for key in Girondins:
		Girondins[key] = float(Girondins[key])/all_sum

	for key in Montagnards:
		Montagnards[key] = float(Montagnards[key])/all_sum

	print_to_excel(Girondins, Montagnards, 'combined_normalized.xlsx')
	print_to_csv(Girondins, "Girondins_counts_normalized.csv")
	print_to_csv(Montagnards, "Montagnards_counts_normalized.csv")

	return([Girondins, Montagnards])


def compute_tfidf(dictionary, num_speeches):
	tfidf = {}
	for bigram in dictionary:
		idf = math.log10(num_speeches) - math.log10(doc_freq[bigram])
		tf = dictionary[bigram]
		tfidf[bigram] = (1+math.log10(tf))*idf
	return tfidf
	

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
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speakers_to_analyze = load_list("Modified Girondins and Montagnards.xlsx")
    Girondins = Counter()
    Montagnards = Counter()
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker, Girondins, Montagnards)
