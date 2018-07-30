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
from make_ngrams import compute_ngrams
import math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from processing_functions import print_to_csv, print_to_excel, load_list, process_excel, remove_diacritic, compute_tfidf, normalize_dicts
from lr_classification import run_train_classification, run_test_classification

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'
bigram_doc_freq = defaultdict(lambda: 0)
unigram_doc_freq = defaultdict(lambda: 0)


def aggregate(speakers_to_analyze_train, speakers_to_analyze_test, raw_speeches, speechid_to_speaker, Girondins, Montagnards):
	speaker_names = set()
	speakers_to_consider = []

	train_total_freq_unigram = {}
	test_total_freq_unigram = {}
	train_total_freq_bigram = {}
	test_total_freq_bigram = {}
	train_number_speeches = 0
	test_number_speeches = 0
	# Keeps track of which speeches contain the given bigram
	train_speeches_bigram = collections.defaultdict(dict)
	test_speeches_bigram = collections.defaultdict(dict)
	train_speeches_unigram = collections.defaultdict(dict)
	test_speeches_unigram = collections.defaultdict(dict)

	bigrams_to_speeches = collections.defaultdict()

	gir_num_speeches = 0
	mont_num_speeches = 0
	gir_docs = {}
	mont_docs = {}

	for speaker in speakers_to_analyze_train.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))
	for speaker in speakers_to_analyze_test.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	for speaker_name in speakers_to_consider:
		print speaker_name
		if speaker_name in speakers_to_analyze_train.index.values:
			party = speakers_to_analyze_train.loc[speaker_name, "Party"]
		else:
			party = speakers_to_analyze_test.loc[speaker_name, "Party"]
		speech = Counter()
		for identity in raw_speeches:
			date = re.findall(date_regex, str(identity))[0]
			if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name == speechid_to_speaker[identity]):
				indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
				indv_speech_unigram = compute_ngrams(raw_speeches[identity], 1)
				if speaker_name in speakers_to_analyze_train.index.values:
					train_number_speeches += 1
					for bigram in indv_speech_bigram:
						augment(bigram_doc_freq, bigram)
						augment(train_total_freq_bigram, bigram)
						if bigram in bigrams_to_speeches:
							bigrams_to_speeches[bigram].append(identity)
						else:
							bigrams_to_speeches[bigram] = []
							bigrams_to_speeches[bigram].append(identity)
					for unigram in indv_speech_unigram:
						augment(unigram_doc_freq, unigram)
						augment(train_total_freq_unigram, unigram)
					train_speeches_bigram[identity] = indv_speech_bigram
					train_speeches_unigram[identity] = indv_speech_unigram
				else:
					test_number_speeches += 1
					for bigram in indv_speech_bigram:
						#augment(bigram_doc_freq, bigram)
						augment(test_total_freq_bigram, bigram)
						if bigram in bigrams_to_speeches:
							bigrams_to_speeches[bigram].append(identity)
						else:
							bigrams_to_speeches[bigram] = []
							bigrams_to_speeches[bigram].append(identity)
					for unigram in indv_speech_unigram:
						#augment(unigram_doc_freq, unigram)
						augment(test_total_freq_unigram, unigram)
					test_speeches_bigram[identity] = indv_speech_bigram
					test_speeches_unigram[identity] = indv_speech_unigram


				### This is only using indv_speech_bigram, can do unigram if needbe
				if party == "Girondins":
					gir_num_speeches += 1
					gir_docs = check_num_speakers(indv_speech_bigram, speaker_name, gir_docs)
					try:
						Girondins = Girondins + indv_speech_bigram
					except NameError:
						Girondins = indv_speech_bigram
				else:
					mont_num_speeches += 1
					mont_docs = check_num_speakers(indv_speech_bigram, speaker_name, mont_docs)
					try:
						Montagnards = Montagnards + indv_speech_bigram
					except NameError:
						Montagnards = indv_speech_bigram
				#speech = speech + indv_speech_ngram
		#speaker_ngrams = compute_ngrams(speech)
		"""pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
		with open(pickle_filename, 'wb') as handle:
			pickle.dump(speech, handle, protocol = 0)"""
		

	model, train = run_train_classification(speechid_to_speaker, speakers_to_analyze_train, train_speeches_bigram, train_speeches_unigram, train_total_freq_bigram, train_total_freq_unigram, bigram_doc_freq, unigram_doc_freq, train_number_speeches)
	
	real_pred = run_test_classification(model, train, speechid_to_speaker, speakers_to_analyze_test, test_speeches_bigram, test_speeches_unigram, test_total_freq_bigram, test_total_freq_unigram, bigram_doc_freq, unigram_doc_freq, train_number_speeches)
	
	real_pred = pd.concat([real_pred, pd.DataFrame(columns = ['Speech Text'])])

	for i, index in enumerate(real_pred.index.values):
		if real_pred['Real classification'].iloc[i] != real_pred['Predicted'].iloc[i]:
			real_pred['Speech Text'].iloc[i] = raw_speeches[real_pred['Speechid'].iloc[i]]

	write_to = pd.ExcelWriter("predictions_with_speeches.xlsx")
	real_pred.to_excel(write_to, 'Sheet1')
	write_to.save()
	
	"""with open('bigrams_to_speeches.csv', 'wb') as outfile:
		writer = csv.writer(outfile)
		for key, val in bigrams_to_speeches.items():
			writer.writerow([key, val])

	Girondins = {k:v for k,v in Girondins.items() if (v >= 3)} #and (len(gir_docs[k]) > 1)}
	print_to_csv(Girondins, "Girondins_counts.csv")

	Montagnards = {k:v for k,v in Montagnards.items() if (v >= 3)} #and (len(mont_docs[k]) > 1)}
	print_to_csv(Montagnards, "Montagnards_counts.csv")

	print_to_excel(Girondins, Montagnards, 'combined_frequency.xlsx')

	num_speeches = gir_num_speeches + mont_num_speeches
	gir_tfidf = compute_tfidf(Girondins, num_speeches, bigram_doc_freq)
	mont_tfidf = compute_tfidf(Montagnards, num_speeches, bigram_doc_freq)

	#compute_distance(gir_tfidf, mont_tfidf)

	print_to_csv(gir_tfidf, 'gir_tfidf.csv')
	print_to_csv(mont_tfidf, 'mont_tfidf.csv')
	print_to_excel(gir_tfidf, mont_tfidf, 'combined_tfidf.xlsx')
	
	normalized = normalize_dicts(Girondins, Montagnards)
	compute_distance(normalized[0], normalized[1])"""


def augment(dictionary, ngram):
	if ngram in dictionary:
		dictionary[ngram] = dictionary[ngram] + 1
	else:
		dictionary[ngram] = 1

def check_num_speakers(speech_data, speaker, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram].add(speaker)
		else:
			party_dict[bigram] = set()
			party_dict[bigram].add(speaker)
	return party_dict
	

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


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speakers_to_analyze_train = load_list("Modified Girondins and Montagnards.xlsx")
    speakers_to_analyze_test = load_list("Girondins and Montagnards Test.xlsx")
    Girondins = Counter()
    Montagnards = Counter()
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate(speakers_to_analyze_train, speakers_to_analyze_test, raw_speeches, speechid_to_speaker, Girondins, Montagnards)
