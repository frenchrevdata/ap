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
from processing_functions import print_to_csv, print_to_excel, load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts


"""calculate the frequency of each speaker
compute distance to girondins frequency vector
compute distance to montagnards frequency vector
add dict of {speaker: [dist to Gir, dist to Montagnard]}"""

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'


def build_vectors(raw_speeches, speechid_to_speaker, speaker_list, gir_frequency, mont_frequency):
	speaker_ngrams = {}
	speaker_distances = collections.defaultdict()
	for identity in raw_speeches:
		### These two following lines are dependent on what time period to do this distance analysis on
		date = re.findall(date_regex, str(identity))[0]
		if (date >= "1792-09-20") and (date <= "1793-06-02"):
			speaker_name = speechid_to_speaker[identity]
			indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
			if speaker_name in speaker_ngrams:
				speaker_ngrams[speaker_name] = speaker_ngrams[speaker_name] + indv_speech_bigram
			else:
				speaker_ngrams[speaker_name] = indv_speech_bigram
	for speaker in speaker_ngrams:
		to_compare = {k:v for k,v in speaker_ngrams[speaker].items() if (v >= 3)}
		normalized = normalize_dicts(to_compare, gir_frequency)
		gir_dist = 	compute_distance(normalized[0], normalized[1])
		normalized = normalize_dicts(to_compare, mont_frequency)
		mont_dist = compute_distance(normalized[0], normalized[1])
		speaker_distances[speaker] = [gir_dist, mont_dist]

	df = pd.DataFrame.from_dict(speaker_distances)
	df = df.transpose()
	df.columns = ["dist to Girondins", "dist to Montagnards"]
	filename = "freq_dist_map.xlsx"
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()


def check_num_speakers(speech_data, speaker, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram].add(speaker)
		else:
			party_dict[bigram] = set()
			party_dict[bigram].add(speaker)
	return party_dict
	

def compute_distance(dict1, dict2):
	diff_counter = {}
	
	"""# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	for bigram in Girondins:
		if bigram in Montagnards:
			diff_counter[bigram] = Girondins[bigram] - Montagnards[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	return(euclidean_distance)
	#print(euclidean_distance)
	#print("---------")"""

	## When every bigram accounted for
	diff_counter = {}
	for bigram in dict2:
		if bigram in dict1:
			diff_counter[bigram] = dict1[bigram] - dict2[bigram]
	for bigram in dict1:
		if bigram not in dict2:
			diff_counter[bigram] = dict1[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	return(euclidean_distance)
	#print(euclidean_distance)


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_2.xlsx')
    gir_frequency = process_excel('Girondins_frequency.xlsx')
    mont_frequency = process_excel("Montagnards_frequency.xlsx")

    build_vectors(raw_speeches, speechid_to_speaker, speaker_list, gir_frequency, mont_frequency)
