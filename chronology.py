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
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, write_to_excel, convert_keys_to_string, compute_difference, cosine_similarity
from scipy import spatial

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

def calculate_distances(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze):
	speaker_ngrams = {}
	speakers_to_consider = []
	speaker_distances = collections.defaultdict()
	chronology = collections.defaultdict(dict)

	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	for identity in raw_speeches:
		date = re.findall(date_regex, str(identity))[0]
		speaker_name = speechid_to_speaker[identity]
		if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name in speakers_to_consider):
			indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
		# 	if speaker_name in speaker_ngrams:
		# 		speaker_ngrams[speaker_name] = speaker_ngrams[speaker_name] + indv_speech_bigram
		# 	else:
		# 		speaker_ngrams[speaker_name] = indv_speech_bigram
			for bigram in indv_speech_bigram:
				if bigram in chronology:
					chronology[bigram].append([speaker_name, identity, indv_speech_bigram[bigram]])
				else:
					chronology[bigram] = [speaker_name, identity, indv_speech_bigram[bigram]]
			# if speaker_name in chronology:
			# 	pairing = chronology[speaker_name]
			# 	for bigram in indv_speech_bigram:
			# 		if bigram in pairing:
			# 			pairing[bigram].append([identity, indv_speech_bigram[bigram]])
			# 		else:
			# 			pairing[bigram] = [identity, indv_speech_bigram[bigram]]
			# else:
			# 	chronology[speaker_name] = {}
			# 	pairing = chronology[speaker_name]
			# 	for bigram in indv_speech_bigram:
			# 		pairing[bigram] = []
			# 		# stores the unique speechid alongside the number of times that bigram is said in that speech for each bigram
			# 		pairing[bigram] = [identity, indv_speech_bigram[bigram]]

	


	df2 = pd.DataFrame.from_dict(chronology)
	df2 = df2.transpose()
	filename2 = "chronology.xlsx"
	writer2 = pd.ExcelWriter(filename2)
	df2.to_excel(writer2, 'Sheet1')
	writer2.save()

	pickle_filename_2 = "chronology.pickle"
	with open(pickle_filename_2, 'wb') as handle:
		pickle.dump(chronology, handle, protocol = 0)


def check_num_speakers(speech_data, speaker, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram].add(speaker)
		else:
			party_dict[bigram] = set()
			party_dict[bigram].add(speaker)
	return party_dict
	
# Computes the euclidean distance between two dictionaries of bigram to bigram frequency
def compute_distance(first_dict, second_dict):
	diff_counter = {}
	
	dict1 = first_dict
	dict2 = second_dict
	# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	"""for bigram in dict1:
		if bigram in dict2:
			diff_counter[bigram] = dict1[bigram] - dict2[bigram]

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
    speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_3.xlsx')

    # There are two tfidf dictionaries, one which only contains bigrams that appear in more than three speeches, below
    #gir_tfidf = process_excel('girondins_tfidf.xlsx')
    #mont_tfidf = process_excel("montagnards_tfidf.xlsx")

    # The second set of dictionaries contain all bigrams said in every speech
    # gir_tfidf = process_excel('girondins_tfidf_allbigrams.xlsx')
    # mont_tfidf = process_excel("montagnards_tfidf_allbigrams.xlsx")

    # doc_freq = pickle.load(open("bigram_doc_freq.pickle", "rb"))

    # file = open('num_speeches.txt', 'r')
    # num_speeches = int(file.read())
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

    calculate_distances(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze)
