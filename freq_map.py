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


def build_vectors(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, gir_frequency, mont_frequency):
	speaker_ngrams = {}
	speakers_to_consider = []
	speaker_distances = collections.defaultdict()
	chronology = collections.defaultdict(dict)

	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	for identity in raw_speeches:
		### These two following lines are dependent on what time period to do this distance analysis on
		date = re.findall(date_regex, str(identity))[0]
		speaker_name = speechid_to_speaker[identity]
		if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name in speakers_to_consider):
			indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
			if speaker_name in speaker_ngrams:
				speaker_ngrams[speaker_name] = speaker_ngrams[speaker_name] + indv_speech_bigram
			else:
				speaker_ngrams[speaker_name] = indv_speech_bigram
		"""
		if speaker_name in chronology:
			pairing = chronology[speaker_name]
			for bigram in indv_speech_bigram:
				if bigram in pairing:
					pairing[bigram].append([identity, indv_speech_bigram[bigram]])
				else:
					pairing[bigram] = [identity, indv_speech_bigram[bigram]]
		else:
			chronology[speaker_name] = {}
			pairing = chronology[speaker_name]
			for bigram in indv_speech_bigram:
				pairing[bigram] = []
				# stores the unique speechid alongside the number of times that bigram is said in that speech for each bigram
				pairing[bigram] = [identity, indv_speech_bigram[bigram]]"""

	
	#########
	## Need to figure out how to merge based on bigrams
	gir = pd.DataFrame.from_dict(gir_frequency, orient = "index")
	mont = pd.DataFrame.from_dict(mont_frequency, orient = "index")
	bohan = pd.DataFrame.from_dict(speaker_ngrams['Alain Bohan'], orient = "index")
	df2 = pd.concat([bohan, gir, mont], axis = 1)
	writer_new = pd.ExcelWriter("freq_test.xlsx")
	df2.to_excel(writer_new, 'Sheet1')
	writer_new.save()

	for speaker in speaker_ngrams:
		#to_compare = {k:v for k,v in speaker_ngrams[speaker].items() if (v >= 3)}
		to_compare = speaker_ngrams[speaker]
		gir_dict = gir_frequency
		mont_dict = mont_frequency
		gir_normalized = normalize_dicts(to_compare, gir_dict)
		gir_dist = 	compute_distance(gir_normalized[0], gir_normalized[1])
		to_compare = speaker_ngrams[speaker]
		mont_normalized = normalize_dicts(to_compare, mont_dict)
		mont_dist = compute_distance(mont_normalized[0], mont_normalized[1])
		speaker_distances[speaker] = [gir_dist, mont_dist]

	

	
	pickle_filename_3 = "speaker_ngrams.pickle"
	with open(pickle_filename_3, 'wb') as handle:
		pickle.dump(speaker_ngrams, handle, protocol = 0)

	df = pd.DataFrame.from_dict(speaker_distances)
	df = df.transpose()
	df.columns = ["dist to Girondins", "dist to Montagnards"]
	filename = "freq_dist_map.xlsx"
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()

	pickle_filename = "freq_dist.pickle"
	with open(pickle_filename, 'wb') as handle:
		pickle.dump(speaker_distances, handle, protocol = 0)

	"""df2 = pd.DataFrame.from_dict(chronology)
	df2 = df2.transpose()
	filename2 = "chronology.xlsx"
	writer2 = pd.ExcelWriter(filename)
	df2.to_excel(writer2, 'Sheet1')
	writer2.save()

	pickle_filename_2 = "chronology.pickle"
	with open(pickle_filename_2, 'wb') as handle:
		pickle.dump(chronology, handle, protocol = 0)"""


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
    """#frequency bigrams when only bigrams occuring more than 3 times accounted for
    gir_frequency = process_excel('girondins_frequency.xlsx')
    mont_frequency = process_excel("montagnards_frequency.xlsx")"""
    #frequency vectors when all possible bigrams accounted for
    gir_frequency = process_excel('girondins_frequency_all.xlsx')
    mont_frequency = process_excel("montagnards_frequency_all.xlsx")
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

    build_vectors(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, gir_frequency, mont_frequency)
