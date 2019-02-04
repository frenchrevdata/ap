#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pickle
import pandas as pd
from pandas import *
import numpy as np
import csv
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

def calculate_chronology(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, Girondins, Montagnards):
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
			for bigram in indv_speech_bigram:
				if bigram in chronology:
					chronology[bigram].append([speaker_name, identity, indv_speech_bigram[bigram]])
				else:
					chronology[bigram] = []
					chronology[bigram].append([speaker_name, identity, indv_speech_bigram[bigram]])


	w = csv.writer(open("chronology.csv", "w"))
	for key, val in chronology.items():
		if (Girondins[key] >= 10) or (Montagnards[key] >= 10):
			w.writerow([key,val])


	pickle_filename_2 = "chronology.pickle"
	with open(pickle_filename_2, 'wb') as handle:
		pickle.dump(chronology, handle, protocol = 0)




if __name__ == '__main__':
	import sys
	raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
	speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
	Girondins = pickle.load(open("Girondins_withlimit.pickle", "rb"))
	Montagnards = pickle.load(open("Montagnards_withlimit.pickle", "rb"))

	# chronology = pickle.load(open("chronology.pickle", "rb"))
	speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_3.xlsx')

	speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

	calculate_chronology(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, Girondins, Montagnards)
