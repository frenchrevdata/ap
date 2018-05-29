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

raw_speeches = {}
speechid_to_speaker = {}
speaker_names = set()

global speakers_to_analyze
Girondins = Counter()
Montagnards = Counter()


def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def aggregate_by_speaker():
	for speechid in speechid_to_speaker:
		speaker_name = speechid_to_speaker[speechid]
		if (speaker_name in speakers_to_analyze.index.values) and (speaker_name not in speaker_names):
			print(speaker_name)
			speaker_names.add(speaker_name)
			speech = ""
			for identity in raw_speeches:
				if speaker_name == speechid_to_speaker[identity]:
					speech = speech + " " + raw_speeches[identity]
			speaker_ngrams = compute_ngrams(speech)
			pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
			with open(pickle_filename, 'wb') as handle:
				pickle.dump(speaker_ngrams, handle, protocol = 0)

def aggregate_by_group():
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
    speakers_to_analyze = load_list("Girondins and Montagnards.xlsx")
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    #aggregate_by_speaker()
    aggregate_by_group()