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

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def aggregate_by_speaker(speakers_to_analyze, raw_speeches, speechid_to_speaker):
	speaker_names = set()
	speakers_to_consider = []
	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))
	for speaker_name in speakers_to_consider:
		print speaker_name
		speech = ""
		for identity in raw_speeches:
			date = re.findall(date_regex, str(identity))[0]
			if (date >= "1792-09-20") and (speaker_name == speechid_to_speaker[identity]):
				speech = speech + " " + raw_speeches[identity]
		speaker_ngrams = compute_ngrams(speech)
		pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
		with open(pickle_filename, 'wb') as handle:
			pickle.dump(speaker_ngrams, handle, protocol = 0)

	"""for speechid in speechid_to_speaker:
		date = re.findall(date_regex, str(speechid))[0]
		if date >= '1792-09-20':
			speaker_name = speechid_to_speaker[speechid]
			if (speaker_name in speakers_to_consider) and (speaker_name not in speaker_names):
				speaker_names.add(speaker_name)
				speech = ""
				for identity in raw_speeches:
					if speaker_name == speechid_to_speaker[identity]:
						speech = speech + " " + raw_speeches[identity]
				speaker_ngrams = compute_ngrams(speech)
				pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
				with open(pickle_filename, 'wb') as handle:
					pickle.dump(speaker_ngrams, handle, protocol = 0)"""

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
    speakers_to_analyze = load_list("Copy of Girondins and Montagnards.xlsx")
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate_by_speaker(speakers_to_analyze, raw_speeches, speechid_to_speaker)
