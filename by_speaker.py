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
speeches_per_speaker = {}
ngrams_per_speaker = {}
global stopwords

def aggregate_by_speaker():
	for speechid in raw_speeches:
		speaker_name = speechid_to_speaker[speechid]
		speech = raw_speeches[speechid]
		if speaker_name in speeches_per_speaker:
			speeches_per_speaker[speaker_name] = speeches_per_speaker[speaker_name] + "" + speech
		else:
			speeches_per_speaker[speaker_name] = speech

def ngrams_by_speaker():
	for speaker in speeches_per_speaker:
		text = speeches_per_speaker[speaker]
		ngrams_per_speaker[speaker] = compute_ngrams(text)

"""def make_ngrams(input, amount):
	token = word_tokenize(input)
	n_grams = ngrams(token, amount)
	return n_grams


def load_stopwords(textfile):
	stopwords_from_file = open(textfile, 'r')
	lines = stopwords_from_file.readlines()
	french_stopwords = []
	for line in lines:
		word = line.split(',')
		#remove returns and new lines at the end of stop words so the parser catches matches
		#also remove accents so the entire analysis is done without accents
		word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
		french_stopwords.append(word_to_append)
	return(french_stopwords)


def remove_stopwords(input):
	filtered_text = ""
	for word in input.split():
		if word not in stopwords:
			filtered_text = filtered_text + " " + word
	return filtered_text


def compute_ngrams(speech):
	speech = speech.replace("'"," ").replace("*", " ").replace("`", " ").replace(";"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
	clean_text = remove_stopwords(speech.lower())
	clean_text = clean_text.replace("mm secretaire", " ").replace("assemble nationale", " ").replace("monsieur president", " ").replace("convention nationale", " ").replace("archives parliamentaire", " ").replace("republique francaise", " ").replace("ordre jour", " ").replace("corps legislatif", " ")
	n_grams = make_ngrams(clean_text, 2)
	speech_ngrams = Counter(n_grams)
	return(speech_ngrams)"""


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    aggregate_by_speaker()
    ngrams_by_speaker()
    pickle_filename = "by_speaker.pickle"
    with open(pickle_filename, 'wb') as handle:
    	pickle.dump(ngrams_per_speaker, handle, protocol = 0)
