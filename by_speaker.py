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


def aggregate_by_speaker(raw_speeches, speechid_to_speaker):
	speaker_names = set()
	for speechid in speechid_to_speaker:
		speaker_name = speechid_to_speaker[speechid]
		if (speaker_name in speakers_to_analyze.index.values) and (speaker_name not in speaker_names):
			speaker_names.add(speaker_name)
			speech = ""
			for identity in raw_speeches:
				if speaker_name == speechid_to_speaker[identity]:
					speech = speech + " " + raw_speeches[identity]
			speaker_ngrams = compute_ngrams(speech)
			pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
			with open(pickle_filename, 'wb') as handle:
				pickle.dump(speaker_ngrams, handle, protocol = 0)



if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate_by_speaker(raw_speeches, speechid_to_speaker)
