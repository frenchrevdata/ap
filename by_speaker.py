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


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    aggregate_by_speaker()
    ngrams_by_speaker()
    pickle_filename = "by_speaker.pickle"
    with open(pickle_filename, 'wb') as handle:
    	pickle.dump(ngrams_per_speaker, handle, protocol = 0)