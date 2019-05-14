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
import xlsxwriter
from processing_functions import remove_diacritic, load_speakerlist, cosine_similarity
from make_ngrams import make_ngrams
from itertools import islice, izip
from Levenshtein import distance

def read_names(name_file):
	# pd_list = pd.read_excel("an_names.xls")
	pd_list = pd.read_excel(name_file)
	pd_list = pd_list.set_index('Last Name')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8').lower()
	pd_list.index = speakers
	full_names = []
	for full_name in pd_list["Full Name"]:
		full_names.append(remove_diacritic(full_name).decode('utf-8').lower())
	pd_list["Full Name"] = full_names
	return pd_list

def speaker_name_split(full_speaker_names):
	speakers_split = []
	for speaker_name in full_speaker_names.index:
		words = re.findall("\w+", speaker_name)
		split = Counter(izip(words, islice(words, 1, None)))
		speakers_split.append(split)
	return speakers_split


# Need to remove diacritic

def compute_speaker_Levenshtein_distance(speaker_name):
	full_speaker_names = read_names("APnames.xlsx")
	# speaker_last_names = read_names("an_last_names.xls")

	distance_size = {}
	for i, speaker in enumerate(full_speaker_names['Full Name']):
		# Levenshtein distance
		# speaker = unicodedata.normalize("NFKD", speaker).encode("ascii", "ignore")
		dist = distance(speaker, speaker_name)
		distance_size[speaker] = dist

	for j, speaker in enumerate(full_speaker_names.index.values):
		# Levenshtein distance
		# speaker = unicodedata.normalize("NFKD", speaker).encode("ascii", "ignore")
		dist = distance(speaker, speaker_name)
		full_name = full_speaker_names["Full Name"].iloc[j]
		if full_name in distance_size:
			if dist < distance_size[full_name]:
				distance_size[full_name] = dist
		else:
			distance_size[full_name] = dist
	dist_size_sorted = sorted(distance_size.items(), key = lambda kv: kv[1])

	return dist_size_sorted[:2]


if __name__ == '__main__':
	import sys
	full_speaker_names = read_names()
	speakers_split = speaker_name_split(full_speaker_names)
	compute_speaker_Levenshtein_distance(full_speaker_names)

