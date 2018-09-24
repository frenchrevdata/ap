#!/usr/bin/env python

import pickle
import pandas as pd
from pandas import *
import collections
from collections import Counter
import regex as re
from make_ngrams import compute_ngrams
import math
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, write_to_excel, convert_keys_to_string, compute_difference, cosine_similarity
from scipy import spatial


def firststep():

	full_date = {}
	byfulldate = {}

	ngrams = {}

	raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))

	for speechid in raw_speeches:
		fulldate = speechid[0:10]
		if (fulldate >= "1792-06-10") and (fulldate <= "1793-08-02"):
			speech_bigrams = compute_ngrams(raw_speeches[speechid], 2)

			print fulldate
			if fulldate in byfulldate:
				byfulldate[fulldate] = byfulldate[fulldate] + speech_bigrams
			else:
				byfulldate[fulldate] = speech_bigrams


			#year_month[speechid] = yearmonth
			#byyearmonth[speechid] = speech_bigrams
	print "here"

	with open("byfulldate.pickle", "wb") as handle:
		pickle.dump(byfulldate, handle, protocol = 0)

	byfulldate = pd.DataFrame.from_dict(byfulldate, orient = "index")

	write_to_excel(byfulldate, "byfulldate.xlsx")


if __name__ == '__main__':
	import sys
	firststep()