# extract all i tags and sc tags and note tags
# look only at p tags

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
from processing_functions import remove_diacritic, load_speakerlist
from make_ngrams import make_ngrams
from parse_speaker_names import compute_speaker_Levenshtein_distance


vol_regex = 'AP_ARTFL_vols\/AP(vol[0-9]{1,2}).xml'


def parseEnc():
	# Assumes all xml files are stored in a Docs folder in the same directory as the python file
    files = os.listdir("Encyclopedie/")
    words = set()
    for filename in files:
        if filename.endswith(".tei"):
        	print(filename)
        	filename = open('Encyclopedie/' + filename, "r")
        	contents = filename.read()
        	soup = BeautifulSoup(contents, 'lxml')

        	paragraphs = soup.find_all('p')
        	for para in paragraphs:
        		if para.find("i"):
        			para.i.extract()
        		if para.find("sc"):
        			para.sc.extract()
        		if para.find("note"):
        			para.note.extract()
        		para = para.get_text()
        		para = para.replace("\n"," ").replace("& ","").replace("; ","").replace(".","").replace(",","").replace("?","").replace("!","").replace("  "," ")
        		paragraph = remove_diacritic(para).decode('utf-8')
        		para = para.lower()
        		paragraph = paragraph.split(" ")
        		words = words.union(paragraph)
    return words

def checkErrors(enc_words):
	files = os.listdir("AP_ARTFL_vols/")
	errors_per_vol = {}
	for filename in files:
	    if filename.endswith(".xml"):
	    	filename = open('AP_ARTFL_vols/' + filename, "r")
	    	volno = re.findall(vol_regex, str(filename))[0]
	    	contents = filename.read()
	    	soup = BeautifulSoup(contents, 'lxml')

	    	num_errors = 0

	    	paragraphs = soup.find_all('p')
	    	for para in paragraphs:
	    		if para.find("note"):
	    			para.note.extract()
	    		para = para.get_text()
	    		para = para.replace("\n"," ").replace(")", "").replace("*","").replace(":","").replace("-","").replace("_","").replace("(","").replace("& ","").replace("; ","").replace(".","").replace(",","").replace("?","").replace("!","").replace("  "," ")
	    		para = re.sub(r'([0-9]{1,4})', ' ', para)
	    		paragraph = remove_diacritic(para).decode('utf-8')
	    		para = para.lower()
	    		words = paragraph.split(" ")
	    		for word in words:
	    			if word not in enc_words:
	    				num_errors += 1
	   	errors_per_vol[volno] = num_errors
	with open("errors_per_vol.pickle", 'wb') as handle:
		pickle.dump(errors_per_vol, handle, protocol = 0)
	w = csv.writer(open("errors_per_vol.csv", "w"))
	for key, val in errors_per_vol.items():
		w.writerow([key,val])


if __name__ == '__main__':
	import sys
	# words = parseEnc()
	# pickle_filename = "enc_words.pickle"
	# with open(pickle_filename, 'wb') as handle:
	# 	pickle.dump(words, handle, protocol = 0)
	enc_words = pickle.load(open("enc_words.pickle", "rb"))
	checkErrors(enc_words)
	# errors_per_vol = pickle.load(open("errors_per_vol.pickle", "rb"))
	# w = csv.writer(open("errors_per_vol.csv", "w"))
	# for key, val in errors_per_vol.items():
	# 	w.writerow([key,val])

