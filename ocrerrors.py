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
page_regex = '<pb n="[\s0-9]+" facs="[\s\S]{0,300}" \/> [\s\S]{0,10000} <pb'


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

def remove_stopwords(input, stopwords):
	filtered_text = ""
	for word in input.split():
		if word not in stopwords:
			filtered_text = filtered_text + " " + word
	return filtered_text

def checkErrors(enc_words, french_stopwords):
	files = os.listdir("AP_ARTFL_vols/")
	errors_per_vol = {}
	errors_per_page = {}

	for filename in files:
	    if filename.endswith(".xml"):
	    	filename = open('AP_ARTFL_vols/' + filename, "r")
	    	volno = re.findall(vol_regex, str(filename))[0]
	    	contents = filename.read()
	    	soup = BeautifulSoup(contents, 'lxml')

	    	num_errors = 0

	    	pages = re.findall(r'<pb n="[\s0-9]+" facs="[\s\S]{0,300}" \/> [\s\S]{0,10000} <pb', contents)
	    	for page in pages:
	    		page_num = BeautifulSoup(page, 'lxml').find_all('pb')
	    		pageno = volno + "_pg" + page_num[0].get("n")
	    		error_per_page = 0
		    	paragraphs = soup.find_all('p')
		    	for para in paragraphs:
		    		if para.find("note"):
		    			para.note.extract()
		    		para = para.get_text().lower()
		    		para = remove_diacritic(para).decode('utf-8')
		    		para = para.replace("'", " ")
	    			paragraph = remove_stopwords(para, french_stopwords)
		    		# para = para.replace("s'","").replace("l'","").replace("d'","")
		    		paragraph = paragraph.replace("\n"," ").replace(")", "").replace("*","").replace(":","").replace("-","").replace("_","").replace("(","").replace("& ","").replace("; ","").replace(".","").replace(",","").replace("?","").replace("!","")
		    		paragraph = re.sub(r'([0-9]{1,4})', ' ', paragraph)
		    		words = paragraph.split(" ")
		    		for word in words:
		    			if word not in enc_words:
		    				print word
		    				error_per_page += 1
		    				num_errors += 1
		    	errors_per_page[pageno] = error_per_page

	   	errors_per_vol[volno] = num_errors
	with open("errors_per_vol.pickle", 'wb') as handle:
		pickle.dump(errors_per_vol, handle, protocol = 0)
	w = csv.writer(open("errors_per_vol.csv", "w"))
	for key, val in errors_per_vol.items():
		w.writerow([key,val])

	with open("errors_per_page.pickle", 'wb') as handle:
		pickle.dump(errors_per_page, handle, protocol = 0)
	w = csv.writer(open("errors_per_page.csv", "w"))
	for key, val in errors_per_page.items():
		w.writerow([key,val])


if __name__ == '__main__':
	import sys
	# words = parseEnc()
	# pickle_filename = "enc_words.pickle"
	# with open(pickle_filename, 'wb') as handle:
	# 	pickle.dump(words, handle, protocol = 0)
	enc_words = pickle.load(open("enc_words.pickle", "rb"))
	stopwords_from_file = open('FrenchStopwords.txt', 'r')
	lines = stopwords_from_file.readlines()
	french_stopwords = []
	for line in lines:
		word = line.split(',')
		#remove returns and new lines at the end of stop words so the parser catches matches
		#also remove accents so the entire analysis is done without accents
		word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
		french_stopwords.append(word_to_append)
	checkErrors(enc_words, french_stopwords)
	# errors_per_vol = pickle.load(open("errors_per_vol.pickle", "rb"))
	# w = csv.writer(open("errors_per_vol.csv", "w"))
	# for key, val in errors_per_vol.items():
	# 	w.writerow([key,val])

