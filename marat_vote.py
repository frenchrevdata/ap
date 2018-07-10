#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
import csv
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import gzip
from make_ngrams import compute_ngrams
import xlsxwriter
import string
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora


#global speaker_list



def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def parseFile(votes):
	justifications = []
	file = open('marat.xml', "r")
	contents = file.read()
	contents = re.sub(r'(<p>(?:DÉPARTEMENT|DEPARTEMENT|DÉPARTEMENE)[\s\S]{1,35}<\/p>)', '', contents)
	soup = BeautifulSoup(contents, 'lxml')
	for talk in soup.find_all('sp'):
		speaker = talk.find('speaker').get_text()
		speaker = remove_diacritic(speaker).decode('utf-8')
		speaker = speaker.replace(".","")

		speech = talk.find_all('p')
		text = ""
		full_speech = ""
		for section in speech:
			text = text + section.get_text()
		full_speech = remove_diacritic(text).decode('utf-8')
		full_speech = full_speech.replace('\n', '').replace('\t', '').replace('\r','')
		full_speech = re.sub(r'([ ]{2,})', ' ', full_speech)

		if len(full_speech) > 30:
			justifications.append(full_speech)

		votes[speaker] = full_speech

	runTopicModel(justifications)

	df = pd.DataFrame.from_dict(votes, orient = 'index')
	writer = pd.ExcelWriter('Marat_Justifications.xlsx')
	df.to_excel(writer)
	writer.save()
	file.close()

def remove_stopwords(input, stopwords):
	filtered_text = ""
	for word in input.split():
		if word not in stopwords:
			filtered_text = filtered_text + " " + word
	return filtered_text

def clean(just_speech):
	stopwords_from_file = open('FrenchStopwords.txt', 'r')
	lines = stopwords_from_file.readlines()
	french_stopwords = []
	for line in lines:
		word = line.split(',')
		#remove returns and new lines at the end of stop words so the parser catches matches
		#also remove accents so the entire analysis is done without accents
		word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
		french_stopwords.append(word_to_append)

	just_speech = just_speech.replace("]"," ").replace("[", " ").replace("&"," ").replace(">"," ").replace("#"," ").replace("/"," ").replace("\`"," ").replace("'"," ").replace("*", " ").replace("`", " ").replace(";"," ").replace("?"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
	clean_text = remove_stopwords(just_speech.lower(), french_stopwords)
	clean_text = clean_text.replace("marat", " ").replace("accusation"," ")
	#exclude = set(string.punctuation)
	#no_punc = ''.join(ch for ch in clean_text if ch not in exclude)
	return clean_text

def runTopicModel(justifications):
	clean_speeches = [clean(justification).split() for justification in justifications]
	dictionary = corpora.Dictionary(clean_speeches)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_speeches]
	Lda = gensim.models.ldamodel.LdaModel
	number_of_topics = 5
	number_of_words = 3
	ldamodel = Lda(doc_term_matrix, num_topics = number_of_topics, id2word = dictionary, passes = 50)
	results = ldamodel.print_topics(num_topics = number_of_topics, num_words = number_of_words)
	filename = "topic_modeling_" + str(number_of_topics) + "_numtopics_" + str(number_of_words) + "_numwords.txt"
	file = open(filename, 'w')
	for topic in results:
		file.write(topic[1] + "\n")
	file.close()
	print(results)

"""def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'AP Speaker Authority List xlsx')
	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list"""


if __name__ == '__main__':
    import sys
    #speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_2.xlsx')
    votes = {}
    parseFile(votes)
    """df = pd.DataFrame.from_dict(justifications)
    writer = pd.ExcelWriter('Marat_Justifications.xlsx')
    df.to_excel(writer)
    writer.save()"""