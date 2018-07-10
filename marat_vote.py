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


#global speaker_list



def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def parseFile(justifications):
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

		justifications[speaker_name] = full_speech

	df = pd.DataFrame.from_dict(justifications, orient = 'index')
    writer = pd.ExcelWriter('Marat_Justifications.xlsx')
    df.to_excel(writer)
    writer.save()
    file.close()


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
    justifications = {}
    parseFile(justifications)
    """df = pd.DataFrame.from_dict(justifications)
    writer = pd.ExcelWriter('Marat_Justifications.xlsx')
    df.to_excel(writer)
    writer.save()"""