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
import xlsxwriter



#daily_regex = '(?:<p>[ ]{0,1}Séance)(?:[\s\S]+)(?:<p>[ ]{0,1}Séance)'
#Seance followed by less than or equal to 4 line breaks (\n) then date value =
daily_regex = '(?:Séance[\s\S]{0,200}<date value=\")(?:[\s\S]+)(?:Séance[\s\S]{0,200}<date value=\")'

#raw_speeches = {}
speechid_to_speaker = {}
names_not_caught = set()
speeches_per_day = {}
speakers_using_find = set()
global speaker_list



def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def parseFiles(raw_speeches):
    files = os.listdir("Docs/")
    dates = set()
    for filename in files:
        if filename.endswith(".xml"):
        	print(filename)
        	filename = open('Docs/' + filename, "r")
        	contents = filename.read()
        	soup = BeautifulSoup(contents, 'lxml')
        	sessions = soup.find_all(['div2', 'div3'], {"type": ["session", "other"]})
        	#sessions.append(soup.find_all('div2', {"type": "other"}))
        	for session in sessions:
		        date = extractDate(session)
		        if (date >= "1789-05-05") and (date <= "1795-01-04") and (date != "error"):
		        	if date in dates:
		        		date = date + "_soir"
		        		if date in dates:
		        			date = date + "2"
		        			findSpeeches(raw_speeches, session, date)
		        		else:
		        			findSpeeches(raw_speeches, session, date)
		        			dates.add(date)		        		
		        	else:
		        		findSpeeches(raw_speeches, session, date)
		        		dates.add(date)
	        filename.close()

def findSpeeches(raw_speeches, daily_soup, date):
	id_base = date.replace("/","_")
	number_of_speeches = 0
	for talk in daily_soup.find_all('sp'):
		try:
			speaker = talk.find('speaker').get_text()
			speaker = remove_diacritic(speaker).decode('utf-8')
			speaker = speaker.replace(".","").replace(":","").replace("MM ", "").replace("MM. ","").replace("M ", "").replace("de ","").replace("M. ","").replace("M, ","").replace("M- ","").replace("M; ","").replace("M* ","")
			if speaker.endswith(","):
				speaker = speaker[:-1]
			if speaker.endswith(", "):
				speaker = speaker[:-1]
			if speaker.startswith(' M. '):
				speaker = speaker[3:]
			if speaker.startswith(' '):
				speaker = speaker[1:]
			if speaker.endswith(' '):
				speaker = speaker[:-1]
		except AttributeError:
			speaker = ""
		speech = talk.find_all('p')
		text = ""
		full_speech = ""
		for section in speech:
			text = text + section.get_text()
		full_speech = remove_diacritic(text).decode('utf-8')
		full_speech = full_speech.replace("\n"," ").replace("--"," ").replace("!"," ")
		full_speech = re.sub(r'([ ]{2,})', ' ', full_speech)
		full_speech = re.sub(r'([0-9]{1,4})', ' ', full_speech)
		speaker_name = ""
		#speech_of_day = speech_of_day + full_speech
		if speaker != "Le President":
			if speaker in speaker_list.index.values:
				#number_of_speeches = number_of_speeches + 1
				#speeches_of_day = speeches_of_day + " " + full_speech
				for j, name in enumerate(speaker_list.index.values):
					if speaker == name:
						speaker_name = speaker_list["FullName"].iloc[j]
				#speaker_name = speaker_list.loc[speaker, "FullName"]
			else:
				for i, name in enumerate(speaker_list['LastName']):
					if (speaker.find(",") == -1) and (speaker.find(" et ") == -1):
						if speaker.find(name) != -1 :
							speaker_name = speaker_list["FullName"].iloc[i]
							speakers_using_find.add(speaker + " : " + remove_diacritic(speaker_name).decode('utf-8') + "\n")
		if speaker_name is not "":
			speaker_name = remove_diacritic(speaker_name).decode('utf-8')
			number_of_speeches = number_of_speeches + 1
			speech_id = "" + id_base + "_" + str(number_of_speeches)
			speechid_to_speaker[speech_id] = speaker_name
			raw_speeches[speech_id] = full_speech
		else:
			names_not_caught.add(speaker + "\n")

	speeches_per_day[id_base] = number_of_speeches




def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'AP Speaker Authority List xlsx')
	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list


# Parses dates from file being analyzed
def extractDate(soup_file):
	dates = soup_file.find_all('date')
	relevant_dates = []
	for date in dates:
		if date.attrs:
			relevant_dates.append(date)
	if (len(relevant_dates) > 0):
		return(relevant_dates[0]['value'])
	else:
		return("error")



if __name__ == '__main__':
    import sys
    speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_2.xlsx')
    raw_speeches = {}
    parseFiles(raw_speeches)
    txtfile = open("names_not_caught.txt", 'w')
    for name in sorted(names_not_caught):
    	txtfile.write(name)
    txtfile.close()
    file = open('speakers_using_find.txt', 'w')
    for item in sorted(speakers_using_find):
    	file.write(item)
    file.close()
    pickle_filename = "speechid_to_speaker.pickle"
    with open(pickle_filename, 'wb') as handle:
    	pickle.dump(speechid_to_speaker, handle, protocol = 0)
    pickle_filename_2 = "raw_speeches.pickle"
    with open(pickle_filename_2, 'wb') as handle:
    	pickle.dump(raw_speeches, handle, protocol = 0)


    
       
   	
