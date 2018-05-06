
# Function to iterate through each document to create each day


# Things to keep track of
### Date of speech
### Speaker name
### Dictionary of speech id to speaker


# Function to iterate through each day to get each speech
### Check if the speaker is president, if so skip
### Check if speaker in dictionary of speaker names. If so, create unique id and store unique id, speaker pairing


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

daily_regex = '(?:<p>[ ]{0,1}Séance)[\s\S]+(?:<p>[ ]{0,1}Séance)'
speechid_to_speaker = {}
speeches_per_day = {}
stopwords = load_stopwords('Docs/FrenchStopwords.txt')
#speaker_list = load_speakerlist()


def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def parseFiles():
    files = os.listdir("Docs/")
    for filename in files:
        if filename.endswith(".xml"):
            filename = open('Docs/' + filename, "r")
            contents = filename.read()
            daily = re.findall(daily_regex, contents, overlapped=True)
		    for day in daily:
		        soup = BeautifulSoup(day, 'lxml')
		        date = extractDate(soup)
		        findSpeeches(soup, date)
            filename.close()


def findSpeeches(daily_soup, date):
	id_base = date.replace("_","")
	number_of_speeches = 0
	for talk in soup.find_all('sp'):
        # Key is speaker name
        try:
            speaker = talk.find('speaker').get_text()
            speaker = remove_diacritic(speaker).decode('utf-8')
            if speaker.endswith('.'):
                speaker = speaker[:-1]
        except AttributeError:
            pass
        
        speech = talk.find_all('p')
        text = ""
        for section in speech:
            text = text + section.get_text()
        full_speech = remove_diacritic(text).decode('utf-8')
        full_speech = full_speech.replace("\n"," ")
        full_speech = re.sub(r'([ ]{2,})', ' ', full_speech)

        # Check if speaker name in master list or is the president
        """if speaker != "Le President":
        	number_of_speeches += 1
        	if speaker in speaker_list:"""
       	### MOVE ALL THIS CODE WITHIN THE IF STATEMENT ONCE HAVE SPEAKER LIST
		speaker_name = speaker_list[speaker]

	    speech_id = number_of_speeches + "_" + number_of_speeches
	    speechid_to_speaker[speech_id] = speaker_name

	    # Store raw speech
	    pickle_filename = "" + uniqueid + ".pickle"
	    # Serializes the dictionary to a pickle file to sanity check. 
	    with open(pickle_filename, 'wb') as handle:
	        pickle.dump(full_speech, handle, protocol = 0)

	    compute_ngrams(speech_id, full_speech)

    speeches_per_day[id_base] = number_of_speeches


def compute_trigrams(input):
    token = word_tokenize(input)
    trigrams = ngrams(token, 3)
    return trigrams


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


#def load_speakerlist(speakernames)
 	


def remove_stopwords(input):
    filtered_text = ""
        for word in input.split():
        if word not in stopwords:
            filtered_text = filtered_text + " " + word
    return filtered_text


def compute_ngrams(uniqueid, speech):
	speech = speech.replace("'"," ").replace(";"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
	clean_text = remove_stopwords(speech.lower())
	trigrams = compute_trigrams(clean_text)
    speech_ngrams = Counter(new_bigrams)
    pickle_filename = "" + uniqueid + "_ngrams" + ".pickle"
    # Serializes the dictionary to a pickle file to sanity check. 
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(speech_ngrams, handle, protocol = 0)
   


# Parses dates from file being analyzed
def extractDate(soup_file):
    dates = soup_file.find_all('date')
    relevant_dates = []
    for date in dates:
        if date.attrs:
            relevant_dates.append(date)
    return(relevant_dates[0]['value'])



if __name__ == '__main__':
    import sys
    parseFiles()


    
       
   	
