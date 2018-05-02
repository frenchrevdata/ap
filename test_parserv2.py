#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
import csv

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


# Parses dates from file being analyzed
def extractDates(soup_file):
    dates = soup_file.find_all('date')
    relevant_dates = []
    for date in dates:
        if date.attrs:
            relevant_dates.append(date)
    return([relevant_dates[0]['value'], relevant_dates[1]['value']])


# Code for parsing all files but getting memory errors
def parseFiles():
    #file = open(sys.argv[1], "r")
    files = os.listdir("Docs/")
    files = sorted(files)
    for filename in files:
        if filename.endswith(".xml"):
            print(filename)
            filename = open('Docs/' + filename, "r")
            contents = filename.read()
            soup = BeautifulSoup(contents, 'lxml')
            doc_dates = extractDates(soup)

            speaker_text_dict = {}

            for talk in soup.find_all('sp'):
                # Key is speaker name
                try:
                    key = talk.find('speaker').get_text()
                    key = remove_diacritic(key).decode('utf-8')
                except AttributeError:
                    print("No speaker found")
                # Value is speaker speech
                value = talk.find('p').get_text()
                value = remove_diacritic(value).decode('utf-8')

                speech = talk.find_all('p')
                text = ""
                for section in speech:
                    text = text + section.get_text()
                value = remove_diacritic(text).decode('utf-8')

                if key and value:
                    # Check if the key already exists
                    if speaker_text_dict.has_key(key):
                        current_value = speaker_text_dict[key]
                        new_value = current_value + " " + value
                        speaker_text_dict[key] = new_value
                    else:
                        speaker_text_dict[key] = value

            csv_filename = "" + doc_dates[0] + " a " + doc_dates[1] + ".csv"
            # Writes the dictionary to a csv file to sanity check. Still some parsing errors
            with open(csv_filename, 'wb') as csvfile:
                writer = csv.writer(csvfile)
                for key, value in speaker_text_dict.items():
                    writer.writerow([key, value])

            filename.close()


            
    #print(speaker_text_dict.keys())


if __name__ == '__main__':
    import sys
    
    parseFiles()
