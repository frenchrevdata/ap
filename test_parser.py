#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def parseFile(file):
    file = open(sys.argv[1], "r")
    contents = file.read()
    soup = BeautifulSoup(contents, 'lxml')
    

    speaker_text_dict = {}

    for talk in soup.find_all('sp'):
        # Key is speaker name
        key = talk.find('speaker').get_text()
        key = remove_diacritic(key).decode('utf-8')
        # Value is speaker speech
        value = talk.find('p').get_text()
        value = remove_diacritic(value).decode('utf-8')

        speaker_text_dict[key] = value

    #print(speaker_text_dict)
    print(speaker_text_dict.keys())



if __name__ == '__main__':
    import sys
    file_name = sys.argv[1]
    parseFile(file_name)
