# -*- coding: utf-8 -*-
"""
Created on Tues Oct 16 23:33:04 2018

@author: Ken Huang
"""
# import nltk
import codecs
import os
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from afinn import Afinn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def flatten(input_list):
    flat_list = []
    for i in input_list:
        if type(i) == list: flat_list += flatten(i)
        else: flat_list += [i]

    return flat_list


def common_words(path):
    with codecs.open(path) as f:
        words = f.read()
        words = json.loads(words)

    return set(words)


def read_novel(book_name, path):
    book_list = os.listdir(path)
    book_list = [i for i in book_list if i.find(book_name) >= 0]
    novel = ''
    for i in book_list:
        with codecs.open(path / i, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read().replace('\r', ' ').replace('\n', ' ').replace("\'", "'")
        novel += ' ' + data

    return novel


def name_entity_recognition(sentence):
    doc = nlp(sentence)
    name_entity = [x for x in doc.ents if x.label_ in ['PERSON', 'ORG']]
    name_entity = [str(x).lower().replace("'s","") for x in name_entity]
    name_entity = [str(x).lower().replace(".","") for x in name_entity]
    name_entity = [x.split(' ') for x in name_entity]
    name_entity = flatten(name_entity)
    name_entity = [x for x in name_entity if len(x) >= 2]
    name_entity = [x for x in name_entity if x not in words]
    
    return name_entity


def iterative_NER(sentence_list, threshold_rate=0.0005):
    output = []
    for i in sentence_list:
        name_list = name_entity_recognition(i)
        if name_list != []:
            output.append(name_list)
    output = flatten(output)
    from collections import Counter
    output = Counter(output)
    output = [x for x in output if output[x] >= threshold_rate * len(sentence_list)]

    return output


def top_names(name_list, novel, top_num=20):
    vect = CountVectorizer(vocabulary=name_list, stop_words='english')
    name_frequency = vect.fit_transform([novel.lower()])
    name_frequency = pd.DataFrame(name_frequency.toarray(), columns=vect.get_feature_names_out())
    name_frequency = name_frequency.T
    name_frequency = name_frequency.sort_values(by=0, ascending=False)
    name_frequency = name_frequency[0:top_num]
    names = list(name_frequency.index)
    name_frequency = list(name_frequency[0])

    return name_frequency, names


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    words = common_words('common_words.txt') # removed I, me, hook
    novel_name = 'Peter Pan'
    novel_folder = Path(os.getcwd()) / 'novels'
    # nltk.download('punkt')
    novel = read_novel(novel_name, novel_folder)
    sentence_list = sent_tokenize(novel)
    preliminary_name_list = iterative_NER(sentence_list)
    name_frequency, name_list = top_names(preliminary_name_list, novel, 25)
    print(name_list)