import networkx as nx
import math
import scipy.stats as st
import re
import networkx as nx
import matplotlib
import numpy as np
import spacy
import itertools as it
import os
from collections import defaultdict
import random
import copy
import sys
from utils import *
import pickle
import math

MODEL = 'es_core_news_sm'
# MODEL = 'en_core_web_md'
# MODEL = 'es_core_news_lg'

# we centralize the extraction
nlp = spacy.load(MODEL)
ruler = nlp.add_pipe("entity_ruler")
patterns = [ {"label": "DNI", "pattern": [{"TEXT": {"REGEX": ".*?[0-9]{8}[A-Z]"}}]},
             {"label": "NUM", "pattern": [{"IS_DIGIT":True}]} ]
ruler.add_patterns(patterns)

def read_corpus(file):
    corpus = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            corpus.append(line)
    return corpus

def is_entity(word):
    return word in ["<ORG>", "<LOC>", "<PER>", "<MISC>", "<GPE>", "<MONEY>", "<DNI>", "<ENTITY>", "<NUM>"]

def find_entity_matches(line):
    return re.finditer('[GPLODN][PEORNU][ERCGIM]_<.*?>|MISC_<.*?>|MONEY_<.*>', line)

#helper funtion to check entities
# def check_entities(sentence):
#     possible_entities = ["CHEMICAL_", "GENE_", "DISEASE_"]
#     sentence_words = sentence.split(" ")
#     total_present = 0
#     entities_present = []
#     for ent in possible_entities:
#         result = [word for word in sentence_words if ent in word]
#         if len(result) > 0:
#             result = set(result)
#             total_present += len(result)
#             entities_present.extend(list(result))
#     return total_present, entities_present
#

def shortest_dependency_path(doc, e1=None, e2=None):
    ## CB: This should also work for NERs, if the dep parsing is pointing at the
    ## head of the entity,
    ## However, the NE would not be added completely
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.i),
                          '{0}'.format(child.i)))
            edges.append(('{0}'.format(child.i),
                          '{0}'.format(token.i)))

    graph = nx.Graph(edges)
    shortest_path = []
    try:
        # print (f'looking shortest path from {e1.i} to {e2.i}')
        shortest_path = nx.shortest_path(graph, source=str(e1.i), target=str(e2.i))
    except nx.NetworkXNoPath:
        print (f'problems with shortest_path - {str(e1.i)} {doc[e1.i]} - {str(e2.i)} {doc[e2.i]}')
        print (graph)
        shortest_path=[]
    return shortest_path

## TODO: to be revisited as they work at string level, and this should be done keeping the
## token order if we want not to lose the original POS tagging
def adv_mod_deps(x, dep_parse):
    for token in dep_parse:
        if token.dep_ == "advmod":
            for word in x:
                if str(token) == word:
                    x.insert(x.index(str(token)), str(token.head.text))
                    break
                if str(token.head.text) == word:
                    x.insert(x.index(str(token.head.text)), str(token))
                    break
    return x

def obtain_children(idx, doc):
    to_process = [c for c in doc[idx].children]
    children=[]
    while len(to_process) != 0:
        c = to_process.pop()
        children.append(c.i)
        for child in c.children:
            if child.offset not in children:
                to_process.append(child)

    return sorted(children)
def add_children_deps(path, dep_parse):
    children_start = obtain_children(int(path[0]), dep_parse)
    children_end = obtain_children(int(path[-1]), dep_parse)
    test = [int(x) for x in (path + children_start + children_end)]
    if int(path[0]) < int(path[-1]):
        return sorted(test)
    else:
        return sorted(test,reverse=True)

def detype(pat):
    words = pat.split(" ")
    strret = list()
    for w in words:
        if is_entity(w):
            strret.append("<ENTITY>")
        else:
            strret.append(w)
    strret = ' '.join(strret)
    print (f'vs {pat}')
    print (f'vs {strret}')
    return strret

def get_strength_confidence(p_s_c, utc):
    strength = dict()
    confidence = dict()
    for pat in utc:
        strength[pat] = len(utc[pat])
    for pat in p_s_c:
        strength[pat] = len(p_s_c[pat])
    for pat in p_s_c:
        confidence[pat] = strength[pat] / strength[(detype(pat))]
    return strength, confidence

def convert_patterns_list(p_s_c):
    p_l_s_c = dict()
    for it in p_s_c:
        p_l_s_c[it] = sorted(p_s_c[it].items(), key=lambda x: x[1], reverse = True)
    l_p_l_s_c = list(p_l_s_c.items())
    return  l_p_l_s_c

# calculate_wilson_score(['a', 'b', 'c'], ['c', 'd', 'a', 'b'], 0.05)

def calculate_wilson_score(s, b, confidence=0.05):
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    pos = len(list(set(s) & set(b)))
    n = len(s)
    ph = 1.0 * pos / n
    sqr = math.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n)
    numer = (ph + z * z / (2 * n))# - z * sqr) #uncomment for lower bound on wilson score
    denom = (1 + z * z / n)
    return numer/denom

#calculate_wilson_score([ 'a','b'], ['a','b','c','d','e','f', 'g','h','i','j'], 0.95)

# strength_pat, conf_pat = get_strength_confidence(p_s_c, utc)
