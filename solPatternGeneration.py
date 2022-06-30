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
import utils

import pickle
import math
import scipy.stats as st

nlp = spacy.load(utils.MODEL)

SUPPORT_THRESHOLD = 5

def is_entity(word):
    return "<ORG>" == word or "<LOC>" == word or "<PER>" == word or "<MISC>" == word

#replacing non entity non frequent n gram by wildcard
def generate_sol_and_sol_pos_patterns(patterns, ngrams, post):
    """A method to generate SOL patterns given the textual patterns and ngrams, along with the sol_pos pattern as well.

    Parameters
    ----------
    patterns : type List
        Textual Patterns
    ngrams : type List of tuples
        NGrams
        post : type List
        POS Tag patterns

    Returns
    -------
    type List
        Returns SOL Patterns
    type List
        List of patterns with POS Tags

    """
    pos_patterns = []
    sol_pos_patterns = []
    for pattern_index, pattern in enumerate(patterns):
        splitted_pattern = []
        matches = re.finditer('[PLO][EOR][RCG][C]?_<.*?>|MISC_<.*?>', pattern)
        prev_match = next(matches)
        line = pattern [0:prev_match.start()]
        splitted_pattern+=pattern [0:prev_match.start()].split()
        line += "<"+pattern[prev_match.start():prev_match.end()].split('_<')[0]+">"
        splitted_pattern.append("<" + pattern[prev_match.start():prev_match.end()].split('_<')[0] + ">")
        for current_match in matches:
            line+=pattern[prev_match.end():current_match.start()]
            splitted_pattern+=pattern[prev_match.end():current_match.start()].split()
            line+="<"+pattern[current_match.start():current_match.end()].split('_<')[0]+">"
            splitted_pattern.append("<"+pattern[current_match.start():current_match.end()].split('_<')[0]+">")
            prev_match = current_match
        line+=pattern[prev_match.end():]
        splitted_pattern+=pattern[prev_match.end():].split()
        pos_line = []
        pos_line_tags = []
        # TODO: This seems to be (times) to prevent from protecting too much, but in the end this substitution is not actually
        # good as it does not protect the long overlapping n-grams (this should be ammended)
        # In practice, it has not too much effect as the textual patterns tend to be small, but if you have
        # sequences of 4 tokens (or 2) => this lead to problems
        times = 0
        mask = [False for x in splitted_pattern]
        for string, support in ngrams:
            current_ngram = string.split()
            if support >= SUPPORT_THRESHOLD and times <= 5:
                for pos in range(len(splitted_pattern)-2):
                    if (current_ngram[0] == splitted_pattern[pos]
                        and current_ngram[1] == splitted_pattern[pos+1]
                        and current_ngram[2] == splitted_pattern[pos+2]):
                        mask[pos:pos+3] = [True]*3
                        times += 1
        line = ' '.join(["$" if mask[i] else splitted_pattern[i] for i in range(len(mask))])
        words=line.split()
        assert len(words) == len(line.split(" "))
        for i in range(len(words)):
            if words[i] != "$" and not is_entity(words[i]):
                words[i] = "*"
        print (words)
        # toks = pattern.split(" ")
        for i in range(len(words)):
            if is_entity(words[i]):
                pos_line.append(splitted_pattern[i])
                pos_line_tags.append(splitted_pattern[i])
            elif words[i] == "$":
                pos_line.append(splitted_pattern[i])
                pos_line_tags.append(post[pattern_index][i])
            elif words[i]!=words[i-1]:
                pos_line.append("*")
                pos_line_tags.append("*")
        strpos = ' '.join(pos_line)
        pos_patterns.append(strpos)
        strpos = ' '.join(pos_line_tags)
        sol_pos_patterns.append(strpos)

    return pos_patterns,sol_pos_patterns

def obtainpat(patlist):
    strpat = list()
    entlist = list()
    toks = patlist.split(" ")
    cnt = 0
    for w in toks:
        if w.startswith("CHEMICAL"):
            strpat.append("<CHEMICAL>")
            entlist.append(w)
        elif w.startswith("DISEASE"):
            strpat.append("<DISEASE>")
            entlist.append(w)
        elif w.startswith("GENE"):
            strpat.append("<GENE>")
            entlist.append(w)
        else:
            strpat.append(w)
            if w!="*":
                cnt+=1
    try:
        assert cnt%3==0
    except AssertionError:
        pass
    strpat = ' '.join(strpat)
    entstr = ' '.join(entlist)
    return strpat, entstr

def get_support_of_sols(sol_patterns, sol_pos_patterns):
    """A function to get support of each of the SOL and POS replaced SOL patterns.

    Parameters
    ----------
    sol_patterns : LIST
    sol_pos_patterns : LIST

    Returns
    -------
    type Tuple
        Returns tuple of dictionaries with keys as pattern and value as support.

    """
    suppcloud = dict()
    poscloud = dict()

    pats = list()
    for i in range(len(sol_patterns)):
        pat, ent = obtainpat(sol_patterns[i])
        if pat not in suppcloud:
            pats.append(pat)
            suppcloud[pat] = dict()
            pospat = sol_pos_patterns[i]
            poscloud[pat] = pospat
        if ent not in suppcloud[pat]:
            suppcloud[pat][ent] = 1
        else:
            suppcloud[pat][ent] += 1
    return pats, poscloud, suppcloud
