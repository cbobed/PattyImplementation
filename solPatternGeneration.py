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

import utils

import pickle
import math
import scipy.stats as st

MAX_NGRAM_LENGTH = 3

class sol_info_fields():
    SINGLE_ORIGINAL_PATTERN = 'original_pattern'
    ORIGINAL_PATTERNS = 'original_patterns'
    SOL_PATTERN = 'sol_pattern'
    SOL_POS_PATTERN = 'sol_pos_pattern'
    ENTITIES = 'entities'
    TYPE_SIGNATURE = 'type_signature'
    STRENGTH = 'strength'
    CONFIDENCE = 'confidence'
    UNTYPED = 'untyped'
    GENERALIZED = 'generalized'

class generalized_patterns_info_fields():
    ACTIVE = 'active'
    ORIGINAL_SOL_PATTERNS = 'original_patterns'
    JOINT_SUPPORT = 'joint_support'

def check_ngram(ngram, pattern, starting_pos):
    if (starting_pos+(len(ngram)-1) > len(pattern)-1) :
        return False
    return all([ngram[i] == pattern[starting_pos+i] for i in range(len(ngram))])

# replacing non entity non frequent n gram by wildcard
# for each sol pattern we now keep the type signatures witnessed,
# the syntactic support, and the entity support (pairs of entities supportting the type signature)
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
    type List of dicts
        containing the pattern, the pos_pattern, the sol_pattern, the type signature and the support (duples of entities)
    """
    sol_patterns = []
    for pattern_index, pattern in enumerate(patterns):
        entities = []
        type_signature = []
        splitted_pattern = []
        matches = utils.find_entity_matches(pattern)
        prev_match = next(matches)
        line = pattern [0:prev_match.start()]
        splitted_pattern+=pattern [0:prev_match.start()].split()
        line += "<"+pattern[prev_match.start():prev_match.end()].split('_<')[0]+">"
        type_signature.append(pattern[prev_match.start():prev_match.end()].split('_<')[0])
        entities.append(pattern[prev_match.start():prev_match.end()].split('_<')[1].rstrip('>'))
        splitted_pattern.append("<" + pattern[prev_match.start():prev_match.end()].split('_<')[0] + ">")
        ## There should be only one more
        for current_match in matches:
            line+=pattern[prev_match.end():current_match.start()]
            splitted_pattern+=pattern[prev_match.end():current_match.start()].split()
            line+="<"+pattern[current_match.start():current_match.end()].split('_<')[0]+">"
            splitted_pattern.append("<"+pattern[current_match.start():current_match.end()].split('_<')[0]+">")
            type_signature.append(pattern[current_match.start():current_match.end()].split('_<')[0])
            entities.append(pattern[current_match.start():current_match.end()].split('_<')[1].rstrip('>'))
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
            if times <= 5:
                for pos in range(len(splitted_pattern)-(MAX_NGRAM_LENGTH-1)):
                    if (check_ngram(current_ngram, splitted_pattern, pos)):
                        # we only consider a substitution if something is going to change in
                        ## any position of the current analyzed is False
                        if any([not x for x in mask[pos:pos+len(current_ngram)]]):
                            times += 1
                            mask[pos:pos+len(current_ngram)] = [True]*len(current_ngram)

        line = ' '.join(["$" if mask[i] else splitted_pattern[i] for i in range(len(mask))])
        words=line.split()
        # assert len(words) == len(line.split(" "))
        # we only keep the tokens belonging to an entity or a substituted ngram ($)
        for i in range(len(words)):
            if words[i] != "$" and not utils.is_entity(words[i]):
                words[i] = "*"
        # print (words)
        # toks = pattern.split(" ")
        for i in range(len(words)):
            if utils.is_entity(words[i]):
                pos_line.append(splitted_pattern[i])
                pos_line_tags.append(splitted_pattern[i])
            elif words[i] == "$":
                pos_line.append(splitted_pattern[i])
                pos_line_tags.append(post[pattern_index][i])
            elif words[i]!=words[i-1]:
                pos_line.append("*")
                pos_line_tags.append("*")
        aux_sol_pattern = {}
        aux_sol_pattern[sol_info_fields.SINGLE_ORIGINAL_PATTERN] = patterns[pattern_index]
        aux_sol_pattern[sol_info_fields.SOL_PATTERN] = ' '.join(pos_line)
        aux_sol_pattern[sol_info_fields.SOL_POS_PATTERN] = ' '.join(pos_line_tags)
        aux_sol_pattern[sol_info_fields.ENTITIES] = entities
        aux_sol_pattern[sol_info_fields.TYPE_SIGNATURE] = type_signature
        sol_patterns.append(aux_sol_pattern)
        # print(f'-----------')
        # print(f'pattern: {pattern}')
        # print(f'pattern key: {aux_sol_pattern["original_pattern"]}')
        # print(f'splitted_pattern:{splitted_pattern}')
        # print(f'mask:{mask}')
        # print(f'line: {line}')
        # print(f'sol_pattern: {aux_sol_pattern["sol_pattern"]}')
        # print(f'sol_pos_pattern: {aux_sol_pattern["sol_pos_pattern"]}')
        # print(f'entities: {aux_sol_pattern["entities"]}')
        # print(f'type_signature: {aux_sol_pattern["type_signature"]}')
        # print(f'-----------')
    return sol_patterns

# def obtainpat(patlist):
#     strpat = list()
#     entlist = list()
#     toks = patlist.split(" ")
#     cnt = 0
#     for w in toks:
#         if is_entity(w):
#             strpat.append(w)
#             entlist.append(w)
#         else:
#             strpat.append(w)
#             if w!="*":
#                 cnt+=1
#
#     strpat = ' '.join(strpat)
#     entstr = ' '.join(entlist)
#     return strpat, entstr

def gather_sol_info(sol_patterns):
    """A function to gather together all the info of each of the SOL and POS replaced SOL patterns.

    Parameters
    ----------
    sol_patterns : LIST of DICTS - Keys: 'original_pattern ', 'sol_pattern', 'sol_pos_pattern', 'entities', 'type_signature'
    Returns
    -------
    type DICT with all the information about the SOL patterns gathered

    """
    sol_info = dict()
    for i in range(len(sol_patterns)):
        pat = sol_patterns[i][sol_info_fields.SOL_PATTERN]
        if pat not in sol_info:
            sol_info[pat] = dict()
            sol_info[pat][sol_info_fields.SOL_POS_PATTERN] = sol_patterns[i][sol_info_fields.SOL_POS_PATTERN]
            sol_info[pat][sol_info_fields.ENTITIES] = set()
            sol_info[pat][sol_info_fields.TYPE_SIGNATURE] = set ()
            sol_info[pat][sol_info_fields.ORIGINAL_PATTERNS] = set ()
        sol_info[pat][sol_info_fields.ENTITIES].add(tuple(sol_patterns[i][sol_info_fields.ENTITIES]))
        sol_info[pat][sol_info_fields.TYPE_SIGNATURE].add(tuple(sol_patterns[i][sol_info_fields.TYPE_SIGNATURE]))
        sol_info[pat][sol_info_fields.ORIGINAL_PATTERNS].add(pat)

    if (len (sol_info[pat][sol_info_fields.TYPE_SIGNATURE]) > 1):
            print (f'Wrong type_signature size: {sol_info[pat]}')
    return sol_info
