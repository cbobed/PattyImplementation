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

def generate_seqmining_dataset(patterns):
    """This function generates a sequence database to mine n-grams from.

    Parameters
    ----------
    patterns : List of Textual Patterns

    Returns
    -------
    type List of Sequences

    """
    smining_dataset = []
    for pattern in patterns:
        matches = re.finditer('[PLO][EOR][RCG][C]?_<.*?>|MISC_<.*?>', pattern)
        prev_match = next(matches)
        smining_dataset.append(pattern[0:prev_match.start()])
        for current_match in matches:
            smining_dataset.append(pattern[prev_match.end():current_match.start()])
            prev_match = current_match
        smining_dataset.append(pattern[prev_match.end():])
    return smining_dataset

def generate_frequent_ngrams(dataset, min_sup):
    """This function mines frequent n-grams from the sequence database

    Parameters
    ----------
    dataset : List of sequences
    min_sup : Minimum support threshold for mining

    Returns
    -------
    Returns a list of n-grams ordered by frequency.

    """
    gen_dict = defaultdict(int)
    for line in dataset:
        lst = line.split()
        for i in range(3, 4):
            for j in range(len(lst) - i + 1):
                gen_dict[tuple(lst[j:j + i])] += 1
    fs = {' '.join(k):v for k,v in gen_dict.items() if v >= min_sup}
    sorted_by_value = sorted(fs.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted_by_value
