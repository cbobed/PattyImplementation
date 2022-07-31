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

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax

# The nlp model is loaded and configured at utils
from utils import *
import utils

import pickle
import math
import scipy.stats as st

MIN_NGRAM_LENGTH=2
MAX_NGRAM_LENGTH=3

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
        matches = find_entity_matches(pattern)
        prev_match = next(matches)
        smining_dataset.append(pattern[0:prev_match.start()])
        for current_match in matches:
            smining_dataset.append(pattern[prev_match.end():current_match.start()])
            prev_match = current_match
        smining_dataset.append(pattern[prev_match.end():])
    return smining_dataset

def generate_frequent_ngrams_only_min_support(dataset, min_sup):
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
        for i in range(MIN_NGRAM_LENGTH, MAX_NGRAM_LENGTH+1):
            for j in range(len(lst) - i + 1):
                gen_dict[tuple(lst[j:j + i])] += 1
    fs = {' '.join(k):v for k,v in gen_dict.items() if v >= min_sup}
    sorted_by_value = sorted(fs.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    return sorted_by_value

def translate_transaction_comma_separated(transaction, translation_table):
    return ','.join([translation_table[x] for x in transaction])

def generate_frequent_ngrams_mining_alg(dataset, alg='APRIORI'):
    ngram_dict = {}
    reverse_dict = {}
    ngram_id = 0
    pattern_database = []
    for line in dataset:
        lst = line.split()
        current_pattern_transaction = []
        for i in range(MIN_NGRAM_LENGTH, MAX_NGRAM_LENGTH+1):
            for j in range(len(lst) - i + 1):
                if (tuple(lst[j:j+i]) not in ngram_dict):
                    ngram_dict[tuple(lst[j:j+i])] = ngram_id
                    reverse_dict[ngram_id] = ' '.join(lst[j:j+i])
                    ngram_id += 1
                current_pattern_transaction.append(ngram_dict[tuple(lst[j:j+i])])
        if (len(current_pattern_transaction) > 0):
            pattern_database.append(current_pattern_transaction)
    tr = TransactionEncoder()
    tr_arr = tr.fit(pattern_database).transform(pattern_database)
    df = pd.DataFrame(tr_arr, columns=tr.columns_)
    if (alg == 'APRIORI'):
        frequent_ngram_combinations = apriori(df, min_support=0.1)
    elif (alg == 'FPGROWTH'):
        frequent_ngram_combinations = fpgrowth(df, min_support=0.1)
    elif (alg == 'FPMAX'):
        frequent_ngram_combinations = fpmax (df, min_support=0.1)

    fs ={}
    for i in range(len(frequent_ngram_combinations)):
        # we "flatten" the itemsets, but we get the max of the supports of all of them
        for item in frequent_ngram_combinations.iloc[i]['itemsets']:
            if reverse_dict[item] not in fs:
                fs[reverse_dict[item]] = 0
            fs[reverse_dict[item]] = max(fs[reverse_dict[item]], frequent_ngram_combinations.iloc[i]['support'])

    sorted_by_support = sorted(fs.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    return sorted_by_support