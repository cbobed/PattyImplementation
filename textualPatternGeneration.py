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

def generate_textual_patterns(corpus):
    """A method to generate textual patterns given the corpus.

    Parameters
    ----------
    corpus : type List
        List of sentences is passed.

    Returns
    -------
    type List
        List of textual patterns

    """
    textual_patterns = []
    for i, sentence in enumerate(corpus):
        dep_parse = nlp(sentence)
        print (f'{i}::{sentence}')
        try:
            if len(dep_parse.ents) == 2:
                path = shortest_dependency_path(dep_parse, dep_parse[dep_parse.ents[0].start], dep_parse[dep_parse.ents[1].start])
                if len(path) != 2:
                    shortest_path = dep_parse.ents[0].label_+'_<'+str(dep_parse[dep_parse.ents[0].start:dep_parse.ents[0].end]) + '> '
                    shortest_path += ' '.join([dep_parse[int(j)].text for j in path[1:-1]])
                    shortest_path += ' '+dep_parse.ents[1].label_+'_<'+str(dep_parse[dep_parse.ents[1].start:dep_parse.ents[1].end]) + '> '
                    textual_patterns.append(adv_mod_deps(shortest_path, dep_parse))

            elif len(dep_parse.ents)> 2:
                pairs = it.combinations(dep_parse.ents, 2)
                for pair in pairs:
                    path = shortest_dependency_path(dep_parse, dep_parse[pair[0].start], dep_parse[pair[1].start])
                    if len(path) != 2:
                        shortest_path = pair[0].label_+'_<'+ str(dep_parse[pair[0].start:pair[0].end]) + '> '
                        shortest_path += ' '.join([dep_parse[int(j)].text for j in path[1:-1]])
                        shortest_path += ' '+pair[1].label_+'_<'+str(dep_parse[pair[1].start:pair[1].end])+'> '
                        textual_patterns.append(adv_mod_deps(shortest_path, dep_parse))
        except Exception as e:
            print (e)
            pass
    return(textual_patterns)

def write_textual_patterns_to_file(pattern_file, textual_patterns):
    """A utility to write the generated textual patterns to a file.

    Parameters
    ----------
    pattern_file : type Path
        Path of the file
    textual_patterns : type List
        List of textual patterns

    Returns
    -------
    type None
        Doesn't return anything

    """
    with open(pattern_file, 'w',encoding='utf-8') as f:
        for p in textual_patterns:
            f.write(str(p) + "\n")

def convert_textual_patterns_to_lower_case(pattern_file):
    """Converts patterns to lower case barring the entities.

    Parameters
    ----------
    pattern_file : type Path
        The file containing the textual patterns

    Returns
    -------
    type List
        Returns the list of textual patterns converted to lowercase.

    """
    textual_patterns = []
    with open(pattern_file, 'r') as f:
        for line in f:
            line = line.strip()
            w = line.split(" ")
            if(len(w) <=2):
                continue
            f = 0
            if w[0].startswith("CHEMICAL_") or w[0].startswith("DISEASE_") or w[0].startswith("GENE_"):
                pass
            else:
                f = 1
            if w[len(w)-1].startswith("CHEMICAL_") or w[len(w)-1].startswith("DISEASE_") or w[len(w)-1].startswith("GENE_"):
                pass
            else:
                f = 1
            if f == 0:
                fl = 0
                for ii in range(len(w)):
                    i = w[ii]
                    fl = 1*("CHEMICAL_" in i) + 1*("DISEASE_" in i) + 1*("GENE_" in i)
                    if fl!=1:
                        w[ii]  = str.lower(i)
                    if fl > 1:
                        break
                if fl > 1:
                    continue
                strmed = ' '.join(w[1:len(w)-1])
                strmed = str(w[0]) + " " + strmed + " " + str(w[len(w)-1])
                textual_patterns.append(strmed)
    return textual_patterns
