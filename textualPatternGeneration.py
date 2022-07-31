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

def generate_textual_patterns_with_pos_tags(corpus, extend_children=False):
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

    patterns_pos_tags = []
    textual_patterns = []
    for i, sentence in enumerate(corpus):
        dep_parse = nlp(sentence)
        # print (f'{i}::{sentence}')
        # print (f'{i}:: Ents{dep_parse.ents}')
        try:
            if len(dep_parse.ents) == 2:
                path = shortest_dependency_path(dep_parse, dep_parse[dep_parse.ents[0].start], dep_parse[dep_parse.ents[1].start])
                # print(f'path: {path}')
                # print(f'dep_parse[[0].start]: {dep_parse[dep_parse.ents[0].start]}')
                # print(f'dep_parse[[1].start]: {dep_parse[dep_parse.ents[1].start]}')
                if len(path) != 2:
                    if (extend_children):
                        path = add_children_deps(path, dep_parse)
                    else:
                        path = [int (x) for x in path]
                    # print (f'path: {path}')
                    ## we add the entity and its POS in parallel
                    shortest_path = dep_parse.ents[0].label_+'_<'+str(dep_parse[dep_parse.ents[0].start:dep_parse.ents[0].end]) + '> '
                    pos_tags = ['<'+dep_parse.ents[0].label_+'>']
                    ## we add all the words in the middle in the same order
                    shortest_path += ' '.join([dep_parse[j].text for j in path[1:-1]])
                    for j in path[1:-1]:
                        pos_tags.append(dep_parse[j].pos_)
                    ## and now the last entity
                    shortest_path += ' '+dep_parse.ents[1].label_+'_<'+str(dep_parse[dep_parse.ents[1].start:dep_parse.ents[1].end]) + '> '
                    pos_tags.append('<'+dep_parse.ents[1].label_+'>')
                    # TODO: update the way of extending with advmod (not yet)
                    # textual_patterns.append(adv_mod_deps(shortest_path, dep_parse))
                    textual_patterns.append(shortest_path)
                    patterns_pos_tags.append(pos_tags)
            elif len(dep_parse.ents)> 2:
                pairs = it.combinations(dep_parse.ents, 2)
                for pair in pairs:
                    # print (f'pair: {pair}')
                    # print(f'dep_parse[pair[0].start]: {dep_parse[pair[0].start]}')
                    # print(f'dep_parse[pair[1].start]: {dep_parse[pair[1].start]}')
                    path = shortest_dependency_path(dep_parse, dep_parse[pair[0].start], dep_parse[pair[1].start])
                    if len(path) != 2:
                        if (extend_children):
                            path = add_children_deps(path, dep_parse)
                        else:
                            path = [int(x) for x in path]
                        # print(f'path: {path}')
                        shortest_path = pair[0].label_+'_<'+ str(dep_parse[pair[0].start:pair[0].end]) + '> '
                        pos_tags = ['<' + pair[0].label_ + '>']
                        shortest_path += ' '.join([dep_parse[j].text for j in path[1:-1]])
                        for j in path[1:-1]:
                            pos_tags.append(dep_parse[j].pos_)
                        shortest_path += ' '+pair[1].label_+'_<'+str(dep_parse[pair[1].start:pair[1].end])+'> '
                        pos_tags.append('<' + pair[1].label_ + '>')
                        # TODO: update this
                        # textual_patterns.append(adv_mod_deps(shortest_path, dep_parse))
                        textual_patterns.append(shortest_path)
                        patterns_pos_tags.append(pos_tags)
        except Exception as e:
            print (e)
            pass
    return textual_patterns, patterns_pos_tags





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
    with open(pattern_file, 'w', encoding='utf-8') as f:
        for p in textual_patterns:
            f.write(str(p) + "\n")

def dump_textual_pos_tags_patterns_to_pickle_file(textual_patterns, pos_tag_patterns, filename):
    with open(filename, 'wb') as f:
        pickle.dump([textual_patterns, pos_tag_patterns], f)

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
    with open(pattern_file, 'r', encoding='UTF-8') as f:
        for line in f:
            print(line)
            line = line.strip().rstrip('\n')
            matches = find_entity_matches(line)
            prev_match = next(matches)
            construct = str.lower(line[0:prev_match.start()])
            for current_match in matches:
                construct += line[prev_match.start():prev_match.end()]
                construct += str.lower(line[prev_match.end():current_match.start()])
                prev_match = current_match
            construct += line[prev_match.start():prev_match.end()]
            construct += str.lower(line[prev_match.end():])
            textual_patterns.append(construct)
    return textual_patterns
