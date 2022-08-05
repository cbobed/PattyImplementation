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

import traceback

import pickle
import math
import scipy.stats as st

def build_pattern_and_pos_pattern(doc, start_ent, end_ent, path):
    path = sorted(path)
    shortest_path = ''
    pos_tags = []
    idx = 0
    while idx<len(path):
        if (path[idx] == start_ent.start):
            shortest_path += ' ' + start_ent.label_+'_<'+ str(doc[start_ent.start:start_ent.end]) + '> '
            pos_tags.append('<' + start_ent.label_ + '>')
            while (idx<len(path)) and (path[idx]<start_ent.end):
                idx+=1
        elif (path[idx] == end_ent.start):
            shortest_path += ' '+end_ent.label_+'_<'+str(doc[end_ent.start:end_ent.end])+'> '
            pos_tags.append('<' + end_ent.label_ + '>')
            while (idx<len(path)) and (path[idx]<end_ent.end):
                idx+=1
        else:
            shortest_path += ' '+doc[path[idx]].text
            pos_tags.append(doc[path[idx]].pos_)
            idx += 1
    return shortest_path.strip(), pos_tags

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
#         print(f'ents: {dep_parse.ents}')
        graph = build_graph_from_dependencies(dep_parse)
        try:
            if len(dep_parse.ents) == 2:
                path = shortest_dependency_path(graph, dep_parse[dep_parse.ents[0].start], dep_parse[dep_parse.ents[1].start])
                if len(path) >= 2:
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
                paths = {}
                for pair in pairs:
                    # print (pair)
                    path = shortest_dependency_path(graph, dep_parse[pair[0].start], dep_parse[pair[1].start])
                    if len(path) >= 2:
                        if (extend_children):
                            path = add_children_deps(path, dep_parse)
                        else:
                            path = [int(x) for x in path]
                            # we store all the information to reconstruct the
                        if (pair[0] not in paths):
                            paths[pair[0]] = []
                        paths[pair[0]].append({'path':path, 'ents':(pair[0], pair[1])})
#                 print(f'paths: {paths}')
                # we now clean the possible overlaps for the paths starting in the same entity
                for start_ent in paths:
#                     print (f'{start_ent}')
#                     print(f'{paths[start_ent]}')
                    if len(paths[start_ent]) > 1:
                        sorted_paths = sorted(paths[start_ent], key=lambda x: len(x['path']))
                        cleaned_paths = []
                        cleaned_paths.append(sorted_paths[0])
                        for current_idx in range(1,len(sorted_paths)):
                            cleaned_path = sorted_paths[current_idx]['path']
#                             print(f'before: {cleaned_path} {[dep_parse[x] for x in cleaned_path]}')
                            for prev_idx in range(len(cleaned_paths)):
#                                 print(f'against: {sorted_paths[prev_idx]["path"]} {[dep_parse[x] for x in cleaned_path]}')
                                cleaned_path = [x for x in cleaned_path if x not in sorted_paths[prev_idx]['path']
                                                           or (start_ent.start <= x and x <= (start_ent.end))]
#                             print(f'after: {cleaned_path} {[dep_parse[x] for x in cleaned_path]}')
                            cleaned_paths.append({'path':cleaned_path, 'ents':sorted_paths[current_idx]['ents']})
#                         print(f'cleaned_paths: {cleaned_paths}')
                        for current_idx in range(len(cleaned_paths)):
                            shortest_path, pos_tags = build_pattern_and_pos_pattern(dep_parse,
                                                                                  cleaned_paths[current_idx]['ents'][0],
                                                                                  cleaned_paths[current_idx]['ents'][1],
                                                                                  cleaned_paths[current_idx]['path'] )
                            # TODO: update this
                            # textual_patterns.append(adv_mod_deps(shortest_path, dep_parse))
                            textual_patterns.append(shortest_path)
                            patterns_pos_tags.append(pos_tags)
                    else:
#                         print(f'llegue')
                        shortest_path, pos_tags = build_pattern_and_pos_pattern(dep_parse,
                                                                                  paths[start_ent][0]['ents'][0],
                                                                                  paths[start_ent][0]['ents'][1],
                                                                                  paths[start_ent][0]['path'] )
#                         print(f'sali')

                        textual_patterns.append(shortest_path)
                        patterns_pos_tags.append(pos_tags)
        except Exception as e:
            print (e)
            traceback.print_exc()
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
