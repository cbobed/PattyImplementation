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

from solPatternGeneration import is_entity
from solPatternGeneration import sol_info_fields as sif
from solPatternGeneration import generalized_patterns_info_fields as gpif
from solPatternGeneration import check_ngram

import pickle
import math
import scipy.stats as st
#pats, poscloud, suppcloud

# registersupport(syncloudwithsupport, syncloud, activesyn, patstr, supps[patstr])

def register_generalized_pattern(generalized_sol_pattern, current_sol_pattern, generalized_patterns, sol_info):
    if generalized_sol_pattern not in sol_info:
        if generalized_sol_pattern not in generalized_patterns:
            generalized_patterns[generalized_sol_pattern] = {}
            generalized_patterns[generalized_sol_pattern][gpif.ACTIVE] = True
            generalized_patterns[generalized_sol_pattern][gpif.ORIGINAL_SOL_PATTERNS] = {}
            generalized_patterns[generalized_sol_pattern][gpif.ORIGINAL_SOL_PATTERNS][current_sol_pattern] = sol_info[current_sol_pattern]
            # for efficiency purposes, we will keep the intersection of all the entities set
            # of the SOL patterns that had led to this generalized one
            generalized_patterns[generalized_sol_pattern][gpif.JOINT_SUPPORT] = set()
            generalized_patterns[generalized_sol_pattern][gpif.JOINT_SUPPORT].update(sol_info[current_sol_pattern][sif.ENTITIES])

            # we initialize the information of the generalized pattern as SOL pattern
            # at this point the POS version is not filled
            generalized_patterns[generalized_sol_pattern][sif.SOL_PATTERN] = generalized_sol_pattern
            generalized_patterns[generalized_sol_pattern][sif.TYPE_SIGNATURE] = set()
            generalized_patterns[generalized_sol_pattern][sif.TYPE_SIGNATURE].update(sol_info[current_sol_pattern][sif.TYPE_SIGNATURE])
            generalized_patterns[generalized_sol_pattern][sif.ENTITIES] = set()
            generalized_patterns[generalized_sol_pattern][sif.ENTITIES].update(sol_info[current_sol_pattern][sif.ENTITIES])
        else:
            generalized_patterns[generalized_sol_pattern][gpif.ORIGINAL_SOL_PATTERNS][current_sol_pattern]= sol_info[current_sol_pattern]
            generalized_patterns[generalized_sol_pattern][sif.TYPE_SIGNATURE].update(sol_info[current_sol_pattern][sif.TYPE_SIGNATURE])
            generalized_patterns[generalized_sol_pattern][sif.ENTITIES].update(sol_info[current_sol_pattern][sif.ENTITIES])

            if (generalized_patterns[generalized_sol_pattern][gpif.ACTIVE]):
                # two SOL patterns have been generalized to the same one, we check the supports of the new one
                generalized_patterns[generalized_sol_pattern][gpif.JOINT_SUPPORT].intersection_update(
                                                                    sol_info[current_sol_pattern][sif.ENTITIES])
                generalized_patterns[generalized_sol_pattern][gpif.ACTIVE] = (len(generalized_patterns[generalized_sol_pattern][gpif.JOINT_SUPPORT]) != 0)

# # syncloudwithsupport, syncloud and activesyn are the ones we are filling
# # syn is the current pattern and supp is directly the dict associated to pattern with all the entity signatures
# def registersupport(syncloudwithsupport, syncloud, activesyn, syn, supp):
#     # entity signatures of the pattern
#     # print(f'registering {syn}...')
#     pattern_entity_signatures = set(supp.keys())
#     if syn not in activesyn:
#         #we initialize this pattern
#         activesyn[syn] = True
#         syncloud[syn] = list()
#         syncloud[syn].append(copy.deepcopy(pattern_entity_signatures))
#         syncloudwithsupport[syn] = dict()
#         for k in supp:
#             syncloudwithsupport[syn][k] = {}
#             syncloudwithsupport[syn][k]['accum_sup'] = supp[k]['accum_sup']
#             syncloudwithsupport[syn][k]['original_sup'] = []
#             syncloudwithsupport[syn][k]['original_sup'].append(supp[k]['original_sup'])
#         return
#
#     if activesyn[syn] == False:
#         return
#
#     # if the current pattern is not a generalized one
#     if syn.startswith("<ENTITY>") == False:
#         # print('Not generalized... ---------------------')
#         for sets in syncloud[syn]:
#             # print(f'comparing...')
#             # print(f'{syn}')
#             # print(f' {sets} with {pattern_entity_signatures} -- {len(sets.intersection(pattern_entity_signatures))}')
#             if len(sets.intersection(pattern_entity_signatures)) == 0:
#                 activesyn[syn] = False
#                 return  ## in the original code this was wrongly tabulated and didn't register the generalized versions
#
#     # we aggregate the supports of the pattern into the generalized version
#     syncloud[syn].append(copy.deepcopy(pattern_entity_signatures))
#     for k in supp:
#         if k not in syncloudwithsupport[syn]:
#             syncloudwithsupport[syn][k] = {}
#             syncloudwithsupport[syn][k]['accum_sup'] = supp[k]['accum_sup']
#             syncloudwithsupport[syn][k]['original_sup'] = []
#             syncloudwithsupport[syn][k]['original_sup'].append(supp[k]['original_sup'])
#         else:
#             syncloudwithsupport[syn][k]['accum_sup'] += supp[k]['accum_sup']
#             syncloudwithsupport[syn][k]['original_sup'].append(supp[k]['original_sup'])
#     return

def anything_generalizable(poss):
    return any([not is_entity(x) and x != "*" for x in poss])

def substitute_ne_placeholders(pattern):
    aux = detype(pattern)
    return aux

def remove_contiguous_asterisks(pattern):
    splitted_pattern = pattern.split(' ')
    result = splitted_pattern[0]
    for i in range(1,len(splitted_pattern)):
        if (splitted_pattern[i-1] != '*' or splitted_pattern[i] != '*'):
            result = ' '.join([result, splitted_pattern[i]])
        # else, we omit that entry
    return result
#


def gensyngen(sol_info, ngrams):
    """A function to do syntactic pattern generalization.
    Parameters
    ----------
    sol_info : Dict of sol dicts with all the information of each SOL pattern. Keyed by the SOL pattern itself
    ngrams : the ngram table to be able to build the compactations accordingly
    """
    generalized_patterns = {}
    aggregated_generalized_patterns = {}
    for current_sol in sol_info:
        print(f'originalPattern: {current_sol}')
        pat = current_sol.split(" ")
        poss = sol_info[current_sol][sif.SOL_POS_PATTERN].split(" ")
        if not anything_generalizable(poss):
            continue

        # # First we register the current SOL pattern with its support
        # # syncloudwithsupport, syncloud and activesyn are the ones we are filling
        # registersupport(syncloudwithsupport, syncloud, activesyn, patstr, supps[patstr])
        #

        # Second we generalize it by substituting the placeholders of the NE with the generic type
        ## Note that the support of the newly coined pattern might not be the same (it's taken care of in registersupport)
        # syn = copy.deepcopy(patstr)
        syn = substitute_ne_placeholders(current_sol)
        register_generalized_pattern (syn, current_sol, generalized_patterns, sol_info)
        # registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])
        # We then generalize the patterns by compacting the ngrams
        # ngram contraction

        # we perform just one pass compacting all the n-grams that could be compacted
        # other more comprehensive estrategies could be applied/implemented here
        mask = [False]*len(pat)
        for i in range(len(pat)):
            if is_entity(pat[i]) or pat[i] == "*":
                mask[i] = True

        # we can work just with the False elements
        # if there is an ngram that is not surrounded by any *, we substitute it by *
        current_pos = 0
        syn = ''
        while current_pos < len(pat):
            if not mask[current_pos]:
                for string, support in sorted(ngrams, key=lambda kv: (len(kv[0]), kv[1]), reverse=True):
                    current_ngram = string.split(' ')
                    if (current_pos+len(current_ngram)) < len(pat):
                        if check_ngram(current_ngram, pat, current_pos) and all([not x for x in mask[current_pos:current_pos+len(current_ngram)]]):
                            if (i-1) > 0:
                                if pat[current_pos] != '*':
                                    syn = ' '.join([syn, '*'])
                                    mask[current_pos:current_pos+len(current_ngram)]=[True]*(len(current_ngram))
                                    current_pos += len(current_ngram)
                            elif i+len(current_ngram) < len(pat):
                                if pat[i+len(current_ngram)] != '*':
                                    syn = ' '.join([syn, '*'])
                                    mask[current_pos:current_pos + len(current_ngram)] = [True] * (len(current_ngram))
                                    current_pos += len(current_ngram)
                            else:
                                syn = ' '.join([syn,pat[current_pos]])
                                current_pos += 1
            else:
                syn = ' '.join([syn,pat[current_pos]])
                current_pos += 1
        print(f'before: {syn}')
        syn = remove_contiguous_asterisks(syn)
        print(f'after: {syn}')
        register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)
        syn = substitute_ne_placeholders(syn)
        register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

        lenpos = 0
        for ipos in range(len(pat)):
            if (is_entity(poss[ipos])) or (poss[ipos] =="*"):
                continue
            else:
                lenpos += 1
                ptemp = copy.deepcopy(pat)

                ptemp[ipos] = poss[ipos]
                syn = ' '.join(ptemp)
                register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

                ptemp[ipos] = "[WORD]"
                syn = ' '.join(ptemp)
                register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

                ptemp[ipos] = poss[ipos]
                syn = ' '.join(ptemp)
                syn = substitute_ne_placeholders(syn)
                register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

                ptemp[ipos] = "[WORD]"
                syn = ' '.join(ptemp)
                syn = substitute_ne_placeholders(syn)
                register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

        if lenpos > 1:
            ptemp = copy.deepcopy(pat)
            for ipos in range(len(pat)):
                if (is_entity(poss[ipos])) or (poss[ipos] =="*"):
                    pass
                else:
                    ptemp[ipos] = poss[ipos]
            syn = ' '.join(ptemp)
            register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)
            syn = substitute_ne_placeholders(syn)
            register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

            ptemp = copy.deepcopy(pat)
            for ipos in range(len(pat)):
                if (is_entity(poss[ipos])) or (poss[ipos] =="*"):
                    pass
                else:
                    ptemp[ipos] = "[WORD]"
            syn = ' '.join(ptemp)
            register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)
            syn = substitute_ne_placeholders(syn)
            register_generalized_pattern(syn, current_sol, generalized_patterns, sol_info)

    print(f'Active :: ')
    for gen_pat in [x for x in generalized_patterns if generalized_patterns[x][gpif.ACTIVE]]:
        print(f'{gen_pat} :::: {generalized_patterns[gen_pat]}')
    print(f'Non Active :: ')
    for gen_pat in [x for x in generalized_patterns if not generalized_patterns[x][gpif.ACTIVE]]:
        print(f'{gen_pat} :::: {generalized_patterns[gen_pat]}')
    print(f'active syn size: {len([x for x in generalized_patterns if generalized_patterns[x][gpif.ACTIVE]])}')
    print(f'complete size: {len(generalized_patterns)}')

    for gen_pat in generalized_patterns:
        gen_pat[sif.SOL_PATTERN] = gen_pat
        gen_pat[sif.ENTITIES] = set()
        gen_pat[sif.TYPE_SIGNATURE] = set()
        for orig_pat in generalized_patterns[gpif.ORIGINAL_SOL_PATTERNS]:
            gen_pat[sif.ENTITIES].update(orig_pat[sif.ENTITIES])
            


    return retsyncloud, untypedcloud
