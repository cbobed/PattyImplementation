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

import pickle
import math
import scipy.stats as st

#pats, poscloud, suppcloud

# registersupport(syncloudwithsupport, syncloud, activesyn, patstr, supps[patstr])

# syncloudwithsupport, syncloud and activesyn are the ones we are filling
# syn is the current pattern and supp is directly the dict associated to pattern with all the entity signatures
def registersupport(syncloudwithsupport, syncloud, activesyn, syn, supp):
    # entity signatures of the pattern
    print(f'registering {syn}...')
    pattern_entity_signatures = set(supp.keys())
    if syn not in activesyn:
        #we initialize this pattern
        activesyn[syn] = True
        syncloud[syn] = list()
        syncloud[syn].append(copy.deepcopy(pattern_entity_signatures))
        syncloudwithsupport[syn] = dict()
        for k in supp:
            syncloudwithsupport[syn][k] = {}
            syncloudwithsupport[syn][k]['accum_sup'] = supp[k]['accum_sup']
            syncloudwithsupport[syn][k]['original_sup'] = []
            syncloudwithsupport[syn][k]['original_sup'].append(supp[k]['original_sup'])
        return

    if activesyn[syn] == False:
        return

    # if the current pattern is not a generalized one
    if syn.startswith("<ENTITY>") == False:
        print('Not generalized... ---------------------')
        for sets in syncloud[syn]:
            print(f'comparing...')
            print(f'{syn}')
            print(f' {sets} with {pattern_entity_signatures} -- {len(sets.intersection(pattern_entity_signatures))}')
            if len(sets.intersection(pattern_entity_signatures)) == 0:
                activesyn[syn] = False
                return  ## in the original code this was wrongly tabulated and didn't register the generalized versions

    # we aggregate the supports of the pattern into the generalized version
    syncloud[syn].append(copy.deepcopy(pattern_entity_signatures))
    for k in supp:
        if k not in syncloudwithsupport[syn]:
            syncloudwithsupport[syn][k] = {}
            syncloudwithsupport[syn][k]['accum_sup'] = supp[k]['accum_sup']
            syncloudwithsupport[syn][k]['original_sup'] = []
            syncloudwithsupport[syn][k]['original_sup'].append(supp[k]['original_sup'])
        else:
            syncloudwithsupport[syn][k]['accum_sup'] += supp[k]['accum_sup']
            syncloudwithsupport[syn][k]['original_sup'].append(supp[k]['original_sup'])
    return

def anything_generalizable(poss):
    return any([not is_entity(x) and x != "*" for x in poss])

def substitute_ne_placeholders(pattern):
    aux = detype(pattern)
    return aux

#pattern is a list- each list elements will be one of entity, n-grams, *
#sup is set of tuples. tuples size will be equal to no. of entity in pattern
def gensyngen(pats, poscloud, supps):
    """A function to do syntactic pattern generalization.

    Parameters
    ----------
    pats : List of patterns.
    poscloud : List of POS patterns
    supps: List of dicts with the support of patterns for each entity singnature

    """
    syncloud = dict()
    activesyn = dict()
    syncloudwithsupport = dict()

    for p in range(len(pats)):
        print(f'originalPattern: {pats[p]}')
        patstr = pats[p]
        pat = patstr.split(" ")
        poss = poscloud[patstr]
        poss = poss.split(" ")

        if not anything_generalizable(poss):
            continue

        # First we register the current SOL pattern with its support
        # syncloudwithsupport, syncloud and activesyn are the ones we are filling
        registersupport(syncloudwithsupport, syncloud, activesyn, patstr, supps[patstr])

        # Second we generalize it by substituting the placeholders of the NE with the generic type
        ## Note that the support of the newly coined pattern might not be the same (it's taken care of in registersupport)
        syn = copy.deepcopy(patstr)
        syn = substitute_ne_placeholders(syn)
        registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])
        # We then generalize the patterns by compacting the ngrams
        #ngram contraction
        Nngram = list()
        for i in range(len(pat)):
            if is_entity(pat[i]) or pat[i] == "*":
                pass
            else:
                Nngram.append(i)
        try:
            assert len(Nngram)%3 == 0
        except AssertionError:
            print(f'assertion error: {patstr}')

        for ii in range(0,len(Nngram),3):
            ing = Nngram[ii]
            syn = " "
            tok = []
            for i in range(len(pat)):
                if (i == ing+1) or (i== ing + 2):
                    pass
                elif i == ing:
                    if (i+3 < len(pat) and (i-1) > 0):
                        if pat[i+3] != "*"  or pat[i-1] != "*":
                            tok.append("*")
                else:
                    tok.append(pat[i])
            syn = ' '.join(tok)
            syn = syn.replace(" * * "," * ")
            registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])
            syn = substitute_ne_placeholders(syn)
            registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

        lenpos = 0

        for ipos in range(len(pat)):
            if (is_entity(poss[ipos])) or (poss[ipos] =="*"):
                continue
            else:
                lenpos += 1
                ptemp = copy.deepcopy(pat)

                ptemp[ipos] = poss[ipos]
                syn = ' '.join(ptemp)
                registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

                ptemp[ipos] = "[WORD]"
                syn = ' '.join(ptemp)
                registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

                ptemp[ipos] = poss[ipos]
                syn = ' '.join(ptemp)
                syn = substitute_ne_placeholders(syn)
                registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

                ptemp[ipos] = "[WORD]"
                syn = ' '.join(ptemp)
                syn = substitute_ne_placeholders(syn)
                registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

        if lenpos > 1:
            ptemp = copy.deepcopy(pat)
            for ipos in range(len(pat)):
                if (is_entity(poss[ipos])) or (poss[ipos] =="*"):
                    pass
                else:
                    ptemp[ipos] = poss[ipos]
            syn = ' '.join(ptemp)
            registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])
            syn = substitute_ne_placeholders(syn)
            registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

            ptemp = copy.deepcopy(pat)
            for ipos in range(len(pat)):
                if (is_entity(poss[ipos])) or (poss[ipos] =="*"):
                    pass
                else:
                    ptemp[ipos] = "[WORD]"
            syn = ' '.join(ptemp)
            registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])
            syn = substitute_ne_placeholders(syn)
            registersupport(syncloudwithsupport, syncloud, activesyn, syn, supps[patstr])

        #raise ValueError("bs")

    print(f'syn size: {len(syncloud)}')
    retsyncloud = dict()
    untypedcloud = dict()
    for syn in syncloud:
        if activesyn[syn] == True:
            if syn.startswith("<ENTITY>") == False:
                retsyncloud[syn] = copy.deepcopy(syncloudwithsupport[syn])
            else:
                untypedcloud[syn] = copy.deepcopy(syncloudwithsupport[syn])
    #print(ghanta)
    print(f'both :{len(retsyncloud) + len(untypedcloud)}')
    print(f'non_active: {len([x for x in activesyn if not activesyn[x]])}')
    return retsyncloud, untypedcloud
