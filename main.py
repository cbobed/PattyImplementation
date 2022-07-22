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
import pickle
import math
from utils import *
import utils
import scipy.stats as st
from dagConstruction import *
from mineSubsumptions import *
from ngramMining import *
from prefixTreeConstruction import *
from solPatternGeneration import *
from syntacticPatternGeneralization import *
from textualPatternGeneration import *
from sid.SentencesLoader import load_sentences

# from spacy.lang.en.examples import sentences
from spacy.lang.es.examples import sentences

# params = {'data_dir': sys.argv[1], 'corpus_fn': sys.argv[2]}
# textual_patterns = generate_textual_patterns(corpus)

# sents = load_sentences(sys.argv[1], limit=10)
# print(f'{len(sents)} sentences in the corpus')

# code to use the spacy set of test sentences
sents = []
for s in sentences:
    sents.append(s)
sents += sents
sents += sents
sents += ['Manolo cobra más de 10000 euros', 'Manolo compra más de Zaragoza']
sents += ['Manolo cobra más de 10000 euros', 'Manolo compra más de Zaragoza']
sents += ['Manolo cobra más de 10000 euros', 'Manolo compra más de Zaragoza']
sents += ['Manolo cobra más de 10000 euros', 'Manolo compra más de Zaragoza']
sents += ['Manolo cobra más de 10000 euros', 'Manolo compra más de Zaragoza']
sents += ['Manolo cobra más de 10000 euros', 'Manolo compra más de Zaragoza']
print (sents)

textual_patterns, post = generate_textual_patterns_with_pos_tags(sents, False)
print (textual_patterns)

# corpus = read_corpus(os.path.join(params['data_dir'], params['corpus_fn']))
# textual_patterns, post = generate_textual_patterns_with_pos_tags(corpus, False)

write_textual_patterns_to_file("file.txt", textual_patterns)
dump_textual_pos_tags_patterns_to_pickle_file(textual_patterns, post, "TexPatPosTag.pkl")
textual_patterns = convert_textual_patterns_to_lower_case("file.txt")

seqmining_dataset = generate_seqmining_dataset(textual_patterns)
ngrams = generate_frequent_ngrams(seqmining_dataset, 1)

sol_patterns, sol_pos_patterns = generate_sol_and_sol_pos_patterns(textual_patterns, ngrams, post)
for i in range(len(sol_patterns)//10):
    print(f'textual_pattern: {textual_patterns[i*10]}')
    print (f'sol_pattern: {sol_patterns[i*10]}')
    print (f'sol_pos_patterns: {sol_pos_patterns[i*10]}')
with open('sp_spp.pkl', 'wb') as f:
    pickle.dump([sol_patterns, sol_pos_patterns], f)
pats, poscloud, suppcloud = get_support_of_sols(sol_patterns, sol_pos_patterns)

## Up to this point we have the support of the patterns, but not the p^u version yet in the dict
with open('pat_pos_supp.pkl', 'wb') as f:
    pickle.dump([pats, poscloud, suppcloud], f)

for p in pats:
    print (f'-------')
    print (f'p: {p}')
#     print (f'pos: {poscloud[p]}')
#     print (f'suppcloud: {suppcloud[p]}')

# pats is a list of the patterns
# poscloud is a dict with the pos version of the pattern used to access it
# suppcloud is a dict accessed by patterns containing dicts with the support of [pattern][entity_signature]
p_s_c, utc = gensyngen(pats, poscloud, suppcloud)
print ("Syntactic Patterns and supports: ")
for k in p_s_c:
    print (f'pattern: {k}')
#     print (f'typed: {p_s_c[k]}')
for k in utc:
    print (f'pattern: {k}')
#     print (f'untyped: {utc[k]}')
with open('pscaftersec5.pkl', 'wb') as f:
    pickle.dump([p_s_c, utc], f)

strength_pat, conf_pat = get_strength_confidence(p_s_c, utc)
with open('strengthconf.pkl', 'wb') as f:
    pickle.dump([strength_pat, conf_pat], f)

print ("---------------")
print ("Strength_pat: ")
for k in strength_pat:
    print(f'{k} - {strength_pat[k]}')
for k in conf_pat:
    print(f'{k} - {conf_pat[k]}')

#
# lconfpat = sorted(conf_pat.items(), key=lambda x: (strength_pat[x[0]]), reverse = True)
# with open("lconfpat.pkl", "wb") as f:
#     pickle.dump(lconfpat, f)
#
# print ('----lconfpat-----')
# for k in lconfpat:
#     print(k)
#
#
# l_p_l_s_c = convert_patterns_list(p_s_c)
# T, invertList = ConstructPrefixTree(l_p_l_s_c)
# SubSump, SubsumW = MineSubsumptions(T, l_p_l_s_c, invertList, 0)
# with open('subsumedgeandweight', 'wb') as f:
#     pickle.dump([SubSump, SubsumW], f)
#
# # print ("subsumptions: ")
# # for k in SubSump:
# #     print (f'{k} --{l_p_l_s_c[k[0]]} || {l_p_l_s_c[k[1]]} -- {SubsumW[k]}')
#
# N = len(l_p_l_s_c)
# dag, caches = DAGcon(SubsumW, N)
# with open('dagcaches', 'wb') as f:
#     pickle.dump([dag, caches], f)
#
# # print("---------------")
# # print("dags: ")
# # for k in dag:
# #     print(k)
# # print("---------------")
# # for k in caches:
# #     print(k)
# # print("---------------")
