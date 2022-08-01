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

sents = load_sentences(sys.argv[1])
print(f'{len(sents)} sentences in the corpus')

# # code to use the spacy set of test sentences
# sents = []
# for s in sentences:
#     sents.append(s)
# sents += sents
# sents += sents
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón se presentó en Zaragoza y saludó varias veces a Manola Perola']
# sents += ['Manolo Salchichón cobra más de 100000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 110000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Longoria cobra más de 120000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 13000 euros', 'Manolo Salchichón compra más de Zaragoza']
# sents += ['Manolo Salchichón cobra más de 10000 euros', 'Manolo Salchichón compra más de Zaragoza']
# print (sents)

print (f'Generating patterns with their POS Tags ... ')
textual_patterns, post = generate_textual_patterns_with_pos_tags(sents, True)
# print (textual_patterns)
print (f'Done. ')

# corpus = read_corpus(os.path.join(params['data_dir'], params['corpus_fn']))
# textual_patterns, post = generate_textual_patterns_with_pos_tags(corpus, False)

write_textual_patterns_to_file("file.txt", textual_patterns)
dump_textual_pos_tags_patterns_to_pickle_file(textual_patterns, post, "TexPatPosTag.pkl")
# This next step could be skipped if required
textual_patterns = convert_textual_patterns_to_lower_case("file.txt")

print (f'Generating the seqmining dataset for mining the most important freq n-gram sets')
seqmining_dataset = generate_seqmining_dataset(textual_patterns)
print (f'Done. ')
# ngrams = generate_frequent_ngrams_only_min_support(seqmining_dataset, 1)
# MINING_ALGORITHM is defined in ngramMining.py
print (f'Mining frequent n-grams with {MINING_ALGORITHM}')
ngrams = generate_frequent_ngrams_mining_alg(seqmining_dataset, MINING_ALGORITHM)
print (f'Done. -- mined {len(ngrams)} ngrams')
print (f'Relevant ngrams: {ngrams}')

print (f'Generating sol and sol_pos patterns ')
sol_patterns = generate_sol_and_sol_pos_patterns(textual_patterns, ngrams, post)
print (f'Done. -- {len(sol_patterns)} patterns so far')
# for i in range(len(sol_patterns)//10):
#     print(f'textual_pattern: {textual_patterns[i*10]}')
#     print (f'sol_pattern: {sol_patterns[i*10]}')
#     print (f'sol_pos_patterns: {sol_pos_patterns[i*10]}')
with open('sp_spp.pkl', 'wb') as f:
    pickle.dump(sol_patterns, f)
print(f'Gathering together all the info about sol_patterns')
sol_info = gather_sol_info(sol_patterns)
print(f'Done. -- {len(sol_info)} patterns collapsed')

## Up to this point we have the support of the patterns, but not the p^u version yet in the dict
with open('pat_pos_supp.pkl', 'wb') as f:
    pickle.dump(sol_info, f)

# for p in range(len(pats)//10):
#     print (f'-------')
#     print (f'p: {pats[p]}')
#     print (f'pos: {poscloud[pats[p]]}')
#     print (f'suppcloud: {suppcloud[pats[p]]}')

# sol_info contains all the information about the SOL pattern, entities, original pattern, ...
# we now try to generalize it, grouping the information as required
print(f'Generalizing patterns syntactically ')
generalized_sol_info = gensyngen(sol_info, ngrams)
print(f'Done . - {len(generalized_sol_info)} generalized patterns  ')
# print ("Syntactic Patterns and supports: ")
# for k in p_s_c:
#     print(f'*****')
#     print (f'pattern: {k} - sup: {p_s_c[k]}')
#     print (f'generalized pattern {detype(k)}: {utc[detype(k)] if detype(k) in utc else "No assoc"} - sup: {utc[detype(k)] if detype(k) in utc else "Nothing"} ')
with open('pscaftersec5.pkl', 'wb') as f:
    pickle.dump([sol_info, generalized_sol_info], f)

# we have to take into account that generalized patterns DO not have an associated confidence (it would be one)
print (f'Calculating the strength and confidence (when applicable ...)')
strength_conf_info = get_strength_confidence(sol_info, generalized_sol_info)
with open('strengthconf.pkl', 'wb') as f:
    pickle.dump(strength_conf_info, f)
print ("Done. ")

print ("Confidence and strength: ordered")
print ("GENERALIZED")
for k in sorted([(y,strength_conf_info[y]) for y in strength_conf_info if strength_conf_info[y][sif.GENERALIZED]], key = lambda x: x[1][sif.STRENGTH], reverse=True):
    print(f' {k[0]} - str:{k[1][sif.STRENGTH]} - conf:{k[1][sif.CONFIDENCE] if sif.CONFIDENCE in k[1] else "No Conf"}')
print( "ORIGINALS")
for k in sorted([(y,strength_conf_info[y]) for y in strength_conf_info if not strength_conf_info[y][sif.GENERALIZED]], key = lambda x: x[1][sif.STRENGTH], reverse=True):
    print(f'{k[0]} - str:{k[1][sif.STRENGTH]} - conf:{k[1][sif.CONFIDENCE] if sif.CONFIDENCE in k[1] else "No Conf"}')

# lconfpat = sorted(conf_pat.items(), key=lambda x: (strength_pat[x[0]]), reverse = True)
# with open("lconfpat.pkl", "wb") as f:
#     pickle.dump(lconfpat, f)
#
# # psc are the patterns with the entities, not the generic ones
# ## before they had directly the support, but not the
# # list of pattern lists syn clouds?
# l_p_l_s_c = convert_patterns_list(p_s_c)
# T, invertList = ConstructPrefixTree(l_p_l_s_c)
#
# SubSump, SubsumW = MineSubsumptions(T, l_p_l_s_c, invertList, 0)
# with open('subsumedgeandweight', 'wb') as f:
#     pickle.dump([SubSump, SubsumW], f)
# #
# # # print ("subsumptions: ")
# # # for k in SubSump:
# # #     print (f'{k} --{l_p_l_s_c[k[0]]} || {l_p_l_s_c[k[1]]} -- {SubsumW[k]}')
# #
# # N = len(l_p_l_s_c)
# # dag, caches = DAGcon(SubsumW, N)
# # with open('dagcaches', 'wb') as f:
# #     pickle.dump([dag, caches], f)
# #
# # # print("---------------")
# # # print("dags: ")
# # # for k in dag:
# # #     print(k)
# # # print("---------------")
# # # for k in caches:
# # #     print(k)
# # # print("---------------")
