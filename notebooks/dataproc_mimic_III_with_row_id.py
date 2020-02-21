#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
import datasets
import log_reg
from dataproc import extract_wvs
from dataproc import get_discharge_summaries
from dataproc import concat_and_split
from dataproc import build_vocab
from dataproc import vocab_index_descriptions
from dataproc import word_embeddings
from constants import MIMIC_3_DIR, DATA_DIR

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
import csv
import math
import operator


# Let's do some data processing in a much better way, with a notebook.
# 
# First, let's define some stuff.

# In[2]:


Y = 'full' #use all available labels in the dataset for prediction
notes_file = '%s/NOTEEVENTS.csv' % MIMIC_3_DIR # raw note events downloaded from MIMIC-III
vocab_size = 'full' #don't limit the vocab size to a specific number
vocab_min = 3 #discard tokens appearing in fewer than this many documents


# # Data processing

# ## Combine diagnosis and procedure codes and reformat them

# The codes in MIMIC-III are given in separate files for procedures and diagnoses, and the codes are given without periods, which might lead to collisions if we naively combine them. So we have to add the periods back in the right place.

# In[3]:


dfproc = pd.read_csv('%s/PROCEDURES_ICD.csv' % MIMIC_3_DIR)
dfdiag = pd.read_csv('%s/DIAGNOSES_ICD.csv' % MIMIC_3_DIR)


# In[4]:


dfdiag['absolute_code'] = dfdiag.apply(lambda row: str(datasets.reformat(str(row[4]), True)), axis=1)
dfproc['absolute_code'] = dfproc.apply(lambda row: str(datasets.reformat(str(row[4]), False)), axis=1)


# In[5]:


dfcodes = pd.concat([dfdiag, dfproc])


# In[6]:


dfcodes.to_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR, index=False,
               columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
               header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])


# ## How many codes are there?

# In[7]:


#In the full dataset (not just discharge summaries)
df = pd.read_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR, dtype={"ICD9_CODE": str})
len(df['ICD9_CODE'].unique())


# ## Tokenize and preprocess raw text

# Preprocessing time!
# 
# This will:
# - Select only discharge summaries and their addenda
# - remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
# - lowercase all tokens

# In[8]:


#This reads all notes, selects only the discharge summaries, and tokenizes them, returning the output filename
disch_full_file = get_discharge_summaries.write_discharge_summaries(out_file="%s/disch_full.csv" % MIMIC_3_DIR)


# Let's read this in and see what kind of data we're working with

# In[9]:


df = pd.read_csv('%s/disch_full.csv' % MIMIC_3_DIR)


# In[10]:


#How many admissions?
len(df['HADM_ID'].unique())


# In[11]:


#Tokens and types
types = set()
num_tok = 0
for row in df.itertuples():
    for w in row[5].split():
        types.add(w)
        num_tok += 1


# In[12]:


print("Num types", len(types))
print("Num tokens", str(num_tok))


# In[13]:


#Let's sort by SUBJECT_ID and HADM_ID to make a correspondence with the MIMIC-3 label file
df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])


# In[14]:


#Sort the label file by the same
dfl = pd.read_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR)
dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])


# In[15]:


len(df['HADM_ID'].unique()), len(dfl['HADM_ID'].unique())


# ## Consolidate labels with set of discharge summaries

# Looks like there were some HADM_ID's that didn't have discharge summaries, so they weren't included with our notes

# In[16]:


#Let's filter out these HADM_ID's
hadm_ids = set(df['HADM_ID'])
with open('%s/ALL_CODES.csv' % MIMIC_3_DIR, 'r') as lf:
    with open('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, 'w') as of:
        w = csv.writer(of)
        w.writerow(['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
        r = csv.reader(lf)
        #header
        next(r)
        for i,row in enumerate(r):
            hadm_id = int(row[2])
            #print(hadm_id)
            #break
            if hadm_id in hadm_ids:
                w.writerow(row[0:3] + [row[-1], '', ''])


# In[17]:


dfl = pd.read_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index_col=None)


# In[18]:


len(dfl['HADM_ID'].unique())


# In[19]:


#we still need to sort it by HADM_ID
dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
dfl.to_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index=False)


# ## Append labels to notes in a single file

# In[20]:


#Now let's append each instance with all of its codes
#this is pretty non-trivial so let's use this script I wrote, which requires the notes to be written to file
sorted_file = '%s/disch_full.csv' % MIMIC_3_DIR
df.to_csv(sorted_file, index=False)


# In[21]:


labeled = concat_and_split.concat_data('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, sorted_file)


# In[22]:


#name of the file we just made
print(labeled)


# Let's sanity check the combined data we just made. Do we have all hadm id's accounted for, and the same vocab stats?

# In[23]:


dfnl = pd.read_csv(labeled)
#Tokens and types
types = set()
num_tok = 0
for row in dfnl.itertuples():
    for w in row[4].split():
        types.add(w)
        num_tok += 1


# In[24]:


print("num types", len(types), "num tokens", num_tok)


# In[25]:


len(dfnl['HADM_ID'].unique())


# ## Create train/dev/test splits

# In[26]:


fname = '%s/notes_labeled.csv' % MIMIC_3_DIR
base_name = "%s/disch" % MIMIC_3_DIR #for output
tr, dv, te = concat_and_split.split_data(fname, base_name=base_name)


# ## Build vocabulary from training data

# In[27]:


vocab_min = 3
vname = '%s/vocab.csv' % MIMIC_3_DIR
build_vocab.build_vocab(vocab_min, tr, vname)


# ## Sort each data split by length for batching

# In[28]:


for splt in ['train', 'dev', 'test']:
    filename = '%s/disch_%s_split.csv' % (MIMIC_3_DIR, splt)
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_full.csv' % (MIMIC_3_DIR, splt), index=False)


# ## Pre-train word embeddings

# Let's train word embeddings on all words

# In[29]:


w2v_file = word_embeddings.word_embeddings('full', '%s/disch_full.csv' % MIMIC_3_DIR, 100, 0, 5)


# ## Write pre-trained word embeddings with new vocab

# In[30]:


extract_wvs.gensim_to_embeddings('%s/processed_full.w2v' % MIMIC_3_DIR, '%s/vocab.csv' % MIMIC_3_DIR, Y)


# ## Pre-process code descriptions using the vocab

# In[31]:


vocab_index_descriptions.vocab_index_descriptions('%s/vocab.csv' % MIMIC_3_DIR,
                                                  '%s/description_vectors.vocab' % MIMIC_3_DIR)


# ## Filter each split to the top 50 diagnosis/procedure codes

# In[32]:


Y = 50


# In[33]:


#first calculate the top k
counts = Counter()
dfnl = pd.read_csv('%s/notes_labeled.csv' % MIMIC_3_DIR)
for row in dfnl.itertuples():
    for label in str(row[5]).split(';'):
        counts[label] += 1


# In[34]:


codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)


# In[35]:


codes_50 = [code[0] for code in codes_50[:Y]]


# In[36]:


codes_50


# In[37]:


with open('%s/TOP_%s_CODES.csv' % (MIMIC_3_DIR, str(Y)), 'w') as of:
    w = csv.writer(of)
    for code in codes_50:
        w.writerow([code])


# In[38]:


for splt in ['train', 'dev', 'test']:
    print(splt)
    hadm_ids = set()
    with open('%s/%s_50_hadm_ids.csv' % (MIMIC_3_DIR, splt), 'r') as f:
        for line in f:
            hadm_ids.add(line.rstrip())
    with open('%s/notes_labeled.csv' % MIMIC_3_DIR, 'r') as f:
        with open('%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), 'w') as of:
            r = csv.reader(f)
            w = csv.writer(of)
            #header
            w.writerow(next(r))
            i = 0
            for row in r:
                hadm_id = row[2]
                if hadm_id not in hadm_ids:
                    continue
                codes = set(str(row[4]).split(';'))
                filtered_codes = codes.intersection(set(codes_50))
                if len(filtered_codes) > 0:
                    w.writerow(row[:4] + [';'.join(filtered_codes)])
                    i += 1


# In[39]:


for splt in ['train', 'dev', 'test']:
    filename = '%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y))
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), index=False)

