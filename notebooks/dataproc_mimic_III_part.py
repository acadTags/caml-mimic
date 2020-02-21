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


print(codes_50)


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

