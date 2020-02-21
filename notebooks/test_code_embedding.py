from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

MIMIC_3_DIR = r'C:\Users\hdong3\OneDrive - University of Edinburgh\My python projects\caml-mimic\mimicdata\mimic3'
dim = 500

#code for testing
model = Word2Vec.load("%s/code-emb-mimic3-tr-%s.model" % (MIMIC_3_DIR,dim))
vector = model.wv['414.01']
print(vector)
print(np.dot(vector,vector))
print('vocab size:',len(model.wv.vocab))