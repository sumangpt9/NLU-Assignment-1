#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import numpy as np
import pickle
import math


# In[2]:


def cosineSimilarity(v1,v2):
    return np.dot(v1,v2)/math.sqrt(np.dot(v1,v1)*np.dot(v2,v2))


# In[6]:


import pandas as pd
df = pd.read_csv("../data/SimLex-999.txt",sep="\t")
#print (df.head())
#df=df.iloc[:,[0,1,3]]
df1 = df[['word1', 'word2','SimLex999']]

#df1 = df.drop(df.columns[[2,4,5,6,7,8,9]], axis=1)
print (df1.head())


# In[9]:


pickle_in = open("../pickles/vocab.pickle","rb")
vocabulary = pickle.load(pickle_in)

pickle_in = open("../pickles/word2id.pickle","rb")
word2idx = pickle.load(pickle_in)

pickle_in = open("../pickles/id2word.pickle","rb")
idx2word = pickle.load(pickle_in)

pickle_in = open("../pickles/embedding.pickle","rb")
W = pickle.load(pickle_in)


# In[12]:


n,m=df1.shape
df1=np.array(df1)
cos_score=np.zeros(n)
sim_score=np.zeros(n)
#for i in range(n):
#    cos_score[i]=-2
c=0
for w1,w2,simscore in df1:
    #print(w1,w2)
    if w1 in vocabulary and w2 in vocabulary:
        i=word2idx[w1]
        j=word2idx[w2]
        cos_score[c]=cosineSimilarity(W[i],W[j])
        sim_score[c]=simscore

    c+=1
print(cos_score,sim_score)


# In[13]:


from scipy import stats
print(stats.spearmanr(cos_score, sim_score))
print(stats.pearsonr(cos_score, sim_score))


# In[ ]:




