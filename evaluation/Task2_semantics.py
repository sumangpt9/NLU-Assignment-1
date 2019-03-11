#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pickle
import math


# In[8]:


def cosineSimilarity(v1,v2):
    return np.dot(v1,v2)/math.sqrt(np.dot(v1,v1)*np.dot(v2,v2))


# In[6]:


pickle_in = open("../pickles/vocab.pickle","rb")
vocabulary = pickle.load(pickle_in)

pickle_in = open("../pickles/word2id.pickle","rb")
word2idx = pickle.load(pickle_in)

pickle_in = open("../pickles/id2word.pickle","rb")
idx2word = pickle.load(pickle_in)

pickle_in = open("../pickles/embedding.pickle","rb")
W = pickle.load(pickle_in)


# In[9]:


filenames=['all_capitals','capital_common_countries','city_state','currency','family']
total=0
total_Entries=0
correct=0
for j in range(len(filenames)):

    df=pd.read_csv('../data/'+filenames[j]+'.txt',sep=" ")
    df=np.array(df)
    #print (df.head)
    for w1,w2,w3,w4 in df:
        total_Entries=total_Entries+1
        if w1 in vocabulary and w2 in vocabulary and w3 in vocabulary and w4 in vocabulary:
            print(w1,w2,w3,w4)
            w1=w1.lower()
            w2=w2.lower()
            w3=w3.lower()
            w4=w4.lower()
            total=total+1
            v=W[word2idx[w1]]-W[word2idx[w2]]+W[word2idx[w3]]
            max_similarity=-2
            max_similar=0
            for i in range(len(vocabulary)):
                if cosineSimilarity(v,W[i])>max_similarity and idx2word[i]!="," and idx2word[i]!="." and idx2word[i]!="!" and idx2word[i]!='?' and idx2word[i]!=w1 and idx2word[i]!=w2 and idx2word[i]!=w3:
                    max_similar=i
                    max_similarity=cosineSimilarity(v,W[i])
            if idx2word[max_similar]==w4:
                correct=correct+1
            print(idx2word[max_similar])


print(correct/total)        


# In[32]:


print(total)


# In[ ]:




