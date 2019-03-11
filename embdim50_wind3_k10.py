#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import pickle

from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
import nltk
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F1
import torch.nn.functional as F
import random

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords


# In[4]:


k=10
window_size =3


# In[5]:


cachedStopWords = stopwords.words("english")

#print(type(reuters))

documents = reuters.fileids()
#print(str(len(documents)) + " documents");

train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
#print(str(len(train_docs)) + " total train documents");


# In[21]:


test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents));
#print(str(len(test_docs)) + " total test documents");


# In[6]:


count=0
count1=0
tokenized_corpus=[]
for id in train_docs:
    for sentence in reuters.sents(id):
        for i in range(len(sentence)):
            sentence[i]=sentence[i].lower()
            if sentence[i].isnumeric():
                sentence[i]="num"
        tokenized_corpus.append(sentence[:-1])
          

    count=count+1

#corpus is now tokenized

vocabulary = []
tokens=[]
for sentence in tokenized_corpus:
    
    
    for token in sentence:
        tokens.append(token)
        #if token.isnumeric():
            #tokens.append("num")
        if token not in vocabulary:
            vocabulary.append(token)

#print(vocabulary)

#print(vocabulary[0:10])
print(len(vocabulary))

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)


# In[7]:




pickle_out = open("pickles/vocab.pickle","wb")
pickle.dump(vocabulary, pickle_out)
pickle_out.close()


# In[6]:



idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
print (len(idx_pairs),":idx_pairs")
idx_pairs1=idx_pairs[int(len(idx_pairs)/10):,:]


# In[22]:


pickle_out = open("pickles/word2id.pickle","wb")
pickle.dump(word2idx, pickle_out)
pickle_out.close()

pickle_out = open("pickles/id2word.pickle","wb")
pickle.dump(idx2word, pickle_out)
pickle_out.close()


# In[7]:


def get_input_layer(word_idx):
    x = np.zeros(vocabulary_size)
    x[word_idx] = 1.0
    return x


# In[8]:


from numba import jit
from numba import vectorize
#@vectorize(['float32(float32, float32, float32,float32,float32)',
#            'float64(float64, float64, float64,float64,float64)'],
#           target='roc')
@jit()
def train(i,li,nli,W1,W2):
    start_time=time.time()
    for j in li:
        dot=np.dot(W2[j],W1[i])
        #e=math.exp(-dot)
        if(dot)>=9 :
            W1[i]=W1[i]
            W2[j]=W2[j]

        elif (dot)>=-8 and (dot)<9:
            e=math.exp(-dot)
            W1[i]=W1[i]+learning_rate*W2[j]*(e)/(1+e)
            W2[j]=W2[j]+learning_rate*W1[i]*(e)/(1+e)
        else:
            W1[i]=W1[i]+learning_rate*W2[j]
            W2[j]=W2[j]+learning_rate*W1[i]                                                             


                                                                               
                                                                               
    for j in nli:
        dot=np.dot(W2[j],W1[i])                                                                   
                                                                           
        if(dot)<=-9 :
            W1[i]=W1[i]
            W2[j]=W2[j]  #switched
              
        elif(dot<8 and dot>-9):
            e=math.exp(dot)
            W1[i]=W1[i]-learning_rate*W2[j]* e/(1+e)
            W2[j]=W2[j]-learning_rate*W1[i]*e/(1+e)
        else:

            W1[i]=W1[i]-learning_rate*W2[j]
            W2[j]=W2[j]-learning_rate*W1[i]  
    #print("time in one train:",time.time()-start_time)
    return(W1,W2)


# In[8]:


embedding_dims = 50
import math
#W1=np.random.randint(2,size=(embedding_dims,vocabulary_size))
#W2=np.random.randint(2,size=(vocabulary_size,embedding_dims))
import time
import tqdm

W1=np.random.rand(embedding_dims,vocabulary_size) 
W2=np.random.rand(vocabulary_size,embedding_dims) 
num_epochs = 10
learning_rate = 0.001
W1=W1.T
for epo in range(num_epochs):
    m=0
    ste=time.time()
    print("epoch:",epo)
    loss_val = 0
    
    #for data in vocabulary:
    li=[]
    nli=[]
       
    start_time = time.time()
    for input1,target in idx_pairs:


        li=[target]
        nli=list()
        i=input1

        c=0
        st=time.time()
        while(c<k):
            r=random.randint(0,len(tokens)-1)
            if word2idx[tokens[r]] not in nli:
                nli.append(word2idx[tokens[r]])
                c+=1
            

       
        W1,W2=train(i,li,nli,W1,W2)

        W=W1


    np.savetxt('data/w_embdim50_wind3_k10.txt',W)
    print("time in this epoch:",time.time()-ste)
    
    
        


# In[21]:


import numpy as np
p=np.loadtxt('data/w_embdim50_wind3_k10.txt')

pickle_out = open("pickles/embedding.pickle","wb")
pickle.dump(p, pickle_out)
pickle_out.close()
str1=""
for i in range(len(vocabulary)):
    str1=str1+vocabulary[i]

    str1=str1+" "+str(p[i])[1:-1]+"\n"

text_file = open("data/w_embdim50_wind3_k10_with_words.txt", "w+")

text_file.write(str1)


# In[9]:


#loss calculation for test
test_corpus=[]
for id in test_docs:
    for sentence in reuters.sents(id):
        for i in range(len(sentence)):
            sentence[i]=sentence[i].lower()
            if sentence[i].isnumeric():
                sentence[i]="num"
        test_corpus.append(sentence[:-1])


# In[10]:


test_vocabulary = []
test_tokens=[]
for sentence in test_corpus:
    
    
    for token in sentence:
        test_tokens.append(token)

        if token not in test_vocabulary:
            test_vocabulary.append(token)

test_word2idx = {w: idx for (idx, w) in enumerate(test_vocabulary)}
test_idx2word = {idx: w for (idx, w) in enumerate(test_vocabulary)}


# In[11]:



test_idx_pairs = []
# for each sentence
for sentence in test_corpus:
    indices = [test_word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            test_idx_pairs.append((indices[center_word_pos], context_word_idx))

test_idx_pairs = np.array(test_idx_pairs) # it will be useful to have this as numpy array


# In[12]:


loss=0
for input1,target in test_idx_pairs:
    c=0
    nli=[]
    st=time.time()
    while(c<k):
        r=random.randint(0,len(tokens)-1)
            #if(i,word2idx[tokens[r]]) not in idx_pairs :
        if word2idx[tokens[r]] not in nli:
            nli.append(word2idx[tokens[r]])
            c+=1
    if(test_idx2word[input1] not in vocabulary):
        v1=np.random.rand(embedding_dims,1)
    else:
        v1=W1[word2idx[test_idx2word[input1]]].reshape(embedding_dims,1)
    if(test_idx2word[target] not in vocabulary):
        v2=np.random.rand(embedding_dims,1)
    else:
        v2=W2[word2idx[test_idx2word[target]]].reshape(embedding_dims,1)
    #print (v1.shape,v2.shape)
    #dot=np.dot(v1.reshape(embedding_dims,1).T,v2.reshape(embedding_dims,1))
    dot=np.dot(v1.T,v2)
    loss=-math.log(1/(1+math.exp(-dot)))
    
    for i in range(k):
        dot=np.dot(v1.T,W2[nli[i]])
        loss=loss-math.log(1/(1+math.exp(dot)))
    loss=loss/(len(test_idx_pairs))


# In[13]:


print(loss)


# In[14]:


def cosineSimilarity(v1,v2):
    return np.dot(v1,v2)/math.sqrt(np.dot(v1,v1)*np.dot(v2,v2))


# In[16]:


import pandas as pd
df = pd.read_csv("data/SimLex-999.txt",sep="\t")

df1 = df[['word1', 'word2','SimLex999']]

print (df1.head())


# In[17]:


import numpy as np
n,m=df1.shape
df1=np.array(df1)
cos_score=np.zeros(n)
sim_score=np.zeros(n)

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


# In[19]:


from scipy import stats
print(stats.spearmanr(cos_score, sim_score))


# In[ ]:




