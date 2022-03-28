# each module is a file name, if you want to import class from that file, u should use this from module import class-name
from posixpath import dirname
from PLSI import PLSI
import re
import string
import numpy as np
from gensim.corpora.dictionary import Dictionary
import os
import tracemalloc
import linecache
import time

params = {
    'name' : "model_base",
    'dirName' : "corpus",
    'ext' : "txt",
    'toload' : False, 
    'stopWordList' : ['a','the'], 
    'ignoreCase' : True, 
    'stem' : "snowball", 
    'topicCount' : 8, 
    'iterations' : 1,
    'randomInit' : '',
    'bonus2' : False
}

# model = PLSI(**params)
# print(
#     model.document_probability(
#         model.dt,model.tw,model.document_words,model.documentNum,model.corpus
#     )
# )


# model = PLSI(
#     dirName='corpus'
# )

# model.save()

# print(
#     model.getDocumentTopic(1)
# )

# print(
#     model.getAllDocumentTopic()
# )

# print(
#     model.getTopicWordVector(0,3)
# )

# print(
#     model.getTopicWordVectorAll(0)
# )

# print(
#     model.ExtendedPrint('b')
# )

# model.save()

# print(model.getAll(3))
# print(model.getProbList('^', sortMethod=2))
# print(model.getProbList2('$', sortMethod=1))
# n = 2
# missingSum = 12
# q,r = divmod(missingSum,n)
# print(q)
# print(r)
# print(
#     [q+1]*r + [q]*(n-r)
# )

# if None == re.match(".*[\w]+.*","!"):
#     print(1)
# else:
#     print(2)


# dt = np.array(
#     [0.4, 0.4, 0.2,0.2857, 0.4285, 0.2857,0.3333, 0.3333, 0.3333]
# ).reshape(3,3)
# tw = np.array([0.5, 0.1667, 0, 0.333,0.2857, 0.1429, 0.1429, 0.4286,0, 0.2, 0.4, 0.4]).reshape(3,4)
# a = (tw[:,0]*dt[0])/np.dot(tw[:,0],dt[0])
# b = (tw[:,1]*dt[0])/np.dot(tw[:,1],dt[0])
# c = (tw[:,2]*dt[0])/np.dot(tw[:,2],dt[0])
# d = (tw[:,3]*dt[0])/np.dot(tw[:,3],dt[0])
# # print(a)
# # print(b)
# # print(c)
# # print(d)
# res = np.column_stack((a.T,b.T,c.T,d.T))
# print(res)


# a = np.array([1,2,3,4])
# indices = [0,1] 
# print(a[indices].sum())



# A = np.array(
#     [[1,1,1],
#     [2,2,2],
#     [5,5,5]
#     ]
# )

# B = np.array(
#     [[0,1,1],
#     [1,1,1],
#     [1,2,1]]
# )

# C = np.einsum('aj,bk->aj',A,B)
# print(C)




