# each module is a file name, if you want to import class from that file, u should use this from module import class-name
from posixpath import dirname
from PLSI import PLSI
import re
import string
import numpy as np
from gensim.corpora.dictionary import Dictionary



model = PLSI(
    "model1","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='random1'
)
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




model_base = PLSI(
    "model_base","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='random1',
    bonus2=False
)

model_random = PLSI(
    "model_base","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='random',
    bonus2=False
)

model_dirchlet = PLSI(
    "model_base","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='dirchlet',
    bonus2=False
)

print(
    model_base.document_probability(model_base.dt,model_base.tw,model_base.document_words,model_base.documentNum,model_base.corpus)
)
print(
    model_random.document_probability(model_random.dt,model_random.tw,model_random.document_words,model_random.documentNum,model_random.corpus)
)
print(
    model_dirchlet.document_probability(model_dirchlet.dt,model_dirchlet.tw,model_dirchlet.document_words,model_dirchlet.documentNum,model_dirchlet.corpus)
)


model_base2 = PLSI(
    "model_base","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='random1',
    bonus2=True
)

model_random2 = PLSI(
    "model_base","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='random',
    bonus2=True
)

model_dirchlet2 = PLSI(
    "model_base","corpus","txt",toload= False, 
    stopWordList=['a','the'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=10,
    randomInit='dirchlet',
    bonus2=True
)

print(
    model_base2.document_probability(
        model_base2.dt,
        model_base2.tw,
        model_base2.document_words,
        model_base2.documentNum,
        model_base2.corpus)
)
print(
    model_random2.document_probability(
        model_random2.dt,
        model_random2.tw,
        model_random2.document_words,
        model_random2.documentNum,
        model_random2.corpus)
)
print(
    model_dirchlet2.document_probability(
        model_dirchlet2.dt,
        model_dirchlet2.tw,
        model_dirchlet2.document_words,
        model_dirchlet2.documentNum,
        model_dirchlet2.corpus)
)
