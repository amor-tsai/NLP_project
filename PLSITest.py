# each module is a file name, if you want to import class from that file, u should use this from module import class-name
from PLSI import PLSI
import re
import string
import numpy as np
from gensim.corpora.dictionary import Dictionary


# model = BigramModel("model1","corpus","tx",stopWordList=["the","a"],singlesen=False,smooth=0.2)
model = PLSI(
    "model1","corpus","tx",toload= False, 
    stopWordList=['the','a','is','are','to','was','as','from','and','can','been','on','an','of','by','also','or','such','which','for','with'], 
    ignoreCase=True, stem="snowball", topicCount=3, iterations=20,
    randomInit='random1'
    )
# model.save()

print(
    model.getDocumentTopic(1)
)

# print(
#     model.getAllDocumentTopic()
# )

print(
    model.getTopicWordVector(0,3)
)

print(
    model.getTopicWordVectorAll(3)
)

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


# tokens = ['hello','how']
# my_vocab = Dictionary([tokens])

# print(my_vocab.token2id['how'])
# print(my_vocab[0])