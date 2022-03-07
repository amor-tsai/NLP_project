'''
CS7322 NLP
author: Amor Tsai
Programming Homework 2: (Topic Model)
Total completion: base + bonus1 and bonus2

NOTICE:
It needs to be run in python3.10 because I use Counter.total(), which is a new feature in python3.10


'''
from numpy import dtype
import nltk
import os
import string
from os.path import isfile, join
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, TweetTokenizer
from nltk.util import ngrams
from nltk.util import pad_sequence
import re
import pickle
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import copy

class PLSI:
#  name: name of the model. The name is used as the filename to store the model
#  dirName: the directory that store the corpus. If dname starts with a “/”, then dname is an absolute path. Otherwise the path is relative starting from the current directory. (Default is the current directory). Any subdirectories of it is ignored.
#  ext: the extension of all files that is consider part of the corpus. Only files that have the specified extension will be read and processed. Default is “*’, which mean all files in the directory will be considered.
#  ToLoad: if true, the bigram model should be loaded from a file named <filename>.plsi. (See the save() function for details), and all other parameters are ignored).
#  stopWordList: a list of words that will be removed from the corpus before calculating the bigram probability. Default is empty
#  ignoreCase: if true, all words are transformed to lowercase. Otherwise, the original case of the word is kept. (i.e. “Apple” and “apple” will be treated as two words).
#  stem: whether to use a Stemmer or not. For this project, if stem = “snowball”, then use the snowball stemmer in NLTK. Any other values are ignored.
#  topicCount: number of topics of the model. Default is 10.
#  Iterations: number of iterations that the algorithm will take. (See below)
    def __init__(self, name: str = "default", dirName: str = ".", ext: str = "*", toload: bool = False, stopWordList: list = [], ignoreCase: bool = True, stem: str="", topicCount:int = 10, iterations:int=20, randomInit:str = ""):
        self.__moduleName = name
        self.__moduleExt = ".plsi"
        self.topicCount = topicCount
        self.iterations = iterations
        self.randomInit = randomInit
        if "snowball" == stem:
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = None
        # toload == True, should load module from file
        if toload:
            fileName = self.__moduleName + self.__moduleExt
            if os.path.isfile(fileName):
                # load the model from file
                self.__model = pickle.load(open(fileName,'rb'))
                self.dt,self.tw,self.topicCount,self.documentNum,self.documentNameList,self.corpus = self.__model
                return
        
        # preprocessing the corpus
        self.getCorpus(dirName,ext,stopWordList,ignoreCase)

        # initialize document-topic vector
        self.document_topic_vector()
        # initialize topic-word vector
        self.topic_word_vector()
        # calculate document topic vector and topic word vector in iterations
        self.calculateDTandTW()
        # save document topic vector and topic word vector in the model
        self.__model = [self.dt,self.tw,self.topicCount,self.documentNum,self.documentNameList,self.corpus]
        

    '''
    Calculate document topic vector and topic word vector in iterations
    '''
    def calculateDTandTW(self):
        '''
        preResults defined here is by purpose to compare if there is a different result
        '''
        preResults = [self.dt,self.tw]
        for iteration in range(self.iterations):
            dt = copy.deepcopy(self.dt)
            tw = copy.deepcopy(self.tw)
            # calculate the intermediate vector
            tmp = self.calculate_intermediate_vector(self.topicCount,self.documentNum,dt,tw,self.corpus)
            # calculate document-topic vector
            newdt = self.calculate_document_topic_vector(tmp,self.topicCount,self.documentNum,dt)
            # calculate topic-word vector
            newtw = self.calculate_topic_word_vector(tmp,self.topicCount,self.documentNum,tw)

            # for bonus 2, here assumes that local optimal is not improved after 5 iterations.
            # if things go to another way, then this code should be applied
            # after k iteration, the result can't be optimized
            # if np.array_equal(self.dt,preResults[0]):
            #     print(
            #         iteration
            #     )
            if iteration >= 5:
                newdt2 = self.re_calculate_document_vector_bonus2(self.dt)
                if not np.array_equal(newdt2,newdt):
                    newdt = newdt2

            # store the results in each iteration
            # preResults = [newdt,newtw]
            self.dt = newdt
            self.tw = newtw
            # print(self.dt)
        
    '''
    based on bonus2, re-calculate document-topic vector
    '''
    def re_calculate_document_vector_bonus2(self,dt):
        if dt.size == 0:
            raise Exception("document topic vector can't be empty")
        topicCount = len(dt[0])
        for i in range(len(dt)):
            for j in range(topicCount):
                p = (1-dt[i][j])/2
                dt[i][j] += p
                for k in range(topicCount):
                    if k != j:
                        dt[i][k] -= p/(topicCount-1)
        return dt
    
    '''
    calculate topic-word vector by intermediate vector tmp
    '''
    def calculate_topic_word_vector(self,tmp,topicCount,documentNum,tw):
        for topicIndex in range(topicCount):
            for key in tw[topicIndex]:
                numerator = 0.0
                denominator = 0.0
                for documentId in range(documentNum):
                    numerator += tmp[documentId][topicIndex][key]
                    denominator += tmp[documentId][topicIndex].total()
                tw[topicIndex][key] = numerator/denominator
        return tw

    '''
    calculate document-topic vector by intermediate vector tmp
    '''
    def calculate_document_topic_vector(self,tmp,topicCount,documentNum,dt):
        for documentId in range(documentNum):
            tSum = sum([tmp[documentId][x].total() for x in range(topicCount)])
            for topicIndex in range(topicCount):
                dt[documentId][topicIndex] = tmp[documentId][topicIndex].total()/tSum
        return dt

    '''
    For each word in each document, store a vector (with dimensionality topicount) to represent the probability of that word comes from topic 1..topicCount.
    '''
    def calculate_intermediate_vector(self,topicCount,documentNum,dt,tw,corpus):
        tmp = [[Counter() for _ in range(topicCount)] for _ in range(documentNum)]
        for documentId in range(documentNum):
            for topicIndex in range(topicCount): #topicIndex
                for word in corpus[documentId]:
                    tmp[documentId][topicIndex][word] = tw[topicIndex][word] * dt[documentId][topicIndex]

            for word in corpus[documentId]:
                keySum = sum([tmp[documentId][j][word] for j in range(topicCount)])
                for j in range(topicCount):
                    tmp[documentId][j][word] /= keySum
        return tmp


    # initialize document topic vector
    def document_topic_vector(self):
        if self.randomInit == "random":
            # Divide the elements of each row by their row-summations
            v = np.random.random((self.documentNum,self.topicCount))
            self.dt = v/v.sum(axis=1,keepdims=1)
        elif self.randomInit == "dirchlet":
            '''
            If the value of this variable is set to “dirchlet”, then each document-topic vector is generated from the Dirichlet random function, 
            with alpha = 0.2 for each dimension. 
            '''
            self.dt = np.random.dirichlet(alpha=[0.2]*self.topicCount,size=self.documentNum)
        else:
            self.dt = np.ones((self.documentNum,self.topicCount),dtype=float)/self.topicCount

    '''
    generate topic word vector 
    '''    
    def topic_word_vector(self): 
        if self.randomInit == "random":
            for i in range(len(self.tw)):
                total = len(self.tw[i])
                v = np.random.random(total)
                pro_list = v/v.sum(axis=0,keepdims=1)
                for j,key in enumerate(self.tw[i].keys()):
                    self.tw[i][key] = pro_list[j]
            
        elif self.randomInit == "dirchlet":
            '''
            For each topic-word vector, it should be initialized such that every word has the same probability.
            '''
            for i in range(len(self.tw)):
                for key in self.tw[i].keys():
                    self.tw[i][key] = 1 / self.tw[i].total()
        else:
            for i in range(len(self.tw)):
                total = self.tw[i].total()
                for key in self.tw[i].keys():
                    self.tw[i][key] = self.tw[i][key] / total

        
    '''
    self.documentNameList has document name and document id(index)
    return a list incorporating sentences
    '''
    def getCorpus(self, dirName:str = ".", ext: str = "*", stopWordList: list = [],ignoreCase: bool = True):
        self.corpus = []
        self.documentNameList = []
        self.tw = [Counter() for _ in range(self.topicCount)]
        for file in os.listdir(dirName):
            if ext != "*" and False == file.endswith(ext):
                continue
            # add document name in list, the index is the document id
            self.documentNameList.append(file)            
            if ignoreCase:
                word_tokens = word_tokenize(open(join(dirName,file),"r").read().lower())
            else:
                word_tokens = word_tokenize(open(join(dirName,file),"r").read())
            arr = []
            for ch in word_tokens:
                #remove stop words and non-alphanumeric words
                if (len(stopWordList) > 0 and ch in stopWordList) or None == re.match(".*[\w]+.*",ch):
                    continue
                if None != self.stemmer:
                    # use stemmer to stem
                    ch = self.stemmer.stem(ch)
                arr.append(ch)
            # add each document of word tokens
            self.corpus.append(arr)
            for i,string in enumerate(arr):
                # add word to each topic
                self.tw[i%self.topicCount][string] += 1
        # total number of documents
        self.documentNum = len(self.documentNameList)
        # print(self.corpus)

    '''
    Return the document-topic vector for a certain document. If docNum is NOT one of the document number you assigned, then use docName to find the document with the name.
    The return should be a list, where the probabilities are ordered from topic 1, topic 2, …etc.
    '''
    def getDocumentTopic(self,docNum:int = -1, docName:str = ""):
        if docNum < 0 or docNum >= len(self.dt):
           docNum = self.documentNameList.index(docName)
        return self.dt[docNum]

    '''
    Return the document-topic vectors for all documents.
    The return should be a list of tuples. Each tuple is made up of (x, y), where x is the name of the document and y is the document-topic vector 
    of that document (represented as a list).
    '''
    def getAllDocumentTopic(self):
        res = []
        for i in range(len(self.dt)):
            res.append(
                tuple([self.documentNameList[i],self.dt[i]])
            )
        return res
    
    '''
    1. Return the topic-word vector for topic with topicNum
    2. topCount specify how many words are returned, based on the probability of the word in that 
    topic (only select the highest ones to return). topCount <= 0, then return all words.
    3. The return should be a dictionary, where each entry is a word and its probabiltiies.
    '''
    def getTopicWordVector(self,topicNum, topCount = 10):
        res = sorted(self.tw[topicNum].items(), key=lambda x:x[1],reverse=True)
        if topCount > 0:
            res = res[0:topCount]
        return dict(res)

    '''
    1.Print all topic-word vectors. topCount has the same definition as above.
    2.Your return should be a list of dictionaries, with the first dictionary is of topic 1, the second one is of topic 2 etc.
    '''
    def getTopicWordVectorAll(self,topCount = 10):
        res = []
        for i in range(len(self.tw)):
            res.append(self.getTopicWordVector(i,topCount))
        return res
    
    '''
    Store the model in a human readable format as follows:
    1. All the output should be printed in the subdirectory named: <model name><suffix>. Notice that suffix can be any string, does not have to start with “.”.
    2. The document-topic should be printed in a file named “document-topics. Each line of the file should be formatted as follows
          The first word is the name of the document
          Then print one space character
          Then each dimension of the topic-document vector is printed (in order of topic 1, topic 2 etc.) There should be one space between each number.
    3. Each topic-word vector should be printed in a separate file named “topic_”<topic number>
          One line for each word’s probability
          Each line of the file should be formatted as <word> <probability of the word>
          The output should be sorted by decreasing order of probability
    '''
    def ExtendedPrint(self,dnamSuffix= ""):
        # create subdirectory
        path = os.path.join(".",self.__moduleName+dnamSuffix)
        if not os.path.isdir(path):
            os.mkdir(path)
        # write each document-topic vector to file
        res = self.getAllDocumentTopic()
        with open(os.path.join(path,'document-topics'),"w") as f:
            for i in range(len(res)):
                f.write('%s %s\n' % (res[i][0],' '.join(map(str,res[i][1]))))
            
        # write each topic-word vector to file
        res = self.getTopicWordVectorAll(0)
        for i in range(len(res)):
            with open(os.path.join(path,"topic_{}".format(i)),"w") as f:
                for k,v in res[i].items():
                    f.write('%s %s\n' % (k,v))

    '''
    Save the model into a single file named <model name>.plsi
    You can choose whatever format you want, as long as you can read it back,
    The format does not have to be human-readable.
    It is ok if you want to reuse the format of ExtendedPrint(), but everything has to be in one file.
    '''
    def save(self):
        if self.__model:
            pickle.dump(self.__model,open(self.__moduleName + ".plsi",'wb'))


