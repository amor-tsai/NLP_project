'''
CS7322 NLP
author: Amor Tsai
Programming Homework 2: (Topic Model)
Total completion: base case + bonus1 + bonus2 + bonus3
For bonus3, please take a look at the Program2 bonus3 report.html

NOTICE:
1. It needs to be run in python3.10 because I use Counter.total(), which is a new feature in python3.10
2. First iteration is used for initializing document-topic vector and topic-word vector. That is, if the input iteraions is 3, it runs only twice.
3. I bet "dirchlet" should be "dirichlet", but I would like to follow up the document requirement and use "dirchlet"

quote :
    For each document q
    For each topic t
    Modify dq in the following way
    For each topic t’ that is not t, set dq[t’] = dq[t’] /2
    After that set dq[t] = 1 - ∑ dq[t’] (where t’ ≠ t)


'''
import os
from os.path import join
import numpy as np
from nltk.tokenize import word_tokenize
import re
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import copy
nltk.download('punkt',quiet=True)

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
    def __init__(self, name: str = "default", dirName: str = ".", ext: str = "*", toload: bool = False, stopWordList: list = [], ignoreCase: bool = True, stem: str="", topicCount:int = 10, iterations:int=20, randomInit:str = "", bonus2:bool = True):
        self.__moduleName = name
        self.__moduleExt = ".plsi"
        self.topicCount = topicCount
        self.iterations = iterations
        self.randomInit = randomInit
        self.bonus2_enable = bonus2 # bonus2 is enable by default
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
                self.dt,self.tw,self.topicCount,self.documentNum,self.documentNameList,self.corpus,self.document_words,self.wordsCount = self.__model
                return
        
        # preprocessing the corpus
        self.preprocess_corpus(dirName,ext,stopWordList,ignoreCase)
        # calculate document topic vector and topic word vector in iterations
        self.calculateDTandTW()
        # save document topic vector and topic word vector in the model
        self.__model = [self.dt,self.tw,self.topicCount,self.documentNum,self.documentNameList,self.corpus,self.document_words,self.wordsCount]
        

    '''
    Calculate document topic vector and topic word vector in iterations
    '''
    def calculateDTandTW(self):
        '''
        preResults defined here is by purpose to compare if there is a different result
        '''
        if 0 == self.iterations:
            raise Exception("iterations could not be zero!")
        # preResults = [self.dt,self.tw]
        # iterations deduce one because initializing document-topic vector and topic-word vector is counted once.
        for iteration in range(self.iterations):
            # initialize document-topic vector and topic-word vector at the first step
            if 0 == iteration:
                # initialize document-topic vector
                self.dt = self.document_topic_vector()
                # initialize topic-word vector
                self.tw = self.topic_word_vector()
                # use document-topic vector and topic-word vector to calculate log sum probabilities
                best_pros = self.document_probability(self.dt,self.tw,self.document_words,self.documentNum,self.corpus)
                continue

            dt = self.dt
            tw = self.tw
 
            # calculate the intermediate vector
            vectors,document_wordsIndex = self.calculate_intermediate_vector(self.topicCount,self.documentNum,self.wordsCount,self.document_words,self.corpus,dt,tw)
            # calculate document-topic vector
            newdt = self.calculate_document_topic_vector(vectors,self.documentNum,dt)
            # calculate topic-word vector
            newtw = self.calculate_topic_word_vector(vectors,document_wordsIndex,self.topicCount,self.documentNum,self.wordsCount,tw)

            # evaluate the model and store the best probability, best dt and tw
            tmp_pros = self.document_probability(newdt,newtw,self.document_words,self.documentNum,self.corpus)

            # for bonus 2, here assumes that local optimal is not improved after 5 iterations.
            if iteration >= 5 and self.bonus2_enable:
                # it's weird here if don't use copy, the value of newdt will change
                newdt2 = self.re_calculate_document_vector_bonus2(newdt)
                tmp_pros2 = self.document_probability(newdt2,newtw,self.document_words,self.documentNum,self.corpus)
                # if better solution generates, then applies this newdt2 and newtw
                if tmp_pros2 > tmp_pros and tmp_pros2 > best_pros:
                    print('{} good result issues'.format(iteration))
                    best_pros = tmp_pros2
                    self.dt = newdt2
                    self.tw = newtw
                # else:
                    # print('{} not good result'.format(iteration))
            else:
                if tmp_pros > best_pros:
                    best_pros = tmp_pros
                    self.dt = newdt
                    self.tw = newtw
            

    '''
    calculate the averge sum of log probability in all documents
    '''
    def document_probability(self,dt,tw,document_words,documentNum,corpus):
        prosSum = [0] * documentNum
        for documentIndex in range(documentNum):
            for ch in document_words[documentIndex]:
                wordIndex = corpus.index(ch)
                prosSum[documentIndex] += np.log10(np.dot(tw[:,wordIndex],dt[documentIndex]))
        return np.round(np.mean(prosSum),4)
    
    '''
    get performance score of single document
    '''
    def single_document_evaluation(self,documentIndex,documentName):
        if documentIndex < 0 or documentIndex > len(self.documentNameList):
            documentIndex = self.documentNameList.index(documentName)
        prosSum = 0
        for ch in self.document_words[documentIndex]:
            wordIndex = self.corpus.index(ch)
            prosSum += np.log10(np.dot(self.tw[:,wordIndex],self.dt[documentIndex]))
        return np.round(prosSum,4)
    
    '''
    get performance score of multiple documents
    '''
    def documents_evaluation(self):
        return self.document_probability(
            self.dt,self.tw,self.document_words,self.documentNum,self.corpus
        )

    '''
    based on bonus2, re-calculate document-topic vector
    '''
    def re_calculate_document_vector_bonus2(self,dt):
        tmp_dt = copy.deepcopy(dt)
        topicCount = len(tmp_dt[0])
        for i in range(len(tmp_dt)):
            for j in range(topicCount):
                '''
                For each topic t' that is not t, set dq[t'] = dq[t'] /2
                After that set dq[t] = 1 - ∑ dq[t'] (where t' ≠ t)
                '''
                tmpSum = 0
                for k in range(topicCount):
                    if j != k:
                        tmp_dt[i][k] /= 2
                        tmpSum += tmp_dt[i][k]
                tmp_dt[i][j] = 1 - tmpSum
        tmp_dt = np.round(tmp_dt,4)
        return tmp_dt
    
    '''
    calculate topic-word vector by intermediate vectors
    '''
    def calculate_topic_word_vector(self,vectors,document_wordsIndex,topicCount,documentNum,wordsCount,tw):
        for topicIndex in range(topicCount):
            for wordIndex in range(wordsCount):
                numerator = 0
                denominator = 0
                for documentIndex in range(documentNum):
                    numerator += vectors[documentIndex][topicIndex][document_wordsIndex[documentIndex][wordIndex]].sum()
                    denominator += vectors[documentIndex][topicIndex].sum()
                tw[topicIndex][wordIndex] = numerator/denominator
        tw = np.round(tw,4)
        return tw

    '''
    calculate document-topic vector by intermediate vector tmp
    '''
    def calculate_document_topic_vector(self,vectors,documentNum,dt):
        for documentId in range(documentNum):
            dt[documentId] = np.sum(vectors[documentId],axis=1)/np.sum(vectors[documentId])
        dt = np.round(dt,4)
        return dt

    '''
    For each word in each document, store a vector (with dimensionality topicount) to represent the probability of that word comes from topic 1..topicCount.
    '''
    def calculate_intermediate_vector(self,topicCount,documentNum,wordsCount,document_words,corpus,dt,tw):
        vectors = [[] for _ in range(documentNum)]
        document_wordsIndex = [[] for _ in range(documentNum)]
        for documentId in range(documentNum):
            tt = np.array([],dtype=float).reshape(topicCount,0)
            tmp_wordsIndex = [[] for _ in range(wordsCount)]
            for i,ch in enumerate(document_words[documentId]):
                wordIndex = corpus.index(ch)
                tmp = (tw[:,wordIndex]*dt[documentId])/np.dot(tw[:,wordIndex],dt[documentId])
                tt = np.column_stack([tt,tmp.T])
                tmp_wordsIndex[wordIndex].append(i)
            
            document_wordsIndex[documentId] = tmp_wordsIndex
            vectors[documentId] = np.round(tt,4)
        return vectors,document_wordsIndex

    '''
    initialize document topic vector
    '''
    def document_topic_vector(self):
        if self.randomInit == "random":
            # Divide the elements of each row by their row-summations
            v = np.random.random((self.documentNum,self.topicCount))
            dt = v/v.sum(axis=1,keepdims=1)
        elif self.randomInit == "dirchlet":
            '''
            If the value of this variable is set to “dirchlet”, then each document-topic vector is generated from the Dirichlet random function, 
            with alpha = 0.2 for each dimension. 
            '''
            dt = np.random.dirichlet(alpha=[0.2]*self.topicCount,size=self.documentNum)
        else:
            dt = np.zeros((self.documentNum,self.topicCount),dtype=float)
            for i,_ in enumerate(self.document_words):
                for j,_ in enumerate(self.document_words[i]):
                    dt[i][(i+j)%self.topicCount] += 1
                dt[i] /= sum(dt[i])
        dt = np.round(dt,4)
        return dt

    '''
    initialize topic word vector 
    '''    
    def topic_word_vector(self):
        # self.tw = [[] for _ in range(self.topicCount)]
        tw = np.zeros((self.topicCount,self.wordsCount),dtype=float)
        topic_word = [Counter() for _ in range(self.topicCount)]
        for i,_ in enumerate(self.document_words):
            for j,ch in enumerate(self.document_words[i]):
                topic_word[(i+j)%self.topicCount][ch] += 1
        # length of words
        length = len(self.corpus)

        if self.randomInit == "random":
            for i in range(len(tw)):
                v = np.random.random(length)
                tw[i] = v/v.sum(axis=0,keepdims=1)
            
        elif self.randomInit == "dirchlet":
            '''
            For each topic-word vector, it should be initialized such that every word has the same probability.
            '''
            for i in range(len(tw)):
                tw[i] = np.ones((length,),dtype=float)/length
        else:
            '''
            base case
            '''
            for i,words in enumerate(topic_word):
                tmpSum = words.total()
                for j in range(self.wordsCount):
                    tw[i][j] = topic_word[i][self.corpus[j]]/tmpSum
        tw = np.round(tw,4)
        return tw
        
    '''
    pre-process the corpus
    '''
    def preprocess_corpus(self, dirName:str = ".", ext: str = "*", stopWordList: list = [],ignoreCase: bool = True):
        self.document_words = []
        self.documentNameList = []
        self.corpus = []# distinctive words combined into a corpus
        for _,file in  enumerate(os.listdir(dirName)):
            if ext != "*" and False == file.endswith(ext):
                continue
            # add document name in list, the index is the document id
            self.documentNameList.append(file)           
            word_tokens = word_tokenize(open(join(dirName,file),"r").read())
            arr = []
            for ch in word_tokens:
                #remove stop words and non-alphanumeric words
                if (len(stopWordList) > 0 and ch in stopWordList) or None == re.match(".*[\w]+.*",ch):
                    continue
                # transform words into lower case
                if ignoreCase:
                    ch = ch.lower()
                # use stemmer to stem
                if None != self.stemmer:
                    ch = self.stemmer.stem(ch)
                arr.append(ch)
                if ch not in self.corpus:
                    self.corpus.append(ch)
            # add each document of word tokens
            self.document_words.append(arr)

        # total number of documents
        self.documentNum = len(self.documentNameList)
        # total number of words
        self.wordsCount = len(self.corpus)



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
        res = []
        for word,proba in zip(self.corpus,self.tw[topicNum]):
            res.append((word,proba))
        res = sorted(res,key=lambda item: item[1],reverse=True)
        if topCount > 0:
            res = res[:topCount]
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
          One line for each word's probability
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


