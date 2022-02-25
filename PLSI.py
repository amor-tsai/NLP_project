# CS7322 NLP
# author: Amor Tsai
#Programming Homework 2: (Topic Model) 
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
import functools

class PLSI:
    def __init__(self, name: str = "default", dirName: str = ".", ext: str = "*", toload: bool = False, stopWordList: list = [], ignoreCase: bool = True, stem: str="", topicCount:int = 10, iterations:int=20):
        self.__moduleName = name
        self.__moduleExt = ".plsi"
        # toload == True, should load module from file
        if toload:
            fileName = self.__moduleName + self.__moduleExt
            if os.path.isfile(fileName):
                # load the model from file
                self.bigrams = pickle.load(open(fileName,'rb'))
                return
        
        if smooth < 0 or smooth > 1:
            smooth = 0
        
        sents = self.getSentences(dirName,ext,singlesen)
        corpus = []
        for sen in sents:
            corpus.append("^")
            for ch in word_tokenize(sen):
                # remove stop words and non-alphanumeric words after tokenization
                if (len(stopWordList) > 0 and ch in stopWordList) or None == re.match(".*[\w]+.*",ch):
                    continue
                # always substitute the word with the first one in otherWordList since they are the same word
                corpus.append(ch)
            corpus.append("$")
 
        unigram = ngrams(corpus,1)
        bigram = ngrams(corpus,2)
        # print(bigram)
        freq_ui = nltk.FreqDist(unigram)
        freq_bi = nltk.FreqDist(bigram)
        freq_single = {}
        for k,v in freq_ui.items():
            freq_single[k[0]] = v
        
        self.bigrams = {}
        lengthOfTypesFromBigram = len(freq_bi.items())
        for k,v in freq_bi.items():
            # need to remove ($,^) from bigram
            if k == ("$","^"):
                continue
            self.bigrams[k] = (k[0],k[1],(v+smooth)/(freq_single[k[0]] + smooth * lengthOfTypesFromBigram))
        # print(self.bigrams)
        self.save()
        # for string in arr:
        #     print(": ",string)
    
    # Return the document-topic vector for a certain document. If docNum is NOT one of the document number you assigned, then use docName to find the document with the name.
    # The return should be a list, where the probabilities are ordered from topic 1, topic 2, …etc.
    def getDocumentTopic(docNum:int = -1, docName:str = ""):
        return

    # Return the document-topic vectors for all documents.
    # The return should be a list of tuples. Each tuple is made up of (x, y), where x is the name of the document and y is the document-topic vector 
    # of that document (represented as a list).
    def getAllDocumentTopic():
        return
    
    # 1. Return the topic-word vector for topic with topicNum
    # 2. topCount specify how many words are returned, based on the probability of the word in that 
    # topic (only select the highest ones to return). topCount <= 0, then return all words.
    # 3. The return should be a dictionary, where each entry is a word and its probabiltiies.
    def getTopicWordVector(topicNum, topCount = 10):
        return

    # 1.Print all topic-word vectors. topCount has the same definition as above.
    # 2.Your return should be a list of dictionaries, with the first dictionary is of topic 1, the second one is of topic 2 etc.
    def getTopicWordVectorAll(topCount = 10):
        return
    
    # Store the model in a human readable format as follows:
    # 1. All the output should be printed in the subdirectory named: <model name><suffix>. Notice that suffix can be any string, does not have to start with “.”.
    # 2. The document-topic should be printed in a file named “document-topics. Each line of the file should be formatted as follows
    #       The first word is the name of the document
    #       Then print one space character
    #       Then each dimension of the topic-document vector is printed (in order of topic 1, topic 2 etc.) There should be one space between each number.
    # 3. Each topic-word vector should be printed in a separate file named “topic_”<topic number>
    #       One line for each word’s probability
    #       Each line of the file should be formatted as <word> <probability of the word>
    #       The output should be sorted by decreasing order of probability
    def ExtendedPrint(dnamSuffix= ""):
        return

    # Save the model into a single file named <model name>.plsi
    # You can choose whatever format you want, as long as you can read it back,
    # The format does not have to be human-readable.
    # It is ok if you want to reuse the format of ExtendedPrint(), but everything has to be in one file.
    def save(self):
        if self.bigrams:
            pickle.dump(self.bigrams,open(self.__moduleName + ".bgmodel",'wb'))


