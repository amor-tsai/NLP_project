# CS7322 NLP
# author: Amor Tsai
#Programming Homework 1: (Language Model) 
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

class BigramModel:
# name: name of the model. The name is used as the filename to store the model
# dirName: the directory that store the corpus. If dname starts with a “/”, then dname is an absolute path. Otherwise the path is relative starting from the current directory.
#  (Default is the current directory)
# ext: the extension of all files that is consider part of the corpus. Only files that have the specified extension will be read and processed. Default is “*’, 
# which mean all files in the directory will be considered
# toload: if true, the bigram model should be loaded from a file named <filename>.bgmodel. (See the save() function for details), (and all other parameters are ignored).
# smooth: the smoothing method that is used. Default Is no smoothing. If smooth is a floating point number between 0 and 1 (strictly greater than 0, less than or equal to 1),
#  then we apply add-k smoothing to the bigram probability. If smooth is other values than no smoothing is done.
# stopWordList: a list of words that will be removed from the corpus before calculating the bigram probability. Default is empty
# otherWordList: a list of words that will be grouped and treated as a single word for calculating bigrams probabilities.
# singlesen: If true, each document will be viewed as a single sentence. Otherwise, the program should segment each documents into sentences and calculate bigrams accordingly.
#  (i.e. the end of the first sentence and the beginning of the second sentence is not treated as a bigram).
    def __init__(self, name: str, dirName: str = ".", ext: str = "*", toload: bool = False, smooth: int = 0, stopWordList: list = [], otherWordList: list = [], singlesen: bool = False):
        self.__moduleName = name
        # toload == True, should load module from file
        if toload:
            fileName = self.__moduleName + ".bgmodel"
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
                if ch in otherWordList:
                    ch = otherWordList[0]
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
        
    # return a list incorporating sentences
    def getSentences(self, dirName:str = ".", ext: str = "*", singlesen: bool = False):
        arr = []
        for file in os.listdir(dirName):
            if ext != "*" and False == file.endswith(ext):
                continue
            # that means, each document should be treated as a sentence
            if singlesen:
                # if len(stopWordList) > 0:
                #     sen = [ch for ch in nltk.tokenize.word_tokenize(sen) if not ch in stopWordList]
                #     print(sen)
                arr.append(
                    open(join(dirName,file),"r").read().lower().translate(str.maketrans('', '', string.punctuation))
                )
            else:
                nltk.download('punkt')
                arr += sent_tokenize(open(join(dirName,file),"r").read().lower())
        return arr

    # Return the probability of the bigram (w1, w2). If either w1 or w2 is not in the corpus, it will return -1.
    # the user can pass “^” to denote “beginning of a sentence, and “$” as the end of sentence. You should also use these symbol to print the corresponding bigrams.
    def getProb(self,w1, w2):
        if (w1,w2) in self.bigrams:
            return self.bigrams[(w1,w2)][2]
        return -1
    
    # Return all the bigrams with w1 as the first word. It should return a list, each item of the list is a tuple (word, prob). 
    # If sortMethod = 1, the tuples are sorted alphabetically, if sortMethod = 2, the tuples are returned in decreasing order of probability (ties are broken alphabetically). 
    # Otherwise the list need not be sorted in any order. If w1 does not exist it will return an empty list.
    # the user can pass “^” to denote “beginning of a sentence, and “$” as the end of sentence. You should also use these symbol to print the corresponding bigrams.
    def getProbList(self, w1, sortMethod = 0):
        res = []
        if self.bigrams:
            for k in self.bigrams.keys():
                if w1 == k[0]:
                    res.append(
                        (self.bigrams[k][1],self.bigrams[k][2])
                    )
        if res:
            if sortMethod == 1:
                # sorted by alphabet
                res = sorted(res,key=lambda x:x[0])
            elif sortMethod == 2:
                # sorted by decreasing order of probability
                res = sorted(res,key=lambda x:(-x[1],x[0]))
        return res
    
    # Return all the bigrams with w2 as the second word. Otherwise the specification is the same as the getProbList() function above.
    # the user can pass “^” to denote “beginning of a sentence, and “$” as the end of sentence. You should also use these symbol to print the corresponding bigrams.
    def getProbList2(self, w2, sortMethod = 0):
        res = []
        if self.bigrams:
            for k in self.bigrams.keys():
                if w2 == k[1]:
                    res.append(
                        (self.bigrams[k][0],self.bigrams[k][2])
                    )
        if res:
            if sortMethod == 1:
                # sorted by alphabet
                res = sorted(res,key=lambda x:x[0])
            elif sortMethod == 2:
                # sorted by decreasing order of probability
                res = sorted(res,key=lambda x:(-x[1],x[0]))
        return res
    
    # Return all the bigrams and their probabilities as a list. Each item in the list is a tuple (word1, word2, prob). 
    # sortMethod is as above, except when sortMethod = 1, the tuples are sorted alphabetically by the first word of the tuple 
    # (break ties with the alphabetical order of the second word), and when sortMethod = 3, 
    # the tuples are sorted by the second word of the tuple (break tie with the alphabetical order of the first word).
    def getAll(self, sortMethod = 0):
        res = self.bigrams.values()
        if res:
            if sortMethod == 1:
                res = sorted(res,key=lambda x:(x[0],x[1]))
            elif sortMethod == 2:
                res = sorted(res,key=lambda x:(-x[2],x[0]))
            elif sortMethod == 3:
                res = sorted(res,key=lambda x:(x[1],x[0]))
        return res
    
    # Save the calculated probabilities in a file. The filename will be appended with the extension “.bgmodel”. 
    # You can select whatever way you want to store the probabilities, but you need to ensure when I read back the probabilities 
    # (with the toload=true flag in the constructor), the correct probabilities are read back.
    def save(self):
        if self.bigrams:
            pickle.dump(self.bigrams,open(self.__moduleName + ".bgmodel",'wb'))

        


        




