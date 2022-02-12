#Programming Homework 1: (Language Model) 
import nltk
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize

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
        print("init works")
        if smooth < 0 or smooth > 1:
            smooth = 0

        reader = PlaintextCorpusReader(dirName,ext)
        print(reader.words())


    # Return the probability of the bigram (w1, w2). If either w1 or w2 is not in the corpus, it will return -1.
    # the user can pass “^” to denote “beginning of a sentence, and “$” as the end of sentence. You should also use these symbol to print the corresponding bigrams.
    def getProb(w1, w2):
        return -1
    
    #Return all the bigrams with w1 as the first word. It should return a list, each item of the list is a tuple (word, prob). 
    # If sortMethod = 1, the tuples are sorted alphabetically, if sortMethod = 2, the tuples are returned in decreasing order of probability (ties are broken alphabetically). 
    # Otherwise the list need not be sorted in any order. If w1 does not exist it will return an empty list.
    # the user can pass “^” to denote “beginning of a sentence, and “$” as the end of sentence. You should also use these symbol to print the corresponding bigrams.
    def getProbList(w1, sortMethod = 0):
        return 
    
    # Return all the bigrams with w2 as the second word. Otherwise the specification is the same as the getProbList() function above.
    # the user can pass “^” to denote “beginning of a sentence, and “$” as the end of sentence. You should also use these symbol to print the corresponding bigrams.
    def getProbList2(w2, sortMethod = 0):
        return 
    
    # Return all the bigrams and their probabilities as a list. Each item in the list is a tuple (word1, word2, prob). 
    # sortMethod is as above, except when sortMethod = 1, the tuples are sorted alphabetically by the first word of the tuple 
    # (break ties with the alphabetical order of the second word), and when sortMethod = 3, 
    # the tuples are sorted by the second word of the tuple (break tie with the alphabetical order of the first word).
    def getAll(sortMethod = 0):
        return
    
    # Save the calculated probabilities in a file. The filename will be appended with the extension “.bgmodel”. 
    # You can select whatever way you want to store the probabilities, but you need to ensure when I read back the probabilities 
    # (with the toload=true flag in the constructor), the correct probabilities are read back.
    def save(filename = "default"):
        return







model = BigramModel("name","corpus/","*")