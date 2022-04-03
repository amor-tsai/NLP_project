from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import *
from gensim.test.utils import datapath
import os
import nltk
from readTextFiles import readTextFiles 
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
stop_words = stopwords.words('english')


class LDA:
    def __init__(self,name:str = 'default', dirName:str = '.', ext: str = '*',toload: bool = False, num_topics=10):
        self.name = name
        if toload:
            self.model = LdaModel.load(self.name)
        else:
            dataset = api.load("text8")
            data = [d for d in dataset]
            print(data)
            return


            common_dictionary = Dictionary(readTextFiles(ext='tt'))
            common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
            self.model = LdaModel(common_corpus,id2word=common_dictionary, num_topics=num_topics)
            print('common_texts ',common_texts)
            print('common_corpus: ',common_corpus)
            print(common_dictionary)
            print(self.model)



    def save(self):
        if self.model:
            self.model.save(datapath(self.name))