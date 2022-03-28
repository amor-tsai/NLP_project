from gensim.test.utils import common_texts, common_corpus, common_dictionary
from gensim.corpora.dictionary import Dictionary
from gensim.models import *

# Create a corpus from a list of texts
lda = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics=10, workers=1)
# Train the model on the corpus.
# lda = LdaModel(common_corpus, num_topics=10)

# Create a new corpus, made of previously unseen documents.
# other_texts = [
#     ['computer', 'time', 'graph'],
#     ['survey', 'response', 'eps'],
#     ['human', 'system', 'computer']
# ]
# other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
# unseen_doc = other_corpus[0]
# vector = lda[unseen_doc]  # get topic probability distribution for a document

# print(vector)
