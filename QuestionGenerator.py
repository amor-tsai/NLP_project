'''
use spacy
python -m spacy download en_core_web_sm

use nltk
use nltk wordnet

'''
import spacy
import nltk
from nltk.corpus import wordnet as wn
# print(spacy.__version__)

# nlp = spacy.load("en_core_web_sm")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

class QuestionGenerator:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        
    '''
    given chunk and entities, return what,who,when,where
    '''
    def getInterrogativeExpressing(self,txt,chunk,entities):
        ent_labels = {'person':'Who','object':'What','date':'When','time':'When','money':'How much', 'org':'Who'}
        other = {'person':'Who','object':'What'}
        res = None
        # use spacy named entity recognization first
        if chunk.text in entities:
            label = entities[chunk.text].label_.lower()
            if label in ent_labels:
                res = (ent_labels[label],txt)
            else:
                res = (ent_labels['object'],txt)
        elif txt in entities:
            label = entities[txt].label_.lower()
            if label in ent_labels:
                res = (ent_labels[label],txt)
            else:
                res = (ent_labels['object'],txt)
        elif txt.lower() in ['me','he','she','him','her']:
            res = (other['person'],txt)
        else:
            try:
                xword = txt                
                x = wn.synsets(xword, pos=wn.NOUN)[0]
                scores = []
                for word in other.keys():
                    w = wn.synsets(word,pos=wn.NOUN)[0]
                    scores.append(w.lch_similarity(x))
                idx = scores.index(max(scores))
                key = list(other.keys())[idx]
                res = (other[key],txt)
            except IndexError as err:
                print(err)
                
        return res
    
    '''
    rebuild the structure of sentence
    '''
    def rebuildStructure(self,root, token, w):
        if not root:
            return []
        seq = []
        # mark, if need to change verb to its lemma
        if token.dep_ == 'nsubj':
            is_lemma = False
        else:
            is_lemma = True
        
        for t in root.subtree:
            if t in token.children:
                continue
            elif t.dep_ == 'prep' and token in t.children and w=='When':
                continue
            if token.dep_ == 'nsubj' and t == token:
                seq.append(w)
                continue
            elif (token.dep_ == 'dobj' or token.dep_ == 'pobj' or token.dep_ == 'npadvmod') and t == token:
                if root.tag_ == 'VBD':
                    seq.insert(0,'did')
                elif root.tag_ == 'VBZ':
                    seq.insert(0,'does')
                elif root.tag_ == '':
                    seq.insert(0,'do')
                seq.insert(0,w)
                continue
            
            # if the token required to replace is nsubj, then keep the original verb, otherwise change to its lemma
            if is_lemma and t == root:
                seq.append(t.lemma_)
            else:
                seq.append(t.text)
        return seq
        
        
    
    def getQuestions(self,sent:str):
        doc = self.nlp(sent)

        tokens = {}
        token_seq = []
        chunks = {}
        entities = {}
        asks = []
        root = None
        rootIndex = None

        for i,token in enumerate(doc):
            tokens[token.text] = token
            token_seq.append(token.text)
            if token.dep_ == 'ROOT':
                root = token
                rootIndex = i
        
        for chunk in doc.noun_chunks:
            chunks[chunk.root.text] = chunk
        
        for ent in doc.ents:
            entities[ent.text] = ent
#             print('{} {}'.format(ent.text,ent.label_))
            if ent.text not in chunks:
                chunks[ent.root.text] = ent
        
        for name in chunks:
            tmp = self.getInterrogativeExpressing(name,chunks[name],entities)
            if tmp:
                asks.append(tmp)
        
        
#         for token in tokens.values():
#             print('token: {} pos: {} dep: {} ent: {} tag: {}'.format(token.text,token.pos_,token.dep_, token.ent_type_, token.tag_))
        
#         print('tokens ',tokens)
#         print('token_seq ',token_seq)
#         print('nounChunks ',chunks)
#         print('entities ',entities)
#         print('asks ',asks)
        
        for w,word in asks:
            print('Question: ',' '.join(self.rebuildStructure(root,tokens[word],w)))

    