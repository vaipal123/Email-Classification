from tabulate import tabulate
#import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
import itertools
from nltk.corpus import stopwords

# TRAIN_SET_PATH = "20ng-no-stop.txt"
# TRAIN_SET_PATH = "r52-all-terms.txt"
# TRAIN_SET_PATH = "r8-no-stop.txt"
#TRAIN_SET_PATH = "SpamCollection.txt"
TRAIN_SET_PATH = "finalDataSet.txt"
GLOVE_6B_50D_PATH = "glove.6B.50d.txt"
GLOVE_840B_300D_PATH = "glove.840B.300d.txt"


stop = set(stopwords.words('english'))


X, y = [], []
with open(TRAIN_SET_PATH, "rb") as infile:
    for line in infile:
        label, text = line.split("\t")

        X.append(text.split())
        y.append(label)
X, y = np.array(X), np.array(y)
print "total examples %s" % len(y)

#X, y = [], []
#with open(TRAIN_SET_PATH, "rb") as infile:
#    for line in infile:
#        z = []
#         label, text = line.split("\t")
#         for i in text.lower().split():
#             if i not in stop:
#                 z.append(i)  
 
#         X.append(z)
#         y.append(label)
# X, y = np.array(X), np.array(y)



with open(GLOVE_6B_50D_PATH, "rb") as lines:
    word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}


glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0]
        nums = map(float, parts[1:])
        if word in all_words:
            glove_small[word] = np.array(nums)
            

model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
model.index2word
w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}

#print model.syn0[0]
#x = {key:value for key,value in w2v.items()[0:4]}
#print x


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)), 
                        ("extra trees", RandomForestClassifier(n_estimators=200))])


etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("extra trees", RandomForestClassifier(n_estimators=200))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), 
                        ("extra trees", RandomForestClassifier(n_estimators=200))])

etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", RandomForestClassifier(n_estimators=200))])


all_models = [
    ("glove_small", etree_glove_small), 
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
]
scores = sorted([(name, cross_val_score(model, X, y, cv=5).mean()) 
                 for name, model in all_models], 
                key=lambda (_, x): -x)
print tabulate(scores, floatfmt=".4f", headers=("model", 'score'))
