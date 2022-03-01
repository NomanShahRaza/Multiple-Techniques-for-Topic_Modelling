"""
@author: Noman Raza Shah
"""
#%% 
# ========================== #
# TOPIC MODELLING TECHNIQUES #
# ========================== #

from __future__ import print_function

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
import string
import argparse
import re


def tag_generateLDA(text):

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    doc_complete = text.split('\n')

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]    
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)
    topic = ldamodel.print_topics(num_topics=5, num_words=5)

    hashtags = []
    for t in topic: 
        for h in t[1].split('+'):
            hashtags.append(' '+h[h.find('"')+1:h.rfind('"')])
    htt = []
    for ht in list(set(hashtags)):
        # print(ht, end=' ')
        print(ht, end='\n')
        htt.append(ht)
    return htt

def lsa(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    x = vectorizer.fit_transform(text)
    lsa = TruncatedSVD(n_components=1,n_iter=100)
    lsa.fit(x)
    terms = vectorizer.get_feature_names() 
    #print(lsa.components_)
    for ind,comp in enumerate(lsa.components_):
        termsInComp = zip(terms,comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:7]
        #print("Concept %d" % ind)
        for term in sortedTerms:
            print(term[0])

def NMF_modelling(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    import numpy as np
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    x = vectorizer.fit_transform(text)
    # Applying Non-Negative Matrix Factorization
    nmf = NMF(n_components=1)
    nmf.fit_transform(x)
    nmf.components_
    terms = vectorizer.get_feature_names() 
    words = np.array(vectorizer.get_feature_names())
    for i, topic in enumerate(nmf.components_):
        print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))
        
#%%


text= """we are working in Artificial intelligence lab. we have few groups who are focus on the area of machine learning and artificial intelligence.
Google tackles the most challenging problems in computer science. Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing 
our research and tools to fuel progress in the field. Our researchers publish regularly in academic journals, release projects as open source, and apply research 
to Google products."""

print('='*70)
print('LDA')
print(tag_generateLDA(text))
print('='*70)
text= ["We are working in an artificial intelligence lab. We have a few groups that are focused on the area of machine learning, deep learning, and artificial intelligence."," Artificial intelligence lab research the field of Artificial intelligence and leverages the power of AI for social good."," The lab has been working in several research fields that are Computer vision, Data Science, Embedded Systems, and Platform. AI lab start building AI solutions with powerful tools and services."," Our teams aspire to make discoveries, and the core to our approach is sharing our research and tools to fuel progress in the field."," At Artificial intelligence Lab, we aspire to produce AI solutions that aim to solve contemporary problems in Pakistan, in collaboration with the industry. We continue to produce quality research and trained individuals who have developed the expertise and have the potential to form the core of the AI revolution in Pakistan in years to come."]
print('='*70)
print('LSA')
print(lsa(text))
print('='*70)

print('='*70)
print('NMF')
print(NMF_modelling(text))
print('='*70)



