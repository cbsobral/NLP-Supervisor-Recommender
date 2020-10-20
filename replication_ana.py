# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:05:21 2020

@author: Ana
"""

import nltk
from nltk.corpus import PlaintextCorpusReader
folder = (r'C:/Users/Ana/Documents/Master of Public Policy/2020.2/Python Programming/Final Project/python_project-master/data/txt')
corpus_list = PlaintextCorpusReader(folder, '.*txt')  # all files ending in 'txt'

#creates a column with file names as file ids.
file_ids = corpus_list.fileids()

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
import gensim


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# lemmatizer goes to root word
wordnet = nltk.WordNetLemmatizer()

stoplist = stopwords.words('english')
additional_stopwords = """question impact professor school dissertation paper take following http nuffield
                          title school session study work topics project partner practice happy plan see supervise
                          research thesis issue design student topic supervision university lab mia mpp"""  # define additional stopwords in a string
stoplist += additional_stopwords.split()

def normalize_token(token):
    """
    Convert token to lowercase, and stem using the Porter algorithm.
    """
    return wordnet.lemmatize(token.lower())

def filter_token(token):
    """
    Evaluate whether or not to retain ``token``.
    """
    token = token.lower()
    return token not in stoplist and token.isalpha() and len(token) > 2

documents=[[normalize_token(token) 
            for token in corpus_list.words(fileids=[fileid])
            if filter_token(token)]
            for fileid in corpus_list.fileids()]

dictionary = gensim.corpora.Dictionary(documents)         
documents_bow = [dictionary.doc2bow(document) for document in documents]

model = gensim.models.LdaModel(documents_bow, 
                               id2word=dictionary,
                               num_topics=15, 
                               update_every=0,
                               random_state=123,
                               passes=500)

for i, topic in enumerate(model.print_topics(num_topics=8, num_words=7)):
    print (i, ':', topic)

#Add Text - Question
yr_text = input("Please insert text:")
#@title Add Document
path = "C:/Users/Ana/Documents/Master of Public Policy/2020.2/Python Programming/Final Project/python_project-master/data/docs/ana.txt" #@param {type:"string"}
yr_p = open(path)
yr_path = yr_p.read()

# if you want to analyze text inserted, change yr_path to yr_text
yr_tokens = nltk.word_tokenize(yr_path)
yr_bow_vector = dictionary.doc2bow(yr_tokens)

# model comparison result
#print(model[yr_bow_vector])

# pd data frame
results = pd.DataFrame(model[yr_bow_vector])
results.columns = ['topic', 'proximity']
results.sort_values(['proximity'], ascending=False, inplace=True)
print(results.nlargest(3,['proximity']))

# table with documents and topic probability
topics = [model[documents_bow[i]] for i in range(len(documents))]
num_topics = 8

def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

document_topic = \
pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]) \
  .reset_index(drop=True).fillna(0)

#document_topic
document_topic.sort_values(5, ascending=False)[5].head(10)

documents_lda = model[documents_bow]

topic_dt = pd.DataFrame(documents_lda)
doc_dt = pd.DataFrame(file_ids)
conc = pd.concat([doc_dt, topic_dt], axis=1)
conc.columns = ['doc_id', 'topic1', 'topic2', 'topic3']
conc.sort_values(['topic1'], ascending=True, inplace=True)
conc