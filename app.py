import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
import gensim
from gensim import models
import numpy as np
#import plotly
#import plotly.express as px
#from io import StringIO
#from nltk.stem import PorterStemmer


### Header
st.title("Supervisor Recommendation Tool")
st.markdown("With this tool you can get good recommendations :)")

### Load Data
@st.cache(allow_output_mutation=True, show_spinner=True)
def get_data():
    path = ('c:\\Users\\carol\\Desktop\\Fall_2020\\Python\\streamlit\\data')
    return PlaintextCorpusReader(path, '.*txt') # import all files ending in 'txt'

corpus_list = get_data() 

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

wordnet = nltk.WordNetLemmatizer()
stoplist = stopwords.words('english')

# define additional stopwords in a string
additional_stopwords = """question impact professor school dissertation paper take following http nuffield
                          title school session study work topics project partner practice happy plan see supervise
                          research thesis issue design student topic supervision university lab mia mpp"""  

stoplist += additional_stopwords.split()

# convert token to lowercase and stem using the Porter algorithm
def normalize_token(token):
  return wordnet.lemmatize(token.lower())

# evaluate whether or not to retain `token`
def filter_token(token):
    token = token.lower()
    return token not in stoplist and token.isalpha() and len(token) > 2

# tokenize and apply functions to files
corpus=[[normalize_token(token) 
            for token in corpus_list.words(fileids=[fileid])
            if filter_token(token)]
            for fileid in corpus_list.fileids()]

# create bag of words for each document
dictionary = gensim.corpora.Dictionary(corpus)         
corpus_bow = [dictionary.doc2bow(document) for document in corpus]

# load model
lda_model =  models.LdaModel.load('lda_model')
#--------------------------------------------------------------------------------------

### Text input or upload ----------
default_value = "Your research could be about politics, or taxation, or health inequalities."
ud_doc = st.text_area("Your research proposal here:", default_value)

#st.file_uploader(label='Only .txt files are allowed.', type='txt' )


### Comparison with lda_model ----------
# pre-process
ud_tokens = nltk.word_tokenize(ud_doc) #change file used for recommendation here
ud_lemma = ' '.join([wordnet.lemmatize(w) for w in ud_tokens]) # ud can be either `ud_text` or `ud_path`
ud_tk_lemma = nltk.word_tokenize(ud_lemma) 

#comparison
ud_bow_vector = dictionary.doc2bow(ud_tk_lemma)

### Results ----------

# table with top 3 topics and score
ud_bow_vector = dictionary.doc2bow(ud_tokens)
results = pd.DataFrame(lda_model[ud_bow_vector])
results.columns = ['Topic', 'Proximity Score']
results.sort_values(['Proximity Score'], ascending=False, inplace=True)
results_3 = (results.nlargest(3,['Proximity Score']))

# get topic numbers
first_choice = int(results_3.iloc[0]['Topic'])
second_choice = int(results_3.iloc[1]['Topic'])
third_choice = int(results_3.iloc[2]['Topic'])

# display table
st.header("Here are the topics most related to your research proposal:")
st.table(results_3.assign(hack='').set_index('hack')) 

### Wods per topic ----------

# create table with top words in the topics above
n_words = 10
topic_words = pd.DataFrame({})

for i, topic in enumerate(lda_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
    topic_words = pd.concat([topic_words, topic_df], ignore_index=True)

tw_results = topic_words[(topic_words['topic'] == first_choice) | (topic_words['topic'] == second_choice) | (topic_words['topic'] == third_choice)] 

import seaborn as sns
from matplotlib import pyplot as plt

# visualize graph with info from the table above
st.header("This are the words most commonly associated with each topic:")
st.set_option('deprecation.showPyplotGlobalUse', False)
chart_data = sns.FacetGrid(tw_results, col="topic", col_wrap=3, sharey=False)
chart_data.map(plt.barh, "word", "topic")
st.pyplot()

### Supervisor ----------
# create table with supervisors, using results from comparison
topics = [lda_model[corpus_bow[i]] for i in range(len(corpus))]
num_topics = 10

def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

document_topic = \
pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]) \
  .reset_index(drop=True).fillna(0)

# supervisor names to index column
document_topic.index = ['Anheier', 'Bryson', 'Cali', 'Cingolani', 'Costello', 'Dawson', 'Flachsland', 'GohdesHW', 
                        'Graf', 'Hallerberg', 'Hammerschmid', 'Hassel', 'Hirth', 'Hustedt', 'Iacovone', 'Jachtenfuchs', 
                        'Jankin', 'Kayser', 'Kreyenfeld', 'Mair', 'Mena', 'MungiuPippidi', 'Munzert', 'Patz', 'Reh', 
                        'Roemmele', 'Shaikh', 'Snower', 'Stockman', 'Traxler', 'Wegrich']


# table with topic probabilities per document
ud_recommend1 = pd.DataFrame(document_topic.sort_values(first_choice, ascending=False)[first_choice]).head(5)
ud_recommend1 = pd.DataFrame(ud_recommend1.index.tolist())

ud_recommend2 = pd.DataFrame(document_topic.sort_values(second_choice, ascending=False)[second_choice]).head(5)
ud_recommend2 = pd.DataFrame(ud_recommend2.index.tolist())

ud_recommend3 = pd.DataFrame(document_topic.sort_values(third_choice, ascending=False)[third_choice]).head(5)  # from document_topic table
ud_recommend3 = pd.DataFrame(ud_recommend3.index.tolist())

concat_recom = pd.DataFrame(np.hstack([ud_recommend1, ud_recommend2, ud_recommend3]))
concat_recom.columns =['Topic '+str(first_choice), 'Topic '+str(second_choice), 'Topic '+str(third_choice)] 

# print results
st.table(concat_recom.assign(hack='').set_index('hack'))