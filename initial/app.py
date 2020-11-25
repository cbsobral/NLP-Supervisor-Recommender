import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
import gensim
from gensim import models
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from matplotlib import pyplot as plt


### Header
st.title("Supervisor Recommendation Tool")

### Data and Model 
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

# Define additional stopwords in a string
additional_stopwords = """question impact professor school dissertation paper take following http nuffield
                          title school session study work topics project partner practice happy plan see supervise
                          research thesis issue design student topic supervision university lab mia mpp"""  

stoplist += additional_stopwords.split()

# Convert token to lowercase and lemmatize 
def normalize_token(token):
  return wordnet.lemmatize(token.lower())

# Evaluate whether or not to retain `token`
def filter_token(token):
    token = token.lower()
    return token not in stoplist and token.isalpha() and len(token) > 2

# Tokenize and apply functions to files
corpus=[[normalize_token(token) 
            for token in corpus_list.words(fileids=[fileid])
            if filter_token(token)]
            for fileid in corpus_list.fileids()]

# Create bag of words for each document
dictionary = gensim.corpora.Dictionary(corpus)         
corpus_bow = [dictionary.doc2bow(document) for document in corpus]

# Load model from topic_choices.ipynb
lda_model =  models.LdaModel.load('lda_model')



### Text input or upload ----------
st.subheader("Your research proposal goes here:")
default_value = "Your research could be about politics, or taxation, or health inequalities."
ud_text = st.text_area("", default_value)

#ud_path = st.file_uploader(label='Only .txt files are allowed.', type='txt' )



### Comparison with lda_model ----------
# Pre-process
ud_tokens = nltk.word_tokenize(ud_text) #  ud can be either `ud_text` or `ud_path`
ud_lemma = ' '.join([wordnet.lemmatize(w) for w in ud_tokens]) 
ud_tk_lemma = nltk.word_tokenize(ud_lemma) 

# Comparison
ud_bow_vector = dictionary.doc2bow(ud_tk_lemma)



### Results ----------
## Table with top 3 topics and score
ud_bow_vector = dictionary.doc2bow(ud_tokens)
results = pd.DataFrame(lda_model[ud_bow_vector])
results.columns = ['Topic', 'Proximity Score']
results.sort_values(['Proximity Score'], ascending=False, inplace=True)
results_3 = (results.nlargest(3,['Proximity Score']))

# Get topic numbers
first_choice = int(results_3.iloc[0]['Topic'])
second_choice = int(results_3.iloc[1]['Topic'])
third_choice = int(results_3.iloc[2]['Topic'])

# Display table
st.subheader("Here are the topics most related to your research proposal:")
st.table(results_3.assign(hack='').set_index('hack')) 



### Words per topic ----------
## Create table with top words in the topics above
n_words = 20
topic_words = pd.DataFrame({})

for i, topic in enumerate(lda_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
    topic_words = pd.concat([topic_words, topic_df], ignore_index=True)

tw_results = topic_words[(topic_words['topic'] == first_choice) | (topic_words['topic'] == second_choice) | (topic_words['topic'] == third_choice)] 

## Word Cloud ----------
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader("This are the words most commonly associated with each topic:")
# First_choice
# Wordcloud
text1_fc = topic_words[(topic_words['topic'] == first_choice)]
text2_fc = text1_fc['word']
text3_fc = ' '.join(text2_fc)
wordcloud1 = WordCloud(max_font_size=40, max_words=100, background_color="white").generate(text3_fc)

# Plot
plt.title('Topic '+str(first_choice))
plt.imshow(wordcloud1)
plt.axis("off")
st.pyplot()

# Second_choice
# Wordcloud
text1_sc = topic_words[(topic_words['topic'] == second_choice)]
text2_sc = text1_sc['word']
text3_sc = ' '.join(text2_sc)
wordcloud2 = WordCloud(max_font_size=40, max_words=100, background_color="white").generate(text3_sc)

# Plot
plt.title('Topic '+str(second_choice))
plt.imshow(wordcloud2)
plt.axis("off")
st.pyplot()

# Third_choice
# Wordcloud
text1_tc = topic_words[(topic_words['topic'] == third_choice)]
text2_tc = text1_tc['word']
text3_tc = ' '.join(text2_tc)
wordcloud3 = WordCloud(max_font_size=40, max_words=100, background_color="white", color_func=lambda *args, **kwargs: (255,0,0)).generate(text3_tc)

# Plot
plt.title('Topic '+str(third_choice))
plt.imshow(wordcloud3)
plt.axis("off")
st.pyplot()

## Visualize graph with info from `tw_results` 
st.subheader("This are the words most commonly associated with each topic (in a graph):")
df = tw_results
fig = px.bar(df, x="value", y="word", orientation='h', text='word',
             facet_col="topic")
fig.update_yaxes(matches=None, visible=False)
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(uniformtext_minsize=7, uniformtext_mode='show')
st.plotly_chart(fig)



### Supervisor Recommendation ----------
## Create table with supervisors, using results from comparison
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

# Supervisor names to index column
document_topic.index = ['Anheier', 'Bryson', 'Cali', 'Cingolani', 'Costello', 'Dawson', 'Flachsland', 'GohdesHW', 
                        'Graf', 'Hallerberg', 'Hammerschmid', 'Hassel', 'Hirth', 'Hustedt', 'Iacovone', 'Jachtenfuchs', 
                        'Jankin', 'Kayser', 'Kreyenfeld', 'Mair', 'Mena', 'MungiuPippidi', 'Munzert', 'Patz', 'Reh', 
                        'Roemmele', 'Shaikh', 'Snower', 'Stockman', 'Traxler', 'Wegrich']


## Table with topic probabilities per document
ud_recommend1 = pd.DataFrame(document_topic.sort_values(first_choice, ascending=False)[first_choice]).head(5)
ud_recommend1 = pd.DataFrame(ud_recommend1.index.tolist())

ud_recommend2 = pd.DataFrame(document_topic.sort_values(second_choice, ascending=False)[second_choice]).head(5)
ud_recommend2 = pd.DataFrame(ud_recommend2.index.tolist())

ud_recommend3 = pd.DataFrame(document_topic.sort_values(third_choice, ascending=False)[third_choice]).head(5)  # from document_topic table
ud_recommend3 = pd.DataFrame(ud_recommend3.index.tolist())

concat_recom = pd.DataFrame(np.hstack([ud_recommend1, ud_recommend2, ud_recommend3]))
concat_recom.columns =['Topic '+str(first_choice), 'Topic '+str(second_choice), 'Topic '+str(third_choice)] 

# Print results
st.subheader("Here are our recommendations:")
st.table(concat_recom.assign(hack='').set_index('hack'))







