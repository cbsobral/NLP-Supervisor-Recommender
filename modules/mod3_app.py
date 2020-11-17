from mod1_data import get_data, corpus_lemma, corpus_stem, ud_lemma, ud_stem  
from mod2_vis import results_plot, get_topic_words, words_vis, get_topics_doc, supervisors, sim_matrix, recommend_df, super_vis
import streamlit as st
from gensim import models, similarities
import plotly.express as px
import pandas as pd
from multiprocessing import freeze_support


# App Header
st.set_page_config(layout="centered", initial_sidebar_state="auto", page_title="SRT") # "auto", "expanded", "collapsed"
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Supervisor Recommendation Tool")

# =============================================================================
# Side Bar
# =============================================================================

st.sidebar.header("Our App")
st.sidebar.write("This is the app description.")
st.sidebar.subheader("Instructions")
st.sidebar.write("1. Upload your file.\n2. Check out the suggestions.")
st.sidebar.subheader("About the Model")
st.sidebar.markdown("We use LDA to extract X topics from the supervision plans from [MyStudies](https://mystudies.hertie-school.org/en/), then we apply similarity score to improve recommendations.")


# =============================================================================
# LDA  and TF-IDF Data and Model
# =============================================================================

# Get Corpus
corpus_list = get_data()

# Load LDA Model
lda_model =  models.LdaModel.load('lda_model1')

# Load Similarities Matrix
sim_model = similarities.MatrixSimilarity.load('sim_model')

# Load dictionary and bow
dictionary, corpus_bow = corpus_lemma(corpus_list)
dict_stem, corpus_stem= corpus_stem(corpus_list)

# =============================================================================
# User's Input
# =============================================================================

# Get text from user
st.header("Your Proposal")
st.markdown("Type or paste (Ctrl+V) your research proposal text and then press `Run Comparison`")
ud_text = st.text_area("")

# Run comparison
run_app = st.button('Run Comparison')

if not run_app:
    st.stop()
else:
    
# =============================================================================
# Unseen Document Pre-processing    
# =============================================================================

    ud_bow_lemma = ud_lemma(ud_text, dictionary)
    ud_bow_stem = ud_stem(ud_text, dict_stem)
 
    
# =============================================================================
# Top 3 Topics Plot
# =============================================================================
    
    results_fig, first_topic, second_topic, third_topic = results_plot(lda_model, ud_bow_lemma) 
    st.header("Matching Topics")
    st.write("Our model indicates that these are the topics more closely related to your proposal")
    st.plotly_chart(results_fig)


# =============================================================================
# Words per topic - Plot and WordCloud
# =============================================================================
    
    # Words per topic table
    topic_words = get_topic_words(model=lda_model, dictionary=dictionary)
   
    # Define colors for Plot and WordCloud = first_topic
    colors_f1 = px.colors.sequential.Redor_r
    colors_w1 = "rgb(177, 63, 100)"
    
    # Get Plot and WordCloud
    fig_w1, wc_1 =  words_vis(topic= first_topic, topic_words=topic_words, 
                                      colors_fig = colors_f1, colors_wc = colors_w1) 
    
    
    # Define colors for Plot and WordCloud = second_topic
    colors_f2 = px.colors.sequential.Pinkyl_r
    colors_w2 = "rgb(225, 83, 131)"
    
    # Get Plot and WordCloud
    fig_w2, wc_2 =  words_vis(topic = second_topic, topic_words=topic_words, 
                                      colors_fig = colors_f2, colors_wc = colors_w2)
    
    # Define colors for Plot and WordCloud = third_topic
    colors_f3 = px.colors.sequential.Oryel_r
    colors_w3 = "rgb(238, 77, 90)"
    
    # Get Plot and WordCloud
    fig_w3, wc_3 =  words_vis(topic= third_topic, topic_words=topic_words, 
                                      colors_fig = colors_f3, colors_wc = colors_w3)
    
        
# =============================================================================
# Professors per Topic
# =============================================================================
    
    # Create table with supervisor prob per topics
    topics = [lda_model[corpus_bow[i]] for i in range(len(corpus_list.fileids()))]
    num_topics = 5
    supervisor_list = supervisors()
    
    document_topic = \
    pd.concat([get_topics_doc(topics_document, num_topics=num_topics) for topics_document in topics]) \
    .reset_index(drop=True).fillna(0)
    
    document_topic.index = supervisor_list # names to index column
        
# =============================================================================
# Sim Matrix
# =============================================================================
    
    sim_pd = sim_matrix(sim_model, ud_bow_stem, supervisor_list)
    
    # First topic table
    recom_1 = recommend_df(document_topic, first_topic, sim_pd, supervisor_list)
    recom_2 = recommend_df(document_topic, second_topic, sim_pd, supervisor_list)
    recom_3 = recommend_df(document_topic, third_topic, sim_pd, supervisor_list)
 
    # Visualization
    fig_s = super_vis(first_topic, second_topic, third_topic, recom_1, recom_2, recom_3)
    
    
# =============================================================================
#  Recommendations
# =============================================================================
    st.header("Recommendations")
    st.write("Select the most relevant topic for you from the dropdown menu to see recommendations")
    st.plotly_chart(fig_s)
    
    
    st.header("Words per Topic")
    st.write("Expand the sections to get a glimpse of the words within each topic")
    # First Topic
    expander1 = st.beta_expander('Topic '+str(first_topic))
    with expander1:
        col1, col2 = st.beta_columns(2)
        col1.plotly_chart(fig_w1, use_column_width=True)
        col2.image(wc_1.to_array(), use_column_width=True) 
    
    # Second
    expander2 = st.beta_expander('Topic '+str(second_topic))
    with expander2:
        col1, col2 = st.beta_columns(2)
        col1.plotly_chart(fig_w2, use_column_width=True)
        col2.image(wc_2.to_array(), use_column_width=True) 
        
    # Third
    expander3 = st.beta_expander('Topic '+str(third_topic))
    with expander3:
        col1, col2 = st.beta_columns(2)
        col1.plotly_chart(fig_w3, use_column_width=True)
        col2.image(wc_3.to_array(), use_column_width=True) 
        
        
if __name__ == "__main__":
    freeze_support()