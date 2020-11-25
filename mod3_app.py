from mod1_data import get_data, corpus_lemma, corpus_stem, ud_lemma, ud_stem  
from mod2_vis import results_plot, get_topic_words, words_vis, get_topics_doc, supervisors, sim_matrix, recommend_df, super_vis
import streamlit as st
from gensim import models, similarities
import plotly.express as px
import pandas as pd
from multiprocessing import freeze_support
import pickle

# App Header
st.set_page_config(layout="centered", initial_sidebar_state="auto", page_title="SRT") # "auto", "expanded", "collapsed"
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Supervisor Recommendation")

# =============================================================================
# Side Bar
# =============================================================================

st.sidebar.subheader("What is this?")
st.sidebar.markdown("This is an app designed to help you match your research proposal with possible thesis supervisors.")
st.sidebar.subheader("What should I do?")
st.sidebar.markdown("1. Copy and paste your research proposal in the box. You can also just type your research interests. \n2. Check out which of the topics in our model best relate to your idea. \n3. Select one of the topics and see which professors match your interests.")
st.sidebar.subheader("Is this witchcraft?")
st.sidebar.markdown("Err... no. We used Python to employ a Latent Dirichlet Allocation (LDA) topic model to unveil the underlying topics from the collection of supervision plans available at [MyStudies](https://mystudies.hertie-school.org/en/). ")
st.sidebar.markdown("Based on the model, we then checked which topics from the supervision plans were best suited to represent your interests. ")
st.sidebar.markdown("Finally, to narrow it down, we ran a similarity score between your proposal and the professors' plans associated with the topic you thought better represented your interests. This is our final output.")
st.sidebar.subheader("What now?")
st.sidebar.markdown("That's it from our side! But you should go over the supervisor plans we selected and make sure you agree with our algorithm before sealing the deal :) ")
st.sidebar.markdown("Fear not, for we've incorporated a hyperlink to make this easier too! Just click on the professor and you will be redirected to the appropriate page.")
st.sidebar.markdown("We hope this can facilitate your choice for supervisors! And good luck with your thesis!")

# =============================================================================
# LDA  and TF-IDF Data and Model
# =============================================================================

# Get Corpus
corpus_list = pickle.load(open("corpus.p", "rb"))

# Load LDA Model
lda_model =  models.LdaModel.load('lda_model1')

# Load Similarities Matrix
sim_model = similarities.MatrixSimilarity.load('sim_model')

# Load dictionary and bow
dict_lemma, corpus_lemma = corpus_lemma(corpus_list)
dict_stem, corpus_stem= corpus_stem(corpus_list)



# =============================================================================
# User's Input
# =============================================================================

# Get text from user
st.header("Your Research Proposal")
st.markdown("Type or paste (Ctrl+V) your text and then press `Run Comparison`")
ud_text = st.text_area("")

# Run comparison
run_app = st.button('Run Comparison')

if not run_app:
    st.stop()
else:
    
# =============================================================================
# Unseen Document Pre-processing    
# =============================================================================

    ud_bow_lemma = ud_lemma(ud_text, dict_lemma)
    ud_bow_stem = ud_stem(ud_text, dict_stem)
 
    
# =============================================================================
# Top 3 Topics Plot
# =============================================================================
    
    results_fig, first_topic, second_topic, third_topic = results_plot(lda_model, ud_bow_lemma) 
    st.header("Matching Topics")
    st.write("These are the topics more closely related to your proposal:")
    st.plotly_chart(results_fig)

  
# =============================================================================
# Words per topic - Plot and WordCloud
# =============================================================================
    
    # Words per topic table
    topic_words = get_topic_words(model = lda_model, dictionary = dict_lemma)
   
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
    
    
    st.header("Words per Topic")
    st.write("Have a look at the words that stand out in each of these topics. Click on (+) to see what we are talking about.")
    
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
      
   
# =============================================================================
# Professors per Topic
# =============================================================================
    
    # Create table with supervisor prob per topics
    topics = [lda_model[corpus_lemma[i]] for i in range(len(corpus_list.fileids()))]
    num_topics = 6
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
    st.write("Which of the previous topics do you think is a better match for you? Select a topic from the dropdown menu to see our supervisor recommendations for each topic.")
    st.plotly_chart(fig_s)
     
        
if __name__ == "__main__":
    freeze_support()
    
    
    