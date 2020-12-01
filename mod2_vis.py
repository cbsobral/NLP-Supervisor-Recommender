import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from multiprocessing import freeze_support

# =============================================================================
# Top 3 Topics
# =============================================================================
    
def results_plot(model, ud_bow):
    # Table with top 3 topics and score
    results = pd.DataFrame(model[ud_bow]) # Comparison with LDA
    results.columns = ['Topic', 'Score']
    results.sort_values(['Score'], ascending=False, inplace=True)
    results_top = (results.nlargest(3,['Score']))  # Top 3 topics
    results_top['Score'] = results_top['Score']*100
    
    # Get topic numbers
    first_topic = int(results_top.iloc[0]['Topic'])
    second_topic = int(results_top.iloc[1]['Topic'])
    third_topic = int(results_top.iloc[2]['Topic'])
    
    
    #  Create bar figure  
    fig = px.bar(results_top, x="Score", text='Score', y="Topic", orientation='h',
             color='Topic', color_continuous_scale=px.colors.sequential.Redor)

    # Define layout options
    fig.update_layout(yaxis_type='category', yaxis_categoryorder = 'total ascending',
                   plot_bgcolor="white", uniformtext_minsize=7, uniformtext_mode='hide', 
                   width = 800, height = 300)
    fig.update(layout_coloraxis_showscale=False)
    
    # Format text
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
  
       
    # Update axis
    fig.update_yaxes(ticks="outside", tickcolor='white', ticklen=8)
    fig.update_xaxes(matches=None, ticks="outside",)
    
    return fig, first_topic, second_topic, third_topic
    
    
# =============================================================================
# Words per Top Topics
# =============================================================================

# Table with top words per topic    
def get_topic_words (model, dictionary):
    # Create table with top n words per topic
    n_words = 20
    topic_words = pd.DataFrame({})

    for i, topic in enumerate(model.get_topics()):
        top_feature_ids = topic.argsort()[-n_words:][::-1]
        feature_values = topic[top_feature_ids]
        words = [dictionary[id] for id in top_feature_ids]
        topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
        topic_words = pd.concat([topic_words, topic_df], ignore_index=True)
     
    return topic_words
    

# Wordcloud and plot for top 3 topics
def words_vis (topic, topic_words, colors_fig, colors_wc):
    # Table with 10 results
    t_words = topic_words[(topic_words['topic'] == topic)].head(10)
    t_words['word'] = t_words['word'].str.capitalize()
    t_words['value'] = t_words['value']*100
    
    # Plots
    fig = px.bar(t_words, x="value", y="word", orientation='h', text='value',
              color_discrete_sequence = colors_fig,
              labels={"value": "Word Frequency in Topic (%)",
                     "word": ""})
    fig.update_layout(yaxis_categoryorder = 'total ascending', uniformtext_minsize=7, uniformtext_mode='show', plot_bgcolor="white")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='auto')
    fig.update_layout(autosize=False, width=500, height=260, margin=dict(l=20, r=200, t=3, b=20))
    fig.update_yaxes(ticks="outside", tickcolor='white', ticklen=8)
    fig.update_xaxes(showticklabels=False)
    
    # WordClouds 
    text1 = topic_words[(topic_words['topic'] == topic)]
    text2 = text1['word']
    text3 = ' '.join(text2)
    wd = WordCloud(max_font_size=40, max_words=100, background_color="white",
                          color_func=lambda *args, **kwargs:(colors_wc)).generate(text3)
    
    plt.figure(figsize=(300, 500))
    plt.imshow(wd, interpolation='bilinear')
    plt.axis("off")
    plt.show()
      
    return fig, wd
    
   
# # =============================================================================
# # Recommendations
# # =============================================================================

# Create index for sim_pd
def supervisors():    
    supervisors  = [
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2778&studyProgramId=1">Helmut<br>Anheier</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2799&studyProgramId=1">Joanna<br>Bryson</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2748&studyProgramId=1">Basak<br>Cali</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2749&studyProgramId=1">Luciana<br>Cingolani</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2800&studyProgramId=1">Cathryn<br>Costello</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2762&studyProgramId=1">Mark<br>Dawson</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2751&studyProgramId=1">Christian<br>Flachsland</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2758&studyProgramId=1">Anita<br>Gohdes</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2753&studyProgramId=1">Lukas<br>Graf</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2754&studyProgramId=1">Mark<br>Hallerberg</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2755&studyProgramId=1">Gerhard<br>Hammerschmid</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2756&studyProgramId=1">Anke<br>Hassel</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2759&studyProgramId=1">Lion<br>Hirth</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2760&studyProgramId=1">Thurid<br>Hustedt</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2761&studyProgramId=1">Leonardo<br>Iacovone</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2762&studyProgramId=1">Markus<br>Jachtenfuchs</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2763&studyProgramId=1">Slava<br>Jankin</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2764&studyProgramId=1">Mark<br>Kayser</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2765&studyProgramId=1">Michaela<br>Kreyenfeld</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2766&studyProgramId=1">Johanna<br>Mair</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2767&studyProgramId=1">Sebastien<br>Mena</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2768&studyProgramId=1">Alina<br>Mungiu-Pippidi</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2769&studyProgramId=1">Simon<br>Munzert</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2770&studyProgramId=1">Ronny<br>Patz</a>',
                    '<a href="hhttps://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2771&studyProgramId=1">Christine<br>Reh</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2772&studyProgramId=1">Andrea<br>Roemmele</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2773&studyProgramId=1">Mujaheed<br>Shaikh</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2774&studyProgramId=1">Dennis<br>Snower</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2775&studyProgramId=1">Daniela<br>Stockman</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2776&studyProgramId=1">Christian<br>Traxler</a>',
                    '<a href="https://mystudies.hertie-school.org/en/course-directory.php?p_id=350&action=show&courseId=2777&studyProgramId=1">Kai<br>Wegrich</a>'
                    ]
    
    return supervisors


# Create table with supervisors, using results from comparison with corpus
def get_topics_doc(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
   
    return res


# =============================================================================
# Similarities
# =============================================================================

# Similarity Matrix
def sim_matrix(sim_model, ud_bow, supervisor_list):
    sim = sim_model[ud_bow]
    sim_pd = pd.DataFrame(sim)
    sim_pd.columns = ['Similarity']
    sim_pd.index = supervisor_list
    
    return sim_pd


# Table with recommendations
def recommend_df (document_topic, topic, sim_pd, supervisor_list):
    # Supervisor recommendations per topic
    r_df = pd.DataFrame(document_topic[topic])
    r_df.columns = ['topic_i']
    r_df.index = supervisor_list
    
    # Concatenate with similarity scores
    c_df = pd.concat([r_df, sim_pd], axis=1)
    c_df = c_df[c_df.topic_i > 0.5]
    c_df = c_df.nlargest(5,['Similarity'])
    c_df = c_df[c_df.Similarity > 0].head(5)
    c_df.sort_values(['Similarity'], ascending=True, inplace=True)
    c_df['Similarity'] = c_df['Similarity']*100 
    
    return c_df


# =============================================================================
# Supervisor Recommendation Plot
# =============================================================================

# Plot with supervisor recommendations per top 3 topics
def super_vis(first_topic, second_topic, third_topic, recom_1, recom_2, recom_3):
    fig = go.Figure()
    
    # Add traces - One per topic
    fig.add_trace(go.Bar(name='Topic '+str(first_topic), y=recom_1.index, x=recom_1['Similarity'], # first_topic
                             text=recom_1['Similarity'], orientation='h',
                             marker={'color': recom_1['Similarity'], 'colorscale': 'Redor'}))
    
    fig.add_trace(go.Bar(name='Topic '+str(second_topic), y=recom_2.index, x=recom_2['Similarity'], # second
                             text=recom_2['Similarity'],orientation='h', visible=False,
                             marker={'color': recom_2['Similarity'], 'colorscale': 'Pinkyl'}))
    
    fig.add_trace(go.Bar(name='Topic '+str(third_topic), y=recom_3.index, x=recom_3['Similarity'], # third
                             text=recom_3['Similarity'], orientation='h', visible=False,
                             marker={'color': recom_3['Similarity'],'colorscale': 'Oryel'}))
    
    fig.update_xaxes(matches=None, title_text='Similarity Score', ticks="outside")
    fig.update_yaxes(matches=None, title_text='Supervisor', ticks="outside", tickcolor='white', ticklen=2)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
    
    
    # Layout options
    fig.update_layout(
        width=680,
        height=400,
        template="plotly_white")
    
    fig.update_layout(
        annotations=[
            dict(text="", showarrow=False,
            x=0, y=1, yref="paper", align="left")])
    
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                showactive = True,
                x = -0.1,
                xanchor = 'left',
                y = 1.3,
                yanchor = 'top',
                buttons=list([
                    dict(label='Topic '+str(first_topic),
                         method="update",
                         args=[{"visible": [True, False, False]},
                                ]),
                    dict(label='Topic '+str(second_topic),
                         method="update",
                         args=[{"visible": [False, True, False]},
                                ]),
                    dict(label='Topic '+str(third_topic),
                         method="update",
                         args=[{"visible": [False, False, True]},
                                ])]))])
    
    return fig


if __name__ == "__main__":
    freeze_support()
