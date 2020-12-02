# **Supervisor Recommendation Tool - Midterm Report**
## **The Project**

In this script, we use the "Natural Language Toolkit" ([nltk](https://www.nltk.org/)) and  [Gensim](https://radimrehurek.com/gensim/) packages to employ topic modeling techniques for classifying the content of the 31 Master Thesis Colloquia supervision plans – downloaded from the Hertie School’s Mystudies database, and converted into .txt files – into different topics.

These supervision plans compose the “corpus” of the work, from which we derive a model that generates per-document topic distributions.

Then, we apply the model to the text of a student's research proposal, to extract the best matches between their interests and the topics of our model.

Finally, we recommend the supervisors that match the student's most prominent topic interest.

Topic modeling is based on the assumption that each document in a text is a combination of certain topics and that each topic is a combination of related words.

The objective of this script is to extract the underlying topics from the collection of Master Thesis Colloquia supervision plans and compare them to the underlying topics of a student’s research proposal.

Due to the scope of our project, we aim to unveil topics that best represent research interests and/or research methodologies, as these are the main criteria for matching students and supervisors.
## Current Development
In the [topic_choices.ipynb](https://github.com/cbsobral/python/blob/master/topic_choices.ipynb) file, our progress can be tracked (we recommend using Google Colaboratory to view it). This file contains brief explanations on the steps we took to develop and elect a topic model.

Although we experimented with both LDA and TD-IDF, we are leaning towards using the former, because so far, it yielded more accurate matches. 

With [streamlit](https://www.streamlit.io/), we are developing the user's interface, where student's can input their research proposals and receive our output. We are still in the initial stages with this, but the script can be found in the [app.py](https://github.com/cbsobral/python/blob/master/app.py) file. You can check how this interface currently looks like accessing the file [st-app-screencast.mp4](https://github.com/cbsobral/python/blob/master/st-app-screencast.mp4).

## Next Steps 
In the next weeks, we plan to test our model with a few research proposals, identify opportunities for enhancement of accuracy in topic matching and improve the code, as needed.

We also intend to develop the user's interface.

## References
- Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
-[Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.](https://www.jmlr.org/papers/v3/blei03a)
- [Machine Learning Plus 'Gensim Tutorial – A Complete Beginners Guide'](https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda)
- [Machine Learning Plus 'Topic modeling visualization – How to present the results of LDA models?'](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
-[Ganesan,Kavita. 'All you need to know about text preprocessing for NLP and Machine Learning'](https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html)
- [Li, Susan.'Topic Modelling in Python with NLTK and Gensim ]('https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21#:~:text=In%20this%20post%2C%20we%20will,a%20document%2C%20called%20topic%20modelling.&text=Research%20paper%20topic%20modelling%20is,of%20papers%20in%20a%20corpus)

