# **Supervisor Recommendation Tool - Final Report**
## **The Project**

In this project, we use the "Natural Language Toolkit" ([nltk](https://www.nltk.org/)) and  [Gensim](https://radimrehurek.com/gensim/) packages to employ topic modeling techniques for classifying the content of the 31 Master Thesis Colloquia supervision plans – downloaded from the Hertie School’s Mystudies database, and converted into .txt files – into different topics.

These supervision plans compose the “corpus” of the work, from which we derive a model that generates per-document topic distributions. 

We also generated a similarity matrix, that displays how supervision plans are similar to each other.

Then, we apply the model to the text of a student's research proposal, to extract the best matches between their interests and the topics of our model.

Finally, based on the selection of the best topic, we recommend the supervisors within that topic, ordered by their similarity score with the student's proposal.

Topic modeling is based on the assumption that each document in a text is a combination of certain topics and that each topic is a combination of related words.

The objective of this script is to extract the underlying topics from the collection of Master Thesis Colloquia supervision plans and compare them to the underlying topics of a student’s research proposal.

Due to the scope of our project, we aim to unveil topics that best represent research interests and/or research methodologies, as these are the main criteria for matching students and supervisors.

If you are interested in a 4 minutes presentation of our work, check out our [video](https://www.youtube.com/watch?v=mOxyVOVVxeA).
You can also test our model yourself, clicking [here](https://share.streamlit.io/cbsobral/python/mod3_app.py).

## Installation Requirements

## The Code

Following up on the feedback we received in our Midterm Report, we have divided our script into modules, and elaborated documentation for each function.

We are working with 4 modules:

### mod0_data
This module defines functions and parameters that will be employed in preprocessing the text (for the corpus and for the user's input), such as Lemmatization, stem, removal of stopwords, removal of non-alphabetic numbers and punctuation.

### mod1_model
This module generates and saves the LDA model employed by the app, as well as the similarity matrix, using the preprocessed supervision plans as input.

### mod2_vis
This module generates the graphs, wordclouds and other graphical representations that will be displayed by the app.


### mod3_app
This module creates the webpage where our app is hosted using streamlit.

Finally, it should be noted that streamlit is connected to this github master folder. So, all files in the master root are there because they are necessary for the app to run smoothly.


## Behind the Code: What changed since Midterm
Our progress can be traced back in the folder [initial](https://github.com/cbsobral/python/initial), where you can still see our progress up to the [Midterm Report](https://github.com/cbsobral/python/blob/master/Midterm%20Report.MD).
For our topic model, after extensive comparison of models, we found that a TD-IDF weighted DFM provided a best selection of words representative to each topic. The topic model itself employs a Latent Dirichlet Allocation (LDA).

We found out that the model was very sensitive to our choice of stopwords to be removed, as well as to the number of topics we employed. We tested the accuracy of our predictions using master thesis proposals of our friends, and then asking them whether the recommendations were consistent with their own perception, since all second year students have just gone through this process of selecting a supervisor. 

Back in the midterm, we were only working with the topic modeling idea. But then, it occured to us that we could improve our output recommendations significantly if we crossed each supervisor plan to the user's input. We accomplished that using a similarity matrix, that first classifies supervision plans in relation to one another, and displays how supervision plans are associated to each topic. Once we have input from the user, we use this matrix to check how the plans of a given topic compare to the user's input.

With the idea of combining the two strategies(i.e. topic modeling and similarity matrix) in mind, we again revisited the number of topics we elected. We found that reducing the number of topics in the model provided for better recommendations in the end. 

With [streamlit](https://www.streamlit.io/), we developed the user's interface, where student's input their research proposals and receive our output. 



## References
- Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
-[Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.](https://www.jmlr.org/papers/v3/blei03a)
- [Machine Learning Plus 'Gensim Tutorial – A Complete Beginners Guide'](https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda)
- [Machine Learning Plus 'Topic modeling visualization – How to present the results of LDA models?'](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
-[Ganesan,Kavita. 'All you need to know about text preprocessing for NLP and Machine Learning'](https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html)
- [Li, Susan.'Topic Modelling in Python with NLTK and Gensim ]('https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21#:~:text=In%20this%20post%2C%20we%20will,a%20document%2C%20called%20topic%20modelling.&text=Research%20paper%20topic%20modelling%20is,of%20papers%20in%20a%20corpus)
- [Documenting Python APIs with docstrings](https://developer.lsst.io/python/numpydoc.html#py-docstring-module-structure)
- [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)

