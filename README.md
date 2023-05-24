# **Supervisor Recommendation Tool - Final Report**

## Project Overview
Our project introduces a sophisticated method for classifying and matching Master's thesis supervision plans using Natural Language Processing (NLP) techniques. We have utilized Python libraries such as the Natural Language Toolkit (NLTK) and Gensim to employ topic modeling techniques on a corpus composed of 31 supervision plans. These plans were sourced from the Hertie School’s MyStudies database, converted into .txt files, and then used to generate per-document topic distributions.

The primary output of our project is a model that not only identifies and categorizes different topics but also creates a similarity matrix, illustrating the degree to which supervision plans are related. This model is then applied to a student's research proposal to extract the best matches between the student's interests and the topics identified in our model.

Upon identifying the most suitable topic, we recommend the supervisors best aligned with that topic, ranked by their similarity score with the student's proposal. Our topic modeling technique rests on the premise that every document is a blend of distinct topics, and each topic is a collection of related words.

The ultimate aim of our project is to uncover topics that encapsulate research interests and methodologies effectively, as these are crucial in pairing students and supervisors. For a concise overview of our work, you may watch our four-minute video presentation or try our model yourself by clicking here.

## Code Structure
In response to the feedback received on our Midterm Report, we structured our project into four primary modules:

mod0_data: This module lays out functions and parameters for preprocessing text from both the corpus and user input. Processes such as lemmatization, stemming, removal of stopwords, non-alphabetic characters, and punctuation are included in this module.

mod1_model: This module constructs and saves the Latent Dirichlet Allocation (LDA) model used by our application, as well as the similarity matrix, using the preprocessed supervision plans as input.

mod2_vis: This module is responsible for generating visual outputs such as graphs, word clouds, and other graphical representations that are showcased by our application.

mod3_app: This module contains our Streamlit application. It is worth noting that the Streamlit app is hosted on this GitHub repository, and all files in the master root are included as they are necessary for the smooth running of the app.

## Running the Application
To execute our application, please ensure you have installed all packages listed in the requirements.txt file. Additionally, objects such as corpus_lemma, corpus_lemma.index, dict_lemma, corpus_stem, corpus_stem.index, dict_stem, sim_model, and lda_model, located in the main GitHub folder, are also required for the application to function correctly.`lda_model.id2word` and `lda_model.state`) -- all located in the main GitHub folder -- are also necessary to run the application. 


## References
- Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
-[Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.](https://www.jmlr.org/papers/v3/blei03a)
- [Machine Learning Plus 'Gensim Tutorial – A Complete Beginners Guide'](https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda)
- [Machine Learning Plus 'Topic modeling visualization – How to present the results of LDA models?'](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
-[Ganesan,Kavita. 'All you need to know about text preprocessing for NLP and Machine Learning'](https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html)
- [Li, Susan.'Topic Modelling in Python with NLTK and Gensim ]('https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21#:~:text=In%20this%20post%2C%20we%20will,a%20document%2C%20called%20topic%20modelling.&text=Research%20paper%20topic%20modelling%20is,of%20papers%20in%20a%20corpus)
- [Documenting Python APIs with docstrings](https://developer.lsst.io/python/numpydoc.html#py-docstring-module-structure)
- [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)

