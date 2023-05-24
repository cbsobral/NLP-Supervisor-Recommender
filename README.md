# **Supervisor Recommendation Tool - Final Report**

## Project Overview
The project leverages Natural Language Processing (NLP) tools to classify and match Master's thesis supervision plans. Python libraries such as the Natural Language Toolkit (NLTK) and Gensim have been employed to apply topic modeling techniques on a corpus comprising 31 supervision plans. These plans, sourced from the Hertie School’s MyStudies database and converted into .txt files, are used to generate per-document topic distributions.

A model has been developed as the core output of the project. It identifies and categorizes different topics and creates a similarity matrix, demonstrating the degree to which supervision plans are interrelated. This model can be applied to a student's research proposal to find the best matches between the student's interests and the topics identified by the model.

Once the most suitable topic is identified, supervisors best aligned with that topic are recommended. They are ranked based on their similarity score with the student's proposal. The topic modeling technique rests on the premise that every document is a blend of distinct topics, and each topic is a collection of related words.

The primary aim of the project is to uncover topics that encapsulate research interests and methodologies effectively, as these are pivotal in pairing students and supervisors. A four-minute video presentation provides an overview of the project, or the model can be tried out by clicking here.

## Code Structure
Following the feedback received on the Midterm Report, the project has been structured into four primary modules:

mod0_data: This module outlines functions and parameters for preprocessing text from both the corpus and user input. Processes include lemmatization, stemming, removal of stopwords, non-alphabetic characters, and punctuation.

mod1_model: This module is responsible for constructing and saving the Latent Dirichlet Allocation (LDA) model used by the application, as well as the similarity matrix. It uses preprocessed supervision plans as input.

mod2_vis: This module generates visual outputs such as graphs, word clouds, and other graphical representations displayed by the application.

mod3_app: This module houses the Streamlit application. It should be noted that the Streamlit app is hosted on the GitHub repository, and all files in the master root are included as they are necessary for the smooth running of the app.

## Running the Application
To execute the application, it is necessary to install all packages listed in the requirements.txt file. Additionally, objects such as corpus_lemma, corpus_lemma.index, dict_lemma, corpus_stem, corpus_stem.index, dict_stem, sim_model, and lda_model, located in the main GitHub folder, are required for the application to operate correctly.


## References
- Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
-[Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.](https://www.jmlr.org/papers/v3/blei03a)
- [Machine Learning Plus 'Gensim Tutorial – A Complete Beginners Guide'](https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda)
- [Machine Learning Plus 'Topic modeling visualization – How to present the results of LDA models?'](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
-[Ganesan,Kavita. 'All you need to know about text preprocessing for NLP and Machine Learning'](https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html)
- [Li, Susan.'Topic Modelling in Python with NLTK and Gensim ]('https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21#:~:text=In%20this%20post%2C%20we%20will,a%20document%2C%20called%20topic%20modelling.&text=Research%20paper%20topic%20modelling%20is,of%20papers%20in%20a%20corpus)
- [Documenting Python APIs with docstrings](https://developer.lsst.io/python/numpydoc.html#py-docstring-module-structure)
- [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)

