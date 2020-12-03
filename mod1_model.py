"""
This module generates and saves the LDA model employed by the app, as well as 
the similarity matrix, using the preprocessed supervision plans as input.
    
"""

# Import packages 
from mod0_data import corpus_lemma, corpus_stem 
import gensim
from gensim import models, corpora, similarities
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords, wordnet
from multiprocessing import freeze_support

# =============================================================================
# Data
# =============================================================================

# Get .txt files from local folder and transform into corpus
url = path = ('c:\\Users\\carol\\Desktop\\Fall_2020\\Python\\streamlit\\data')
corpus_list = PlaintextCorpusReader(url, '.*txt')  # import all files ending in 'txt'

# Apply lemmatize function 
dict_lemma, corpus_lemma = corpus_lemma(corpus_list)

# Save lemmatized corpus and dictionary
dict_lemma.save('dict_lemma')
corpora.MmCorpus.serialize('corpus_lemma', corpus_lemma)

# Apply stem function 
dict_stem, corpus_stem = corpus_stem(corpus_list)

# Save stemmed corpus and dictionary
dict_stem.save('dict_stem')
corpora.MmCorpus.serialize('corpus_stem', corpus_stem)


# =============================================================================
# LDA
# =============================================================================

# Generate LDA model with lemmatized documents
lda_model = gensim.models.LdaModel(corpus_lemma, 
                                    id2word = dict_lemma,
                                    num_topics = 6, 
                                    random_state = 123, # seed for consistency
                                    passes = 2000, 
                                    alpha = 'symmetric', 
                                    chunksize = 31)

# Save LDA model
lda_model.save('lda_model')

# =============================================================================
# Similarity Matrix
# =============================================================================

# Create similarity matrix with stemmed documents
sim_model = similarities.SparseMatrixSimilarity(corpus_stem, len(dict_stem.token2id))

# Save similarity matrix
sim_model.save('sim_model', separately = None)


if __name__ == "__main__":
    freeze_support()   