# Import packages 
import gensim
from mod1_data import get_data, corpus_lemma, corpus_stem
from gensim import similarities
from multiprocessing import freeze_support

# =============================================================================
# Load Data
# =============================================================================

# Get Corpus

corpus_list = get_data()

dict_lemma, corpus_lemma = corpus_lemma(corpus_list)
dict_stem, corpus_stem = corpus_stem(corpus_list)



# =============================================================================
# LDA
# =============================================================================

lda_model = gensim.models.LdaMulticore(corpus_lemma, 
                                id2word = dict_lemma,
                                num_topics = 6, # best results with 5 topics
                                random_state = 123, # seed for consistency
                                passes = 2000, 
                                alpha = 'symmetric', 
                                chunksize = 31, minimum_probability = 0.01)

lda_model.save('lda_model2')

# =============================================================================
# Similarities
# =============================================================================

sim_model = similarities.SparseMatrixSimilarity(corpus_stem, len(dict_stem.token2id))

sim_model.save('sim_model', separately = None)


if __name__ == "__main__":
    freeze_support()   