import nltk
import gensim
from gensim import models
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from multiprocessing import freeze_support


# =============================================================================
# Pre-processing Tools
# =============================================================================

# Stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

ps = PorterStemmer()
wnt = nltk.WordNetLemmatizer() # Lemmatizer

stoplist = stopwords.words('english') # Stopwords

# Define additional stopwords in a string
additional_stopwords = """http ask ha ox mcc christine task discussion chapter submit indicate io bot los angeles 
                          semester auto author colomb affair ly bit client database evidence willing note 
                          report william williams stanford www mair dawson ercas also hertie question professor 
                          title school session topics project partner practice plan see supervise
                          thesis issue student topic supervision university lab mia mpp org etc"""   
                         
stoplist += additional_stopwords.split() # Join both lists


# =============================================================================
# Corpus Pre-processing
# =============================================================================

# Base functions 
# Convert token to lowercase and lemmatize
def lemma_token(token):
  return wnt.lemmatize(token.lower())

# Convert token to lowercase and stem
def stem_token(token):
  return ps.stem(token.lower())

# Evaluate whether or not to retain `token`
def filter_token(token):
    token = token.lower()
    return token.isalpha()

# Lemmatize
def corpus_lemma(corpus_list):
    
    # Filter and lemmatize
    documents=[[lemma_token(token) 
            for token in corpus_list.words(fileids=[fileid])
            if filter_token(token)]
            for fileid in corpus_list.fileids()]           

    corpus =[[token for token in doc 
           if len(token) > 1
           and token not in stoplist]
           for doc in documents]

    
    # Create dictionary and bow for documents
    dict_lemma = gensim.corpora.Dictionary(corpus)
    corpus_bow = [dict_lemma.doc2bow(document) for document in corpus]
    tfidf = models.TfidfModel(corpus_bow) 
    corpus_lemma = tfidf[corpus_bow]
    return dict_lemma, corpus_lemma

# Stem 
def corpus_stem (corpus_list):
    # Filter and stem
    documents=[[stem_token(token) 
            for token in corpus_list.words(fileids=[fileid])
            if filter_token(token)]
            for fileid in corpus_list.fileids()]           

    corpus = [[token for token in doc 
            if len(token) > 1
            and token not in stoplist]
            for doc in documents]
   
    dict_stem = gensim.corpora.Dictionary(corpus)
    corpus_bow = [dict_stem.doc2bow(document) for document in corpus]
    tfidf = models.TfidfModel(corpus_bow) 
    corpus_stem = tfidf[corpus_bow]

    return dict_stem, corpus_stem


# =============================================================================
# User's Input Pre-processing
# =============================================================================
    
# Lemmatize
def ud_lemma (ud_text, dict_lemma):
    ud_tokens = nltk.word_tokenize(ud_text)
    ud_filter = [token for token in ud_tokens if token not in stoplist and token.isalpha() and len(token) > 1]
    ud_lemma = ' '.join([wnt.lemmatize(token) for token in ud_filter]) # ud can be either `ud_text` or `ud_path`
    ud_tk_lemma = nltk.word_tokenize(ud_lemma) 
    ud_bow_lemma = dict_lemma.doc2bow(ud_tk_lemma)
    return ud_bow_lemma
    
# Stem
def ud_stem (ud_text, dict_stem):
    ud_tokens = nltk.word_tokenize(ud_text) 
    ud_filter = [token for token in ud_tokens if token.isalpha() and len(token) > 1]
    ud_stem = ' '.join([ps.stem(token) for token in ud_filter]) 
    ud_tk_stem = nltk.word_tokenize(ud_stem) 
    ud_bow_stem = dict_stem.doc2bow(ud_tk_stem)
    return ud_bow_stem


if __name__ == "__main__":
    freeze_support()