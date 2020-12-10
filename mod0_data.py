
"""
This module defines functions and parameters that will be employed for 
preprocessing the text (for the corpus and the user's input), such as 
Lemmatization, stem, removal of stopwords, removal of non-alphabetic 
characters, and punctuation.
    
"""

import nltk
import gensim
from gensim import models
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from multiprocessing import freeze_support


# =============================================================================
# Pre-processing Tools
# =============================================================================

# Download stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

ps = PorterStemmer() # stemmer
wnt = nltk.WordNetLemmatizer() # lemmatizer

stoplist = stopwords.words('english') # stopwords

# Define additional stopwords in a string
additional_stopwords = """http ask ha ox mcc christine task discussion chapter submit indicate io bot los angeles 
                          semester auto author colomb affair ly bit client database evidence willing note 
                          report william williams stanford www mair dawson ercas also hertie question professor 
                          title school session topics project partner practice plan see supervise
                          thesis issue student topic supervision university lab mia mpp org etc"""   
                         
stoplist += additional_stopwords.split() # join both lists


# =============================================================================
# Corpus Pre-processing
# =============================================================================

# Base functions 
# Convert token to lowercase and lemmatize
def lemma_token(token):
    """
    Converts token to lowecase and lemmatizes.

    Parameters
    ----------
    token: str 
    A token (usually a word) in the corpus.

    Returns
    -------
    Token with lowercase characters and lemmatized root.

    """
    return wnt.lemmatize(token.lower())


# Convert token to lowercase and stem
def stem_token(token):
    """
    Converts token to lowecase and stem.

    Parameters
    ----------
    token : str 
    A token (usually a word) in the corpus.

    Returns
    -------
    Token with lowercase characters and stemmed root.

    """
    return ps.stem(token.lower())


# Evaluate whether or not to retain `token`
def filter_token(token):
    """
    Transforms token in lowercase and returns only alphabetic characters.

    Parameters
    ----------
    token : str 
    A token (usually a word) in the corpus.

    Returns
    -------
    Token with lowercase characters and only alphabetic characters.

    """
    token = token.lower()
    
    return token.isalpha()


# Lemmatize
def corpus_lemma(corpus_list):
    """
    Creates lemmatized corpus and dictionary.

    Parameters
    ----------
    corpus_list : list of documents
        List of plaintext documents.

    Returns
    -------
    dict_lemma : corpora.dictionary.Dictionary
        Dictionary encapsulates the mapping between normalized words and their 
        integer ids. The main function is doc2bow, which converts a collection 
        of words to its bag-of-words representation: a list of (word_id, word_frequency) 
        2-tuples.
    corpus_lemma : corpora.mmcorpus.Mmcorpus  
        Corpus serialized using the sparse coordinate Matrix Market format, 
        after tf-idf weighting, lemmatization, removal of non alphabetic 
        characters, stopwords and words of less than 1 character.

    """
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
     """
    Creates stemmed corpus and dictionary.

    Parameters
    ----------
    corpus_list : list of documents
        List of plaintext documents.

    Returns
    -------
    dict_stem :corpora.dictionary.Dictionary
        Dictionary encapsulates the mapping between normalized words and their 
        integer ids. The main function is doc2bow, which converts a collection 
        of words to its bag-of-words representation: a list of (word_id, word_frequency) 
        2-tuples.
    corpus_stem : corpora.mmcorpus.Mmcorpus  
        Corpus serialized using the sparse coordinate Matrix Market format,
        tf-idf weighting of tokens stemmed, with alphabetic characters only, 
        composed of more than 1 character, and without stopwords.

    """
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
    """
    Preprocesses users input through lemmatization, lowercase, removal of 
    stopwords and tokens with less than 2 characters and creates a bag of 
    words.

    Parameters
    ----------
    ud_text : str
        Tokens produced by the user.
    dict_lemma : corpora.dictionary.Dictionary
        Dictionary encapsulates the mapping between normalized words and their 
        integer ids. The main function is doc2bow, which converts a collection 
        of words to its bag-of-words representation: a list of (word_id, word_frequency) 
        2-tuples.
    Returns
    -------
    ud_bow_lemma : list of (int, int) tuples
        List of (token_id, token_count) tuples of user input, after lemmatized, 
        removal of non alphabetic characters, stopwords and words of less than 
        2 characters.

    """
    ud_tokens = nltk.word_tokenize(ud_text)
    ud_filter = [token for token in ud_tokens if token not in stoplist and token.isalpha() and len(token) > 1]
    ud_lemma = ' '.join([wnt.lemmatize(token) for token in ud_filter]) 
    ud_tk_lemma = nltk.word_tokenize(ud_lemma) 
    ud_bow_lemma = dict_lemma.doc2bow(ud_tk_lemma)
    
    return ud_bow_lemma
    
# Stem
def ud_stem (ud_text, dict_stem):
     """
    Preprocesses users input through stem, lowercase, removal of  tokens with 
    less than 2 characters and creates a bag of words.

    Parameters
    ----------
    ud_text : list of str
        List of tokens produced by the user.
    dict_stem : corpora.dictionary.Dictionary
        Dictionary encapsulates the mapping between normalized words and their 
        integer ids. The main function is doc2bow, which converts a collection 
        of words to its bag-of-words representation: a list of (word_id, word_frequency) 
        2-tuples.
    Returns
    -------
    ud_bow_stem : list of (int, int) tuples
        List of (token_id, token_count) tuples of user input, after stem, 
        removal of non alphabetic characters, stopwords, and of words of less 
        than 2 characters.
        
     """
     ud_tokens = nltk.word_tokenize(ud_text) 
     ud_filter = [token for token in ud_tokens if token not in stoplist and token.isalpha() and len(token) > 1]
     ud_stem = ' '.join([ps.stem(token) for token in ud_filter]) 
     ud_tk_stem = nltk.word_tokenize(ud_stem) 
     ud_bow_stem = dict_stem.doc2bow(ud_tk_stem)
     
     return ud_bow_stem


if __name__ == "__main__":
    freeze_support()