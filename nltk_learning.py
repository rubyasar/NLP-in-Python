# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:46:21 2021

@author: rubyasar
"""
import nltk

nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize     import sent_tokenize
from nltk.tokenize     import word_tokenize
from nltk.corpus       import stopwords
from nltk.stem         import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk              import pos_tag
from nltk.probability  import FreqDist


text="CM named Yoshua Bengio, Geoffrey Hinton, and Yann LeCun recipients of the 2018 ACM A.M. Turing Award for conceptual and engineering breakthroughs that have made deep neural networks a critical component of computing. Bengio is Professor at the University of Montreal and Scientific Director at Mila, Quebec’s Artificial Intelligence Institute; Hinton is VP and Engineering Fellow of Google, Chief Scientific Adviser of The Vector Institute, and University Professor Emeritus at the University of Toronto; and LeCun is Professor at New York University and VP and Chief AI Scientist at Facebook."

""" Sentence Tokenizer """
print(sent_tokenize(text))

""" Words Tokenizer """
word_token=word_tokenize(text)
print(word_token)

""" List of StopWords in English from NLTK """

en_stopwords=set(stopwords.words("english"))
print(en_stopwords)

""" Removal of StopWords from Text"""

no_stopword_token=[word for word in word_token if word not in en_stopwords]
print(word_token)
print(no_stopword_token)

""" Word Stemming to Root Words """

stemmed_token=[]
for word in no_stopword_token:
    stemmed_token.append(PorterStemmer().stem(word))
print(word_token)
print(no_stopword_token)
print(stemmed_token)    

""" Word Lemmatizing to Base Words """

lemmatized_token=[]
for word in no_stopword_token:
    lemmatized_token.append(WordNetLemmatizer().lemmatize(word))
print(word_token)
print(no_stopword_token)
print(stemmed_token) 
print(lemmatized_token)    

""" Tagging Parts of Speech """
print(pos_tag(word_token))

""" Frequency Distribution of words """
freq_dist=FreqDist(word_token)
freq_dist.plot(30,cumulative=False)

freq_dist=FreqDist(lemmatized_token)
freq_dist.plot(30,cumulative=False)













