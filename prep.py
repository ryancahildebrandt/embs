#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:45:41 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import nltk
import pandas as pd
import random
import re
import streamlit as st
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

random.seed(42)

spell = SpellChecker()

@st.cache(persist = True)
def prep_load():
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('omw-1.4')


def prep_lower(in_text):
  return [i.lower() for i in in_text]

def prep_punct(in_text):
  return [i.translate(str.maketrans('', '', string.punctuation)) for i in in_text]

def prep_stop(in_text):
  t = []
  for i in in_text:
    t.append(" ".join([j for j in word_tokenize(i) if j not in stopwords.words()]))
  return t

def prep_lemma(in_text):
  t = []
  for i in in_text:
    t.append(" ".join([WordNetLemmatizer().lemmatize(k) for k in word_tokenize(i)]))
  return t

def prep_stem(in_text):
  t = []
  for i in in_text:
    t.append(" ".join([PorterStemmer().stem(k) for k in word_tokenize(i)]))
  return t

def prep_spell(in_text):
  t = []
  for i in in_text:
    t.append(" ".join([j if j in string.punctuation else spell.correction(j) for j in word_tokenize(i)]))
  return t

clause_reg = "[\.\!\\\/\|,\?\;\:_\-=+]"
clause_words = ["and","about","but","so","because","since","though","although","unless","however","until"]
clause_sep = f"{clause_reg}{' | '.join(clause_words)}".replace("] ", "]")

def prep_clause(in_text):
  t = []
  for i in in_text:
    for j in re.split(clause_sep, i, flags = re.IGNORECASE):
      if j != "":
        t.append(str.strip(j))
  return t

def prep_ex(in_text, func):
  out = pd.DataFrame.from_dict({"Before" : in_text, "After" : func(in_text)})
  return out