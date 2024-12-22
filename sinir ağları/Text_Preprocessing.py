from nltk.stem import PorterStemmer
import spacy
import pandas as pd

# İLK BAĞLANTILAR
nlp = spacy.load('en_core_web_sm')  # Spacy modelini yükler (İngilizce dil modeli)
stemmer = PorterStemmer()  # NLTK'nın PorterStemmer'ını kullanarak kök bulma işlemi yapılacak
stop_words = nlp.Defaults.stop_words  # Spacy'nin varsayılan stopwords (durak kelimeler) listesi

# 1. METİN İŞLEME

# NOKTALAMALARI TEMİZLEME
def separate_punc(doc_text):
    # Eğer bir token (kelime) yeni satır veya noktalama işareti değilse, onu alır
    # Spacy ile her kelimeyi alır ve yalnızca anlamlı kelimeleri (noktalama işareti olmayanları) seçer
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

# DURAK KELİMELERİNİ (STOPWORDS) TEMİZLEME
def remove_stopwords(text):
    # Verilen metni küçük harfe dönüştürür, sonra stopwords listesine dahil olan kelimeleri temizler
    words = text.lower().split()  # Metni küçük harfe dönüştürüp kelimelere ayırır
    words = [w for w in words if w not in stop_words]  # Stopwords listesinde olmayan kelimeleri alır
    return ' '.join(words)  # Temizlenmiş kelimeleri birleştirip metin olarak döndürür

# KÖK BULMA (STEMMING) - NLTK
def stem_text(text):
    # NLTK'nın PorterStemmer'ı kullanılarak her kelime köküne indirilir
    return ' '.join([stemmer.stem(word) for word in text.split()])  # Kelimeleri köklerine indirir ve birleştirir

# LEMMATİZASYON - SPACY
def lemmatize_text(text):
    # Spacy'nin dil modeli ile kelimeleri lemmatize (köken haline getirme) eder
    doc = nlp(text)  # Metni Spacy ile işler
    return ' '.join([token.lemma_ for token in doc])  # Her token'ın lemmatize (köken) halini alır ve birleştirir

# BİRLEŞTİRİLMİŞ BİR ÖN İŞLEME FONKSİYONU
def preprocess_text(text):
    # Öncelikle metni küçük harfe dönüştürür, stop kelimeleri çıkarır, kök bulma ve lemmatizasyon işlemi yapar
    nlp = spacy.load('en_core_web_sm')  # Spacy modelini tekrar yükler
    stemmer = PorterStemmer()  # NLTK'nın PorterStemmer'ını tekrar başlatır
    stop_words = nlp.Defaults.stop_words  # Spacy'nin stopwords listesini tekrar alır

    words = text.lower().split()  # Metni küçük harfe dönüştürüp kelimelere ayırır
    # Kök bulma ve stop kelimelerinden arındırma işlemi yapılır ve ardından lemmatizasyon yapılır
    return " ".join([token.lemma_ for token in nlp(" ".join([stemmer.stem(word) for word in words if word not in stop_words]))])
