import pandas as pd
import numpy as np
# FRE SCORE
from textstat import flesch_reading_ease
# SENTIMENT ANALYSIS USING VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# POS TAGGING
import spacy
# VECTORISING TEXT AND CREATING PIPELINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
# COSINE SIMILARITY BETWEEN REVIEWS
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np

# SpaCy modelini yükleyelim
nlp = spacy.load('en_core_web_sm')

# 1. Okunabilirlik Skoru (FRE SCORE)
def add_readability_score(df):
    # Flesch Okunabilirlik Endeksi (FRE) hesaplanır ve her incelemeye eklenir
    df['READABILITY_FRE'] = df['REVIEW_TEXT'].apply(
        lambda d: flesch_reading_ease(d))

# 2. Duygu Kategorisi ve Puan Kategorisi
def add_sentiment_category(df, threshold):
    # Duygu skoru pozitifse "pozitif", negatifse "negatif" olarak etiketleme işlemi yapılır
    def assign_sentiment_category(score):
        if score > threshold:
            return 'positive'
        else:
            return 'negative'

    df['SENTIMENT_CATEGORY'] = df['SENTIMENT_SCORE'].apply(
        assign_sentiment_category)

def add_rating_category(df, threshold):
    # Ürün puanları "pozitif" veya "negatif" olarak sınıflandırılır
    def assign_rating_category(rating):
        if rating > threshold:
            return 'positive'
        else:
            return 'negative'

    df['RATING_CATEGORY'] = df['RATING'].apply(assign_rating_category)

# 3. Uyum Kolonu (Rating ve Text Arasında)
def add_coherence_column(df):
    # İnceleme metninin duygu kategorisi ile ürün puanı kategorisinin uyumlu olup olmadığı kontrol edilir
    df['COHERENT'] = df['SENTIMENT_CATEGORY'] == df['RATING_CATEGORY']

# 4. VADER Duygu Skoru
def add_vader_sentiment_score(df):
    # VADER duygu analiz aracını kullanarak duygu skorları hesaplanır
    sid = SentimentIntensityAnalyzer()
    df['SENTIMENT_SCORE'] = df['REVIEW_TEXT'].apply(
        lambda d: sid.polarity_scores(d)['compound'])

# 5. Başlık Uzunluğu
def add_title_length(df):
    # İnceleme başlıklarının uzunluğu hesaplanır
    df['TITLE_LENGTH'] = df['REVIEW_TITLE'].apply(lambda d: len(d))

# 6. Dilbilgisel Etiketler (POS TAGS) - Fiiller, İsimler, Sıfatlar, Zarflar
def add_pos_tags(df):
    def count_pos(Pos_counts, pos_type):
        # Belirli bir dilbilgisel türün (isim, fiil, sıfat, zarf) sayısını döndürür
        pos_count = Pos_counts.get(pos_type, 0)
        return pos_count

    def pos_counts(text):
        # Her inceleme metni için dilbilgisel etiketlerin sayılması
        doc = nlp(text)
        Pos_counts = doc.count_by(spacy.attrs.POS)
        return Pos_counts

    poscounts =  df['REVIEW_TEXT'].apply(pos_counts)
    df['NUM_NOUNS'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.NOUN))  # İsim sayısı
    df['NUM_VERBS'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.VERB))  # Fiil sayısı
    df['NUM_ADJECTIVES'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.ADJ))  # Sıfat sayısı
    df['NUM_ADVERBS'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.ADV))  # Zarf sayısı

# 7. Ürünün Ortalama Puanı
def add_average_rating(df):
    # Her ürün için ortalama puan hesaplanır
    average_ratings = df.groupby('PRODUCT_ID')['RATING'].mean()
    df['AVERAGE_RATING'] = df['PRODUCT_ID'].map(average_ratings)

# 8. Ortalama Puan Değişimi
def add_rating_deviation(df):
    # Her bir incelemenin puanının, o ürünün ortalama puanından sapması hesaplanır
    df['RATING_DEVIATION'] = abs(df['RATING'] - df['AVERAGE_RATING'])

# 9. Ürün İçin Toplam İnceleme Sayısı
def add_total_reviews(df):
    # Her ürün için toplam inceleme sayısı hesaplanır
    num_reviews = df.groupby('PRODUCT_ID').size()
    df['NUM_REVIEWS'] = df['PRODUCT_ID'].map(num_reviews)

# 10. İsimli Varlıkların Sayısı
def add_named_entities(df):
    def count_entities(text):
        # İnceleme metninde geçen isimli varlıkların (kişi adı, yer adı vb.) sayısını döndürür
        doc = nlp(text)
        ent_count = len([ent.text for ent in doc.ents])
        return ent_count

    df['NUM_NAMED_ENTITIES'] = df['REVIEW_TEXT'].apply(count_entities)

# 11. İnceleme Metni Uzunluğu
def add_review_length(df):
    # İnceleme metninin uzunluğunu (karakter sayısını) hesaplar
    df['REVIEW_LENGTH'] = df['REVIEW_TEXT'].apply(lambda d: len(d))

# 12. İncelemeler Arası Maksimum Kosinüs Benzerliği
def add_max_similarity(df):
    # İncelemeler arasındaki benzerliği ölçmek için kosinüs benzerliği hesaplanır
    tfidfvectoriser = TfidfVectorizer()
    tfidf_matrix = tfidfvectoriser.fit_transform(df['REVIEW_TEXT'])
    cosine_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    max_similarities = []
    for i, row in enumerate(cosine_similarity_matrix):
        # Her incelemenin en yüksek benzerliğe sahip olduğu inceleme ile olan benzerliği hesaplanır
        max_similarity = max(row[:i].tolist() + row[i+1:].tolist())
        max_similarities.append(max_similarity)

    df['MAX_SIMILARITY'] = max_similarities

# 13. Kelime Başına Ortalama Uzunluk
def add_avg_word_length(df):
    def calculate_average_word_length(text):
        # İnceleme metnindeki kelimelerin ortalama uzunluğunu hesaplar
        words = text.split()
        total_word_length = sum(len(word) for word in words)
        average_word_length = total_word_length / \
            len(words) if len(words) > 0 else 0
        return average_word_length 
    
    df['AVG_WORD_LENGTH'] = df['REVIEW_TEXT'].apply(calculate_average_word_length)

# Tüm Özellikleri Birleştirme (MAX SIMILARITY Hariç, Zaman Alabilir)
def preprocess_features(df):
    # Veriyi işlemek için tüm yukarıdaki fonksiyonları çağırıyoruz
    add_readability_score(df)
    add_vader_sentiment_score(df)
    add_sentiment_category(df, threshold=0.0)
    add_rating_category(df, threshold=3.0)
    add_coherence_column(df)
    add_title_length(df)
    add_pos_tags(df)
    add_average_rating(df)
    add_rating_deviation(df)
    add_total_reviews(df)
    add_named_entities(df)
    add_review_length(df)
    add_avg_word_length(df)
