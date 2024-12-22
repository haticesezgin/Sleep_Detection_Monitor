from ozellik_muhendisligi import preprocess_features
from text_processes import preprocess_text
import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/amazon_reviews.txt", sep="\t")

# PREPROCESS TEXT: REMOVE STOP WORDS, STEM AND LEMMATISE WORDS
df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(preprocess_text)

# ADD OTHER FEATURES
preprocess_features(df)

# PRINT FINAL COLUMN
print(df.columns)
print(df.head())
print("Kütüphaneler başarıyla yüklendi!")
