from nltk.util import ngrams # type: ignore

def calculate_jaccard_similarity(text1, text2):
    # Metinleri iki kelimelik bi-gramlara ayırıyoruz
    ngrams1 = list(ngrams(text1.split(), 2))  # text1'i bi-gramlara ayır
    ngrams2 = list(ngrams(text2.split(), 2))  # text2'yi bi-gramlara ayır

    # Bi-gramları kümelere dönüştür
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    
    # Kesişim ve birleşim hesapla
    intersection = len(list(set1.intersection(set2)))
    union = len(list(set1.union(set2)))
    
    # Jaccard benzerlik skorunu hesapla
    similarity_score = float(intersection) / union
    return similarity_score
