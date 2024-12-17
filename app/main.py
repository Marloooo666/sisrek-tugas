import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import os

# Memuat dataset baru (dataset berada di direktori yang sama dengan skrip)
dataset_path = os.path.join(os.getcwd(), 'beritasport.csv')  # Menggunakan direktori kerja saat ini
berita_df = pd.read_csv(dataset_path)

# Memastikan bahwa kolom yang benar digunakan
# Jika kolom yang benar adalah 'Product Name', sesuaikan dengan kolom yang ada
if 'Product Name' not in berita_df.columns:
    st.error("Kolom 'Product Name' tidak ditemukan dalam dataset.")
else:
    # Fungsi untuk pembersihan teks
    clean_spcl = re.compile('[/(){}\[\]\|@,;]')
    clean_symbol = re.compile('[^0-9a-z #+_]')
    sastrawi = StopWordRemoverFactory()
    stopworda = sastrawi.get_stop_words()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def clean_text(text):
        # Memastikan bahwa input adalah string
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()  # Mengubah teks menjadi huruf kecil
        text = clean_spcl.sub(' ', text)  # Menghapus karakter spesial
        text = clean_symbol.sub('', text)  # Menghapus simbol selain angka dan huruf
        text = stemmer.stem(text)  # Melakukan stemming
        text = ' '.join(word for word in text.split() if word not in sastrawi.get_stop_words())  # Menghapus stopword
        return text

    # Menerapkan pembersihan teks pada kolom 'Product Name'
    berita_df['desc_clean'] = berita_df['Product Name'].apply(clean_text)

    # Menghitung TF-IDF dan Cosine Similarity
    berita_df.set_index('Product Name', inplace=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
    tfidf_matrix = tf.fit_transform(berita_df['desc_clean'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(berita_df.index)

    # Fungsi rekomendasi berita
    def recomendation(keyword):
        recommended_berita = []

        # Mengecek apakah ada berita yang cocok dengan kata kunci
        matching_berita = indices[indices.str.contains(keyword, case=False, na=False)]
        if not matching_berita.empty:
            base_berita = matching_berita.iloc[0]
            idx = indices[indices == base_berita].index[0]
            score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
            top_indexes = list(score_series.iloc[1:].index)

            # Menyusun daftar rekomendasi berdasarkan kemiripan
            for i in top_indexes:
                berita_title = berita_df.index[i]
                similarity_score = score_series[i]
                result = f"{berita_title} - {similarity_score:.2f}"
                if result not in recommended_berita:
                    recommended_berita.append(result)

            return recommended_berita
        else:
            return f"Tidak ada berita yang cocok dengan kata kunci '{keyword}'."

    # Layout aplikasi Streamlit
    st.title("Sistem Rekomendasi Berita")
    st.sidebar.header("Opsi Pencarian")

    # Input dari pengguna untuk kata kunci pencarian
    keyword = st.sidebar.text_input("Masukkan kata kunci untuk pencarian:")

    # Menampilkan rekomendasi
    if keyword:
        recommendations = recomendation(keyword)
        if isinstance(recommendations, list):
            st.write(f"Rekomendasi untuk '{keyword}':")
            for rec in recommendations:
                st.write(rec)
        else:
            st.write(recommendations)
