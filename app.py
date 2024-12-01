import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk memuat model dan TfidfVectorizer
def load_model(model_name):
    model_filename = f'{model_name}.pkl'
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"Model {model_name} tidak ditemukan.")
        return None

def load_tfidf_vectorizer():
    tfidf_filename = 'tfidfvector.pkl'
    if os.path.exists(tfidf_filename):
        with open(tfidf_filename, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        return tfidf_vectorizer
    else:
        st.error("TfidfVectorizer tidak ditemukan.")
        return None

# Antarmuka pengguna Streamlit
st.title("Prediksi Bullying dengan Model Machine Learning")

# Pilih model yang akan digunakan
model_choice = st.selectbox("Pilih Model", ("LinearSVC", "LogisticRegression", "MultinomialNB", 
                                           "DecisionTreeClassifier", "AdaBoostClassifier", 
                                           "BaggingClassifier", "SGDClassifier"))

# Memuat model yang dipilih
model = load_model(model_choice)
if model is None:
    st.stop()

# Memuat TfidfVectorizer
tfidf_vectorizer = load_tfidf_vectorizer()
if tfidf_vectorizer is None:
    st.stop()

# Input untuk memasukkan tweet
tweet = st.text_area("Masukkan tweet yang akan diuji (tekan ctrl+enter untuk mereset kembali sebelum tekan tombol prediksi):", "")
st.write("sebelum tekan tombol prediksi tekan **ctrl+enter** untuk mereset kembali")
# Prediksi ketika tombol ditekan
if st.button("Prediksi"):
    if tweet.strip() == "":
        st.warning("Harap masukkan tweet terlebih dahulu!")
    else:
        # Mengubah tweet menjadi representasi TF-IDF
        tweet_vectorized = tfidf_vectorizer.transform([tweet])

        # Melakukan prediksi menggunakan model yang dipilih
        prediction = model.predict(tweet_vectorized)

        # Menampilkan hasil prediksi dengan kategori yang jelas
        if prediction[0] == 'cyberbullying':
            st.write("Tweet ini **mengandung bullying**.")
        elif prediction[0] == 'not_cyberbullying':
            st.write("Tweet ini **tidak mengandung bullying**.")
        else:
            st.write(f"Prediksi tidak diketahui: {prediction[0]}")