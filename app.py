import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import pandas as pd

# NLTK stopwords indir (ilk çalıştırmada bir kez çalışır)
nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')

# Başlık
st.title("🔑 Anahtar Kelime Çıkarıcı (TF-IDF ile)")
st.markdown("Bu uygulama, girdiğiniz metindeki en anlamlı kelimeleri TF-IDF algoritması ile çıkarır.")

# Kullanıcıdan metin al
text_input = st.text_area("Lütfen metninizi buraya girin:")

if st.button("Anahtar Kelimeleri Çıkar"):
    if text_input.strip() == "":
        st.warning("⚠️ Lütfen bir metin girin.")
    else:
        # TF-IDF hesapla
        vectorizer = TfidfVectorizer(stop_words=turkish_stopwords)
        X = vectorizer.fit_transform([text_input])

        # Skorları DataFrame olarak düzenle
        df_keywords = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out(), columns=["Skor"])
        df_keywords = df_keywords.sort_values("Skor", ascending=False)

        # Sonuçları göster
        st.subheader("📌 En Anlamlı Anahtar Kelimeler")
        st.dataframe(df_keywords.head(10))
