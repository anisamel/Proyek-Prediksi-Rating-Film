import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk memuat pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load model, encoder, dan kolom fitur
mlb = load_pickle('encorder1.sav')
model = load_pickle('prediksi_model1.sav')
x_train_columns = load_pickle('x_train_columns1.sav')

st.title("Prediksi Rating Film Berdasarkan Genre")

# Input pengguna untuk genre
all_genres = list(mlb.classes_)
selected_genres = st.multiselect("Pilih Genre", options=all_genres)

# Tombol prediksi
if st.button("Prediksi"):
    if not selected_genres:
        st.error("Silakan pilih setidaknya satu genre!")
    else:
        try:
            def predict_rating(new_genre, mlb, model, x_train_columns):
                # Validasi genre
                invalid_genres = [genre for genre in new_genre if genre not in mlb.classes_]
                if invalid_genres:
                    raise ValueError(f"Genre tidak valid: {', '.join(invalid_genres)}")

                # Transformasi ke format one-hot
                new_genre_encoded = pd.DataFrame(mlb.transform([new_genre]), columns=mlb.classes_)
                new_genre_encoded = new_genre_encoded.reindex(columns=x_train_columns, fill_value=0)

                # Prediksi rating
                predicted_rating = model.predict(new_genre_encoded)
                return predicted_rating[0]

            rating = predict_rating(selected_genres, mlb, model, x_train_columns)
            st.success(f"Rating yang diprediksi: {rating:.2f}")
        except ValueError as e:
            st.error(e)
