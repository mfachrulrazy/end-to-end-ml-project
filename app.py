import streamlit as st
import pandas as pd
import numpy as np
import joblib
from babel.numbers import format_currency

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.title("Prediksi Harga Rumah di Indonesia")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/rf_model.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        feature_scaler = joblib.load("models/feature_scaler.pkl")
        price_scaler = joblib.load("models/price_scaler.pkl")
        return model, feature_columns, feature_scaler, price_scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

# load asset
model, feature_columns, feature_scaler, price_scaler = load_assets()

if model is None:
    st.error("Model could not be loaded. Please check the model files.")
else:
    st.markdown("### Masukkan fitur-fitur rumah untuk prediksi harga:")
    with st.form(key='prediction_form'):
        st.header("Fitur Rumah")
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Luas Tanah (m2)", min_value=30.0, max_value=1000.0, value=120.0, step=10.0)
            bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3, step=1)
            garage = st.number_input("Luas Garasi (mobil)", min_value=0, max_value=5, value=1, step=1)
        with col2:
            building_area = st.number_input("Luas Bangunan (m2)", min_value=20.0, max_value=800.0, value=90.0, step=10.0)
            bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=8, value=1, step=1)
            city = st.selectbox("Kota", options=['Jakarta Selatan', 'Jakarta Timur', 'Jakarta Barat', 'Jakarta Pusat','Jakarta Utara', 'Bogor', 'Depok', 'Tangerang', 'Tangerang Selatan', 'Bekasi', 'surabaya', 'malang', 'sidoarjo', 'makassar', 'gowa', 'maros', 'parepare', 'palopo', 'bulukumba', 'takalar'])
        submit_button = st.form_submit_button(label='Prediksi Harga') 

    if submit_button:
        try:
            input_data = {
                'area': [area],
                'building_area': [building_area],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'garage': [garage],
                'city': [city]
            }
            input_df = pd.DataFrame(input_data)
            # scaling feature
            input_df[['area', 'building_area']] = feature_scaler.transform(
                input_df[['area', 'building_area']]
            )
            # one hot encoding
            input_df = pd.get_dummies(input_df, 
                columns=['city', 'bedrooms', 'bathrooms', 'garage'],
                prefix=['City', 'Bedroom', 'Bathroom', 'Garage'])
            # reindex to match training feature columns
            input_processed = input_df.reindex(columns=feature_columns, fill_value=0)

            # predict
            predicted_price_scaled = model.predict(input_processed)
            original_price = price_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0] * 1000000 # in millions rupiah

            # format currency
            original_price = format_currency(original_price, 'IDR', locale='id_ID')

            st.success("Prediksi berhasil!")
            # inverse scaling
            st.metric(
                label="Harga Prediksi Rumah:",
                #value=predicted_price_scaled
                value = original_price
                )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
