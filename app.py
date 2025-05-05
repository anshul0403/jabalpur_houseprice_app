import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('house_price_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title("🏠 Jabalpur House Price Predictor")

# Input form
location = st.selectbox("Location", label_encoders['Location'].classes_)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=1400)
property_type = st.selectbox("Property Type", label_encoders['Property_Type'].classes_)
floor = st.number_input("Floor", 0, 30, 2)
total_floors = st.number_input("Total Floors", 1, 50, 5)
property_age = st.number_input("Property Age", 0, 100, 5)
parking = st.selectbox("Parking", label_encoders['Parking'].classes_)
lift = st.selectbox("Lift", label_encoders['Lift'].classes_)
security = st.selectbox("Security", label_encoders['Security'].classes_)
furnished = st.selectbox("Furnished", label_encoders['Furnished'].classes_)

if st.button("Predict Price"):
    input_dict = {
        'Location': label_encoders['Location'].transform([location])[0],
        'BHK': bhk,
        'Bathrooms': bathrooms,
        'Size_sqft': size,
        'Property_Type': label_encoders['Property_Type'].transform([property_type])[0],
        'Floor': floor,
        'Total_Floors': total_floors,
        'Property_Age': property_age,
        'Parking': label_encoders['Parking'].transform([parking])[0],
        'Lift': label_encoders['Lift'].transform([lift])[0],
        'Security': label_encoders['Security'].transform([security])[0],
        'Furnished': label_encoders['Furnished'].transform([furnished])[0],
    }

    input_df = pd.DataFrame([input_dict])
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{predicted_price:.2f} Lakhs")
