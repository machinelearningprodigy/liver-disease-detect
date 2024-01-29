import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

st.set_page_config(page_icon="🦰")

# Load the saved XGBoost model
xgb_model = pickle.load(open('liver.pkl', 'rb'))

# List of features that you used during training
trained_features = ['Age of the patient', 'Gender of the patient', 'Total Bilirubin', 'Direct Bilirubin',
                    'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
                    'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin',
                    'A/G Ratio Albumin and Globulin Ratio']

# Gender labels
gender_labels = {
    0: 'Female',
    1: 'Male'
}

def clean_column_names(df):
    # Clean up column names to match the training data
    df.columns = [col.strip() for col in df.columns]
    return df

def main():
    st.markdown(
        "<h1 style='text-align: center;'>Liver Disease Detection</h1>",
        unsafe_allow_html=True,
)

    # User input
    age = st.slider("Enter your age:", min_value=1, max_value=100, value=50)
    gender = st.radio("Select your gender:", options=list(gender_labels.values()))
    total_bilirubin = st.slider("Enter total bilirubin level:", min_value=0.1, max_value=8.0, value=1.0)
    direct_bilirubin = st.slider("Enter direct bilirubin level:", min_value=0.1, max_value=4.0, value=0.5)
    alkphos = st.slider("Enter Alkaline Phosphotase level:", min_value=20, max_value=300, value=150)
    sgpt = st.slider("Enter Alamine Aminotransferase level:", min_value=10, max_value=200, value=50)
    sgot = st.slider("Enter Aspartate Aminotransferase level:", min_value=10, max_value=200, value=50)
    total_proteins = st.slider("Enter total proteins level:", min_value=2.0, max_value=10.0, value=6.0)
    alb_albumin = st.slider("Enter Albumin level:", min_value=1.0, max_value=5.0, value=3.0)
    ag_ratio = st.slider("Enter Albumin/Globulin Ratio:", min_value=0.1, max_value=2.5, value=1.0)

    # Convert categorical inputs to numerical
    gender_numeric = [key for key, value in gender_labels.items() if value == gender][0]

    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({
        'Age of the patient': [age],
        'Gender of the patient': [gender_numeric],
        'Total Bilirubin': [total_bilirubin],
        'Direct Bilirubin': [direct_bilirubin],
        'Alkphos Alkaline Phosphotase': [alkphos],
        'Sgpt Alamine Aminotransferase': [sgpt],
        'Sgot Aspartate Aminotransferase': [sgot],
        'Total Protiens': [total_proteins],
        'ALB Albumin': [alb_albumin],
        'A/G Ratio Albumin and Globulin Ratio': [ag_ratio]
    })

    # Clean up column names
    user_data = clean_column_names(user_data)

    # Set feature names explicitly for prediction
    user_data = user_data[trained_features]

    # Convert DataFrame to NumPy array
    user_data_array = user_data.to_numpy()

    # Use the correct model for predictions
    prediction_proba = xgb_model.predict(user_data_array)

    # Display prediction
    prediction = 1 if prediction_proba[0] > 0.5 else 0
    prediction_text = f"Prediction: {'Liver Disease Present' if prediction == 1 else 'No Liver Disease'}"
    prediction_color = 'red' if prediction == 1 else 'green'
    prediction_html = f"<p style='color: {prediction_color}; text-align: center; font-weight: bold; width: 50%; margin: 0 auto; padding: 10px; border: 2px solid {prediction_color}; border-radius: 5px;'>{prediction_text}</p>"
    st.markdown(prediction_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
