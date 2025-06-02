import streamlit as st
import joblib  # ✅ Use joblib instead of pickle

# Load trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')        # Make sure the file exists
vectorizer = joblib.load('vectorizer.pkl')              # Make sure the file exists

# Custom styles (same as you already have)
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #1a1a1a;
            font-size: 2.5rem;
            margin-bottom: 0.5em;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            border: 1px solid #ced4da;
            padding: 0.5rem;
            color: #000000;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .css-1cpxqw2 {
            color: #212529;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title('Spam Message Classifier')

# Input message
message = st.text_input('Enter a message')

# Predict button
if st.button('Predict'):
    # Transform the input using the loaded vectorizer
    input_dtm = vectorizer.transform([message])

    # Predict using the loaded model
    prediction = model.predict(input_dtm)[0]

    # Display result
    st.markdown(f"**Prediction:** `{prediction}`")
    if prediction == 'spam':
        st.warning('This message is spam ❌')
    else:
        st.success('This message is legitimate ✅')

# Optional animation
st.balloons()
