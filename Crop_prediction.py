import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
# Load the model from the .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(input_data):
    model=load_model('RandomForest.pkl')
    prediction = model.predict([input_data])
    return prediction

# Streamlit app
def main():
    st.title('Crop Prediction App 🌽')
    st.write('Enter the values and click the button to execute the model.')

    # Input fields for 7 values
    value1 = st.number_input('Nitrogen', value=0.0)
    value2 = st.number_input('Phosphorous', value=0.0)
    value3 = st.number_input('Potassium', value=0.0)
    value4 = st.number_input('Temperature', value=0.0)
    value5 = st.number_input('Humidity', value=0.0)
    value6 = st.number_input('ph', value=0.0)
    value7 = st.number_input('Rainfall', value=0.0)

    # Button to execute model
    if st.button('Execute Model'):
        input_data = [value1, value2, value3, value4, value5, value6, value7]
        prediction = predict(np.array(input_data))
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()