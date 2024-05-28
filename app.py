
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "/content/drive/MyDrive/Datasets/David/Deployment/cnn_lstm_model.h5"
model = load_model(model_path)

# Define the input fields
def user_input_features():
    duration = st.number_input('Duration', min_value=0, max_value=1000000, value=0)
    protocol_type = st.selectbox('Protocol Type', [0, 1, 2])
    service = st.selectbox('Service', list(range(67)))
    flag = st.selectbox('Flag', list(range(12)))
    src_bytes = st.number_input('Source Bytes', min_value=0, max_value=1000000000, value=0)
    dst_bytes = st.number_input('Destination Bytes', min_value=0, max_value=1000000000, value=0)
    wrong_fragment = st.number_input('Wrong Fragment', min_value=0, max_value=1000, value=0)
    hot = st.number_input('Hot', min_value=0, max_value=1000, value=0)
    logged_in = st.selectbox('Logged In', [0, 1])
    count = st.number_input('Count', min_value=0, max_value=1000, value=0)
    srv_count = st.number_input('Service Count', min_value=0, max_value=1000, value=0)
    serror_rate = st.slider('Serror Rate', 0.0, 1.0, 0.0)
    srv_serror_rate = st.slider('Service Serror Rate', 0.0, 1.0, 0.0)
    rerror_rate = st.slider('Rerror Rate', 0.0, 1.0, 0.0)
    srv_rerror_rate = st.slider('Service Rerror Rate', 0.0, 1.0, 0.0)
    same_srv_rate = st.slider('Same Service Rate', 0.0, 1.0, 0.0)
    diff_srv_rate = st.slider('Different Service Rate', 0.0, 1.0, 0.0)
    srv_diff_host_rate = st.slider('Service Different Host Rate', 0.0, 1.0, 0.0)
    dst_host_count = st.number_input('Destination Host Count', min_value=0, max_value=1000, value=0)
    dst_host_srv_count = st.number_input('Destination Host Service Count', min_value=0, max_value=1000, value=0)
    dst_host_same_srv_rate = st.slider('Destination Host Same Service Rate', 0.0, 1.0, 0.0)
    dst_host_diff_srv_rate = st.slider('Destination Host Different Service Rate', 0.0, 1.0, 0.0)
    dst_host_same_src_port_rate = st.slider('Destination Host Same Source Port Rate', 0.0, 1.0, 0.0)
    dst_host_srv_diff_host_rate = st.slider('Destination Host Service Different Host Rate', 0.0, 1.0, 0.0)
    dst_host_serror_rate = st.slider('Destination Host Serror Rate', 0.0, 1.0, 0.0)
    dst_host_srv_serror_rate = st.slider('Destination Host Service Serror Rate', 0.0, 1.0, 0.0)
    dst_host_rerror_rate = st.slider('Destination Host Rerror Rate', 0.0, 1.0, 0.0)
    dst_host_srv_rerror_rate = st.slider('Destination Host Service Rerror Rate', 0.0, 1.0, 0.0)

    data = {
        'duration': duration,
        'protocol_type': protocol_type,
        'service': service,
        'flag': flag,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'wrong_fragment': wrong_fragment,
        'hot': hot,
        'logged_in': logged_in,
        'count': count,
        'srv_count': srv_count,
        'serror_rate': serror_rate,
        'srv_serror_rate': srv_serror_rate,
        'rerror_rate': rerror_rate,
        'srv_rerror_rate': srv_rerror_rate,
        'same_srv_rate': same_srv_rate,
        'diff_srv_rate': diff_srv_rate,
        'srv_diff_host_rate': srv_diff_host_rate,
        'dst_host_count': dst_host_count,
        'dst_host_srv_count': dst_host_srv_count,
        'dst_host_same_srv_rate': dst_host_same_srv_rate,
        'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
        'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
        'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
        'dst_host_serror_rate': dst_host_serror_rate,
        'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
        'dst_host_rerror_rate': dst_host_rerror_rate,
        'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app
st.title("Network Anomaly Detection")

st.sidebar.header('User Input Features')
df = user_input_features()

st.subheader('User Input Features')
st.write(df)

# Split input into categorical and numerical features
X_protocol_type = df['protocol_type']
X_service = df['service']
X_flag = df['flag']
X_numerical = df.drop(['protocol_type', 'service', 'flag'], axis=1)

# Make prediction
prediction = model.predict([X_protocol_type, X_service, X_flag, X_numerical])
prediction_prob = prediction[0][0]

st.subheader('Prediction')
if prediction_prob > 0.5:
    st.write('Anomaly Detected')
else:
    st.write('Normal Traffic')

st.subheader('Prediction Probability')
st.write(prediction_prob)

