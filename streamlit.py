import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .sidebar .sidebar-content h2, .sidebar .sidebar-content h3, .sidebar .sidebar-content p {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #004080;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0059b3;
        color: white;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 5px;
        border: 1px solid #cccccc;
    }
    .stSlider .stSlider>div>div {
        color: #ffffff;
    }
    .stDataFrame div {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header with styling
st.title(" Gemstone Price Prediction ")
st.markdown("""
    <h3 style="color: #80bfff;">Predict the price of gemstones based on their characteristics.</h3>
    <hr style="border:1px solid #004080;">
""", unsafe_allow_html=True)

# Load the pickled model and pipeline
model_path = 'model.pkl'
pipeline_path = 'preprocessor.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(pipeline_path, 'rb') as file:
    preprocessor = pickle.load(file)

# Load the dataset to retrieve columns and preview the data
data_path = 'gemstone.csv'
df = pd.read_csv(data_path)
st.subheader("Data Preview")
st.write("The first few rows of the dataset:")
st.dataframe(df.head())

# Sidebar for user input
st.sidebar.header('Input Features')
st.sidebar.markdown("### Enter the features of the gemstone below:")

def user_input_features():
    cut = st.sidebar.selectbox('Cut', df['cut'].unique())
    color = st.sidebar.selectbox('Color', df['color'].unique())
    clarity = st.sidebar.selectbox('Clarity', df['clarity'].unique())
    carat = st.sidebar.slider('Carat', float(df['carat'].min()), float(df['carat'].max()), float(df['carat'].mean()))
    depth = st.sidebar.slider('Depth', float(df['depth'].min()), float(df['depth'].max()), float(df['depth'].mean()))
    table = st.sidebar.slider('Table', float(df['table'].min()), float(df['table'].max()), float(df['table'].mean()))
    x = st.sidebar.slider('X', float(df['x'].min()), float(df['x'].max()), float(df['x'].mean()))
    y = st.sidebar.slider('Y', float(df['y'].min()), float(df['y'].max()), float(df['y'].mean()))
    z = st.sidebar.slider('Z', float(df['z'].min()), float(df['z'].max()), float(df['z'].mean()))

    data = {
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'carat': carat,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the user input using the loaded pipeline
preprocessed_input = preprocessor.transform(input_df)

# Predict price using the loaded model
prediction = model.predict(preprocessed_input)

# Convert the prediction to a float
predicted_price = float(prediction[0])

st.subheader("Prediction")
st.markdown(f"""
    <h2 style="color: #80bfff;">Predicted Price: ${predicted_price:,.2f}</h2>
""", unsafe_allow_html=True)

st.write("Note: The prediction is based on the Random Forest model loaded from a pickle file.")
