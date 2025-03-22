import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Custom CSS to enhance UI
def add_custom_css():
    st.markdown(
        """
        <style>
            .main-title { text-align: center; color: #ff4b4b; font-size: 36px; font-weight: bold; }
            .sidebar .sidebar-content { background-color: #222; color: white; }
            .stButton>button { background-color: #ff4b4b; color: white; border-radius: 8px; padding: 10px; font-size: 18px; }
            .stButton>button:hover { background-color: #e63939; }
            img { border-radius: 15px; box-shadow: 0px 0px 15px #ff4b4b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()

st.sidebar.title('🥔 Potato Leaf Prediction System')
st.sidebar.markdown("Detect potato leaf diseases with AI-powered predictions! 🚀")
app_mode = st.sidebar.radio('Navigate', ['🏠 Home', '🔍 Disease Recognition'])

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras", compile=False)

model = load_model()

# Function for making predictions
def model_prediction(test_image):
    image = test_image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return np.argmax(predictions)

class_names = ['🌱 Early Blight', '🍂 Late Blight', '✅ Healthy']

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 class='main-title'>Potato Leaf Disease Detection System 🥔</h1>", unsafe_allow_html=True)
    st.image("Diseases.png", use_column_width=True, caption="Potato Leaf Disease Categories", output_format="PNG")
    st.write("### 🌟 Features of This System:")
    st.markdown("- 🥔 **Accurate AI Model** for potato leaf disease detection.\n- 📷 **Upload Images** for real-time analysis.\n- 🎨 **Attractive UI** with smooth animations.")

# Disease Recognition Page
elif app_mode == "🔍 Disease Recognition":
    st.header("🥔 Upload Potato Leaf Image for Analysis")
    test_image = st.file_uploader("📤 Choose an image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", width=400, use_column_width=True, output_format="PNG")  
        
        if st.button("🧠 Predict Disease"):
            with st.spinner("Analyzing Image... 🕵️‍♂️"):
                time.sleep(2)  # Simulating processing time
                result_index = model_prediction(image)
                prediction_text = f"🌟 Model Prediction: {class_names[result_index]}"
                st.success(prediction_text)
                
                if result_index == 0 or result_index == 1:
                    st.warning("🚨 This potato leaf may need attention! Consult an expert.")
                else:
                    st.balloons()
                    st.success("✅ Your potato leaf is healthy!")
