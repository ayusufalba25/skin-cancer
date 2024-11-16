import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

# Get the model
densenetmodel = tf.keras.models.load_model('model/DenseNet121_final_model2.keras')
incv3model = tf.keras.models.load_model('model/InceptionV3_final_model2.keras')

# Predicted probability from the weights
wDense=0.8500000000000001
wIncV3=0.1499999999999999
classes = ['Benign', 'Malignant']

st.title("Selamat Datang di SKI-NET")
st.text("Skin Cancer Detection Using Transfer Learning Neural Networks")
st.text("Dibuat Oleh: Yukindulu")

st.header("Prediksi Kanker Kulit")

st.subheader("Contoh Prediksi")

selected_image = st.radio(
    "Pilih gambar: ",
    ("Gambar 1", "Gambar 2", "Gambar 3")
)

if selected_image == "Gambar 1":
    st.image("apps/example1.jpeg", caption="Demo Data: Contoh 1", use_container_width=200)
    selected_image_path = "apps/example1.jpeg"
elif selected_image == "Gambar 2":
    st.image("apps/example2.jpeg", caption="Demo Data: Contoh 2", use_container_width=200)
    selected_image_path = "apps/example2.jpeg"
elif selected_image == "Gambar 3":
    st.image("apps/example3.jpeg", caption="Demo Data: Contoh 3", use_container_width=200)
    selected_image_path = "apps/example3.jpeg"

if st.button("Jalankan"):
    # Load the image with the expected target size
    image = tf.keras.preprocessing.image.load_img(selected_image_path, target_size=(224, 224))  # Adjust size based on your model's input

    # Convert the image to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Normalize the pixel values to [0, 1] (if your model requires it)
    image_array = image_array / 255.0

    # Add a batch dimension (the model expects inputs as a batch of images)
    image_array = tf.expand_dims(image_array, axis=0)

    pred_densenet = densenetmodel.predict(image_array)
    pred_densenet = pred_densenet[:, 1]

    pred_incv3 = incv3model.predict(image_array)
    pred_incv3 = pred_incv3[:, 1]

    predicted_prob_all = (wDense*pred_densenet) + (wIncV3*pred_incv3)
    predicted_class_all = classes[(predicted_prob_all > 0.5).astype(int)]
    st.write(predicted_class_all)





uploaded_files = st.file_uploader(
    "insert an image", accept_multiple_files=False, type=["png", "jpg"]
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Open and display each image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred with {uploaded_file.name}: {e}")

submitter =  f"{uploaded_files}", st.button("submit image", use_container_width= True)