
from pyngrok import ngrok
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = keras.models.load_model('C:\Users\HAFFIZUDDIN\Desktop\app\model(VGG16-20).h5')  # Replace with the actual path to your model

# Define a function to make predictions
def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return prediction

# Streamlit App
def main():
    st.title("Image Classification App")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        prediction = predict(uploaded_image)

        class_names = ['Apple___Apple_scab' , 'Apple___Black_rot' , 'Apple___Cedar_apple_rust' , 'Apple___healthy' ,'Cherry_(including_sour)___Powdery_mildew ' , 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot' , 'Corn_(maize)___Common_rust_ ', 'Corn_(maize)___Northern_Leaf_Blight' , 'Corn_(maize)___healthy ,Grape___Black_rot','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'  , 'Grape___healthy'  , 'Potato___Early_blight' , 'Potato___Late_blight' , 'Potato___healthy'  , 'Tomato___Early_blight', 'Tomato___Late_blight'  ,'Tomato___Leaf_Mold' ,'Tomato___Septoria_leaf_spot'  ,'Tomato___Target_Spot'  , 'Tomato___healthy' ]  
        st.write("Prediction:")
        st.write(f"Class: {class_names[np.argmax(prediction)]}")
        st.write(f"Confidence: {100 * np.max(prediction):.2f}%")


if __name__ == '__main__':
    main()
