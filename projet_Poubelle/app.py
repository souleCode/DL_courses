import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array,load_img

@st.cache_data()
def load():
    model_path= 'poubelle_model.keras'
    model = load_model(model_path,compile=False)
    return model

#chargement du model

model =load()

def predict(upload):
    img = Image.open(upload)
    img =np.asarray(img)
    img_resize =cv2.resize(img,(224,224))
    img_resize = np.expand_dims(img_resize,axis=0)
    pred =model.predict(img_resize)  
    
    rec_probability =  pred[0][0]
    return rec_probability


st.title('Poubelle Intelligente')

upload =st.file_uploader('Chargez une image de votre objet',
                         type=['jpeg','png','jpg'])
c1,c2 =st.columns(2)
if upload:
    rec_probability = predict(upload)*100
    org_probability = (1-rec_probability)*100
    c1.image(Image.open(upload))
    if rec_probability > 50:
        c2.write(f"Je suis certain a {rec_probability:.2f}% que l'objet est recyclable")
    else:
        c2.write(f"Je suis certain a {org_probability:.2f}% que l'objet n'est pas recyclable")
   
