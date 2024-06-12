import os
import keras
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('Classificação de Frutas Segundo a Espécie')
st.image('banner.png')
st.write('Carregue imagens para classificar:')

fruits_names = ['Morango', 'Pessego', 'Romã']


model1 = tf.keras.models.load_model('Modelo_01_100.keras') #Modelo Aprendizado Profundo - 500 Epocas

model2 = tf.keras.models.load_model('Modelo_2_Normal.keras') #Modelo Rede Normal - 500 Epocas

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(300,300))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array , 0)
    
    predictions = model1.predict(input_image_exp_dim)
    result1 = tf.nn.softmax(predictions[0])

    predictions = model2.predict(input_image_exp_dim)
    result2 = tf.nn.softmax(predictions[0])

    outcome = ''
    ' Rede neural Aprendizado Profundo : A imagem pertence a classe ' + fruits_names[np.argmax(result1)] + ' com percentual de acerto de ' +  str(np.max(result1)*100) + '%' 
    ''
    ' Rede neural Convencional : A imagem pertence a classe ' + fruits_names[np.argmax(result2)] + ' com percentual de acerto de ' +  str(np.max(result2)*100) + '%'
    
    return outcome

uploaded_file = st.file_uploader('Carregar imagem')
if uploaded_file is not None :
    with open(os.path.join(uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, width = 200)
    st.markdown(classify_images(uploaded_file))

# Rodar: streamlit run app.py