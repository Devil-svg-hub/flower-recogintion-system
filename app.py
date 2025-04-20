import os
import keras
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, RandomFlip, RandomRotation, RandomZoom, Rescaling
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Check if model exists, if not train it
if not os.path.exists('Flower_Recog_Model.h5'):
    st.info('Training model for first time use. This may take a few minutes...')
    
    img_size = 180
    batch = 32
    
    # Load and preprocess the dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'Images/',
        seed=123,
        validation_split=0.2,
        subset='training',
        batch_size=batch,
        image_size=(img_size, img_size)
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'Images/',
        seed=123,
        validation_split=0.2,
        subset='validation',
        batch_size=batch,
        image_size=(img_size, img_size)
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Data augmentation
    data_augmentation = Sequential([
        RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ])
    
    # Create the model
    model = Sequential([
        data_augmentation,
        Rescaling(1./255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5)
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(train_ds, epochs=15, validation_data=val_ds)
    
    # Save the model
    model.save('Flower_Recog_Model.h5')
else:
    model = load_model('Flower_Recog_Model.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

# Create upload directory if it doesn't exist
if not os.path.exists('upload'):
    os.makedirs('upload')

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join('upload', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)
    st.markdown(classify_images(file_path))

