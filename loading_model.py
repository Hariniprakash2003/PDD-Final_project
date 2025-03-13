# import numpy as np
# import cv2
# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf

# # Define the model architecture
# def parkinson_disease_detection_model(input_shape=(128, 128, 1)):
#     regularizer = tf.keras.regularizers.l2(0.001)
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Conv2D(128, (5, 5), padding='same', strides=(1, 1), name='conv1', activation='relu', 
#                      kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
#     model.add(MaxPool2D((9, 9), strides=(3, 3)))

#     model.add(Conv2D(64, (5, 5), padding='same', strides=(1, 1), name='conv2', activation='relu', 
#                      kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
#     model.add(MaxPool2D((7, 7), strides=(3, 3)))
    
#     model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv3', activation='relu', 
#                      kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
#     model.add(MaxPool2D((5, 5), strides=(2, 2)))

#     model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv4', activation='relu', 
#                      kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
#     model.add(MaxPool2D((3, 3), strides=(2, 2)))    
    
#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', name='fc1'))
#     model.add(Dropout(0.5))
#     model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform', name='fc3'))
    
#     optimizer = Adam(3.15e-5)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Load the weights
# model = parkinson_disease_detection_model(input_shape=(128, 128, 1))
# model.load_weights('model_weights.h5')

# # Load an image for prediction (assuming you have a processed image)
# processed_image = ...  # Your preprocessed image

# # Perform prediction
# prediction = model.predict(np.expand_dims(processed_image, axis=0))

# # Display the prediction result (you can replace this with your visualization)
# print("Prediction:", prediction)

from tftrainer import model

model.load_weights('parkinson_disease_detection.h5')