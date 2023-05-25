import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Function model
def set_neural_network_architecture(input_size, num_classes):

    # Entradas
    sequence=tf.keras.layers.Input(shape=input_size)
    
    # Add layer branches
    # Primer bloque
    Inception_1a_3x3_reduce = tf.keras.layers.Conv2D(16, (1, 1))(sequence)
    Inception_1a_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_1a_3x3_reduce)
    Inception_1a_3x3 = tf.keras.layers.Conv2D(18, (3, 3), padding='same')(Inception_1a_3x3_relu_reduce)
    
    Inception_1a_1x1 = tf.keras.layers.Conv2D(18, (1, 1))(sequence)
    
    Inception_1a_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(sequence)
    Inception_1a_pool_proj = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1a_pool)

    
    Inception_1a_5x5_reduce = tf.keras.layers.Conv2D(16, (1, 1))(sequence)
    Inception_1a_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_1a_5x5_reduce)
    Inception_1a_5x5 = tf.keras.layers.Conv2D(18, (5, 5), padding='same')(Inception_1a_5x5_relu_reduce)

    depthcat_1a = tf.keras.layers.Concatenate([Inception_1a_3x3,Inception_1a_1x1,Inception_1a_pool_proj,Inception_1a_5x5], axis=-1)
    Inception_1a_relu = tf.keras.layers.ReLU()(depthcat_1a)

    # Segundo bloque
    Inception_1b_3x3_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Inception_1a_relu)
    Inception_1b_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_1b_3x3_reduce)
    Inception_1b_3x3 = tf.keras.layers.Conv2D(18, (3, 3), padding='same')(Inception_1b_3x3_relu_reduce)
    
    Inception_1b_1x1 = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1a_relu)
    
    Inception_1b_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(Inception_1a_relu)
    Inception_1b_pool_proj = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1b_pool)

    
    Inception_1b_5x5_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Inception_1a_relu)
    Inception_1b_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_1b_5x5_reduce)
    Inception_1b_5x5 = tf.keras.layers.Conv2D(18, (5, 5), padding='same')(Inception_1b_5x5_relu_reduce)

    depthcat_1b = tf.keras.layers.Concatenate([Inception_1b_3x3,Inception_1b_1x1,Inception_1b_pool_proj,Inception_1b_5x5], axis=-1)
    Inception_1b_relu = tf.keras.layers.ReLU()(depthcat_1b)
    

    # Tercer bloque
    Inception_1c_3x3_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Inception_1b_relu)
    Inception_1c_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_1c_3x3_reduce)
    Inception_1c_3x3 = tf.keras.layers.Conv2D(18, (3, 3), padding='same')(Inception_1c_3x3_relu_reduce)
    
    Inception_1c_1x1 = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1b_relu)
    
    Inception_1c_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(Inception_1b_relu)
    Inception_1c_pool_proj = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1c_pool)

    
    Inception_1c_5x5_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Inception_1b_relu)
    Inception_1c_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_1c_5x5_reduce)
    Inception_1c_5x5 = tf.keras.layers.Conv2D(18, (5, 5), padding='same')(Inception_1c_5x5_relu_reduce)

    depthcat_1c = tf.keras.layers.Concatenate([Inception_1c_3x3,Inception_1c_1x1,Inception_1c_pool_proj,Inception_1c_5x5], axis=-1)
    
    #       Adición Layer

    Addition_1 = tf.keras.layers.Add()([depthcat_1c, Inception_1a_relu])
    Addition_1_relu = tf.keras.layers.ReLU()(Addition_1)

    # Cuarto bloque
    
    Inception_1d_3x3_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Addition_1_relu)
    Inception_1d_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_1d_3x3_reduce)
    Inception_1d_3x3 = tf.keras.layers.Conv2D(18, (3, 3), padding='same')(Inception_1d_3x3_relu_reduce)
    
    Inception_1d_1x1 = tf.keras.layers.Conv2D(18, (1, 1))(Addition_1_relu)
    
    Inception_1d_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(Addition_1_relu)
    Inception_1d_pool_proj = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1d_pool)

    
    Inception_1d_5x5_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Addition_1_relu)
    Inception_1d_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_1d_5x5_reduce)
    Inception_1d_5x5 = tf.keras.layers.Conv2D(18, (5, 5), padding='same')(Inception_1d_5x5_relu_reduce)

    depthcat_1d = tf.keras.layers.Concatenate([Inception_1d_3x3,Inception_1d_1x1,Inception_1d_pool_proj,Inception_1d_5x5], axis=-1)
    Inception_1d_relu = tf.keras.layers.ReLU()(depthcat_1d)

    # Quinto bloque
    
    Inception_1e_3x3_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Inception_1d_relu)
    Inception_1e_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_1e_3x3_reduce)
    Inception_1e_3x3 = tf.keras.layers.Conv2D(18, (3, 3), padding='same')(Inception_1e_3x3_relu_reduce)
    
    Inception_1e_1x1 = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1d_relu)
    
    Inception_1e_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(Inception_1d_relu)
    Inception_1e_pool_proj = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1e_pool)

    
    Inception_1e_5x5_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Inception_1d_relu)
    Inception_1e_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_1e_5x5_reduce)
    Inception_1e_5x5 = tf.keras.layers.Conv2D(18, (5, 5), padding='same')(Inception_1e_5x5_relu_reduce)

    depthcat_1e = tf.keras.layers.Concatenate([Inception_1e_3x3,Inception_1e_1x1,Inception_1e_pool_proj,Inception_1e_5x5], axis=-1)
    
    #       Adición Layer

    Addition_2 = tf.keras.layers.Add()([depthcat_1e, Addition_1_relu])
    Addition_2_relu = tf.keras.layers.ReLU()(Addition_2)

    # Sexto bloque
    
    Inception_1f_3x3_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Addition_2_relu)
    Inception_1f_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_1f_3x3_reduce)
    Inception_1f_3x3 = tf.keras.layers.Conv2D(18, (3, 3), padding='same')(Inception_1f_3x3_relu_reduce)
    
    Inception_1f_1x1 = tf.keras.layers.Conv2D(18, (1, 1))(Addition_2_relu)
    
    Inception_1f_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(Addition_2_relu)
    Inception_1f_pool_proj = tf.keras.layers.Conv2D(18, (1, 1))(Inception_1f_pool)

    
    Inception_1f_5x5_reduce = tf.keras.layers.Conv2D(16, (1, 1))(Addition_2_relu)
    Inception_1f_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_1f_5x5_reduce)
    Inception_1f_5x5 = tf.keras.layers.Conv2D(18, (5, 5), padding='same')(Inception_1f_5x5_relu_reduce)

    depthcat_1f = tf.keras.layers.Concatenate([Inception_1f_3x3,Inception_1f_1x1,Inception_1f_pool_proj,Inception_1f_5x5], axis=-1)

    # Normalizacion
    Batch_normalization = tf.keras.layers.BatchNormalization()(depthcat_1f)

    # Aplanado
    Flatten = tf.keras.layers.Flatten()(Batch_normalization)
    # LSTM
    Lstm =  tf.keras.layers.LSTM(128, return_sequences=True)(Flatten)

    dense_classification = tf.keras.layers.Dense(num_classes, activation='softmax', name='classoutput')(Lstm)

    model = tf.keras.Model(inputs=sequence, outputs=dense_classification)

    return model


#Lectura de datos

# Directorio que contiene los archivos JSON
directorio = 'StaticData'

# Patrón para buscar archivos JSON
patron_archivos = os.path.join(directorio, '*.json')

# Obtener la lista de archivos JSON en el directorio
archivos_json = glob.glob(patron_archivos)

# Lista para almacenar los DataFrames
dataframes = []

# Iterar sobre los archivos JSON
for archivo_json in archivos_json:
    # Cargar el archivo JSON como DataFrame
    df = pd.read_json(archivo_json)
    # Agregar el DataFrame a la lista
    dataframes.append(df)

# Combinar los DataFrames en uno solo
df_completo = pd.concat(dataframes, ignore_index=True)

# Filtrar los valores nulos en la columna "Spectograms"
df_completo_filtrado = df_completo.dropna(subset=['Spectograms'])

# Divide los datos en entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Obtén las características y las etiquetas de entrenamiento y validación
x_train = train_df['Spectograms'].values
y_train = train_df['Gesture'].values
x_val = val_df['Spectograms'].values
y_val = val_df['Gesture'].values

# Convierte las características y las etiquetas en matrices NumPy
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

# Configuración de las opciones de entrenamiento
gpu_device = "/gpu:1"
max_epochs = 60
mini_batch_size = 64
initial_learn_rate = 0.04
learn_rate_drop_factor = 0.2
learn_rate_drop_period = 8
gradient_threshold = 1
validation_patience = 5

input_size = (13,24,8)
num_classes = 6

# Configuración del entorno GPU
with tf.device(gpu_device):
    # Definir la arquitectura de la red neuronal
    model = set_neural_network_architecture(input_size, num_classes)

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=initial_learn_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Configuración de las callbacks
lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * learn_rate_drop_factor if epoch % learn_rate_drop_period == 0 else lr)
early_stopping = EarlyStopping(monitor='val_loss', patience=validation_patience)

# Entrenamiento del modelo
history = model.fit(x_train, y_train, epochs=max_epochs, batch_size=mini_batch_size, 
                    validation_data=(x_val, y_val), shuffle=False, verbose=1,
                    callbacks=[lr_scheduler, early_stopping])



pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()


