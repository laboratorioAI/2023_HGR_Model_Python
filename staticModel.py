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
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import datetime 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

#Lectura de datos

# Directorio que contiene los archivos JSON
directorio = 'D:\Marco Salazar Tesis\CodigoTesis\StaticData'

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

scaler = MinMaxScaler(feature_range=(0, 1))

# Convierte los elementos de la columna "Spectograms" en arrays de NumPy
df_completo["Spectograms"] = df_completo["Spectograms"].apply(np.array)

df_completo["Spectograms"] = df_completo["Spectograms"].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten().reshape((13,24,8)))

# Restaura la forma original de los datos
df_completo["Spectograms"] = df_completo["Spectograms"].apply(lambda x: x.reshape(13, 24, 8))

# Guarda el scaler en un archivo binario
#with open('scalerStaticTestingAll.pkl', 'wb') as f:
#    pickle.dump(scaler, f)

#print(df_completo.at[0, "Spectograms"])
#print(type(df_completo.at[0, "Spectograms"]))
#print((df_completo.at[0, "Spectograms"].shape))


#scaler.fit(df_completo["Spectograms"].values.reshape(-1, 1))


#with open('escalador.pkl', 'wb') as f:
#    pickle.dump(scaler, f)


# Filtrar los valores nulos en la columna "Spectograms"
#df_completo_filtrado = df_completo.dropna(subset=['Spectograms'])

#df_completo.to_json('static.json', orient='records')
# Muestra de los datos
#print(df_completo)
#df_completo.to_json('static.json', orient='records')
# Divide los datos en entrenamiento y validación
train_df, val_df = train_test_split(df_completo, test_size=0.2, random_state=42)


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

#Creación de variables nuevas
x_train_reshaped = []
x_val_reshaped = []

# Aplicar el remodelado a cada valor de la lista
for valor in x_train:
    #print(valor)
    arreglo_reshaped = np.array(valor).reshape((13, 24, 8))
    x_train_reshaped.append(arreglo_reshaped)

# Aplicar el remodelado a cada valor de la lista
for valor in x_val:
    arreglo_reshaped = np.array(valor).reshape((13, 24, 8))
    x_val_reshaped.append(arreglo_reshaped)

# Conversión a numpyArray 
x_train = np.array(x_train_reshaped)
x_val = np.array(x_val_reshaped)

modelS = tf.keras.models.load_model('D:/Marco Salazar Tesis/resultSLast_20230806_144454/modelo_20230806_144454.h5')
label_encoder = LabelEncoder()

# Predecir las etiquetas para los datos de validación
y_val_pred = modelS.predict(x_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Obtener las etiquetas verdaderas en formato numérico
y_val_true_encoded = label_encoder.fit_transform(y_val)

# Obtener la matriz de confusión
confusion_mat = confusion_matrix(y_val_true_encoded, y_val_pred_classes)

# Crear una figura y un eje para el gráfico
fig, ax = plt.subplots(figsize=(8, 6))

# Crear el mapa de calor de la matriz de confusión con etiquetas
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", ax=ax)

# Configurar etiquetas y títulos utilizando el mapeo inverso de etiquetas
labels = label_encoder.inverse_transform(np.unique(y_val_true_encoded))
ax.set_xticklabels(labels, rotation=45)
ax.set_yticklabels(labels, rotation=0)
ax.set_xlabel("Predicción")
ax.set_ylabel("Valor Real")
ax.set_title("Matriz de Confusión")

# Mostrar el gráfico
plt.show()

# Guardar el gráfico como imagen
fig.savefig("matriz_confusionS.png")

############################################################################################

# Función para generación de bloque Inception


def inception_block(sequence, num_filters):
    # 3x3 convolution branch
    Inception_3x3_reduce = tf.keras.layers.Conv2D(num_filters[0], (1, 1))(sequence)
    Inception_3x3_relu_reduce = tf.keras.layers.ReLU()(Inception_3x3_reduce)
    Inception_3x3 = tf.keras.layers.Conv2D(num_filters[1], (3, 3), padding='same')(Inception_3x3_relu_reduce)

    # 1x1 convolution branch
    Inception_1x1 = tf.keras.layers.Conv2D(num_filters[2], (1, 1))(sequence)

    # Max pooling branch
    Inception_pool = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=(1,1))(sequence)
    Inception_pool_proj = tf.keras.layers.Conv2D(num_filters[3], (1, 1))(Inception_pool)

    # 5x5 convolution branch
    Inception_5x5_reduce = tf.keras.layers.Conv2D(num_filters[4], (1, 1))(sequence)
    Inception_5x5_relu_reduce = tf.keras.layers.ReLU()(Inception_5x5_reduce)
    Inception_5x5 = tf.keras.layers.Conv2D(num_filters[5], (5, 5), padding='same')(Inception_5x5_relu_reduce)

    # Concatenate all branches
    depthcat = tf.keras.layers.Concatenate(axis=-1)([Inception_3x3, Inception_1x1, Inception_pool_proj, Inception_5x5])
    
    return depthcat

# Función generadora de modelo

# Function model
def set_neural_network_architecture(input_size, num_classes, filtros_a, filtros_b, filtros_c,
                                    filtros_d, filtros_e, filtros_f, hidden_units):

    # Entradas
    sequence=tf.keras.layers.Input(input_size)

    # Primer bloque Inception
    Inception_a = inception_block(sequence, filtros_a)

    # Layer ReLU
    ReLu_a = tf.keras.layers.ReLU()(Inception_a)

    # Segundo bloque Inception
    Inception_b = inception_block(ReLu_a, filtros_b)

    # Layer ReLU
    ReLu_b = tf.keras.layers.ReLU()(Inception_b)

    # Tercer bloque Inception
    Inception_c = inception_block(ReLu_b, filtros_c)

    # Capa de Adición
    Addition_1 = tf.keras.layers.Add()([Inception_c, ReLu_a])

    # Layer ReLU
    ReLu_c = tf.keras.layers.ReLU()(Addition_1)

    # Cuarto bloque Inception
    Inception_d = inception_block(ReLu_c, filtros_d)

    # Layer ReLU
    ReLu_d = tf.keras.layers.ReLU()(Inception_d)

    # Quinto bloque Inception
    Inception_e = inception_block(ReLu_d, filtros_e)

    # Capa de Adición
    Addition_2 = tf.keras.layers.Add()([Inception_e, Addition_1])

    # Layer ReLU
    ReLu_e = tf.keras.layers.ReLU()(Addition_2)

    # Sexto bloque Inception
    Inception_f = inception_block(ReLu_e, filtros_f)

    # Capa de Normalización
    Batch_normalization = tf.keras.layers.BatchNormalization()(Inception_f)#(ReLu_d)#

    # Capa de Aplanado
    
    Flatten = tf.keras.layers.Flatten()(Batch_normalization)

    # Cambio de forma para adaptarse a LSTM (Reshape)
    Reshape = tf.keras.layers.Reshape((1, 22464))(Flatten)

    # Capa LSTM
    Lstm = tf.keras.layers.LSTM(hidden_units, return_sequences=False)(Reshape)

    # Dropout
    dropout = tf.keras.layers.Dropout(0.2)(Lstm)

    # Capa de clasificación (Densa)
    dense_classification = tf.keras.layers.Dense(num_classes, activation='softmax', name='classoutput')(dropout)

    # Generación del modelo
    model = tf.keras.Model(inputs=sequence, outputs=dense_classification)

    return model

# Configuración de las opciones de entrenamiento
gpu_device = "/gpu:1"
max_epochs = 10
mini_batch_size = 32
initial_learn_rate = 0.0005
learn_rate_drop_factor = 0.2
learn_rate_drop_period = 8
gradient_threshold = 1
validation_patience = 5

input_size = (13,24,8)
num_classes = 6

#Filtros a usar

filtro1 = [12, 14, 14, 14, 12, 14]
filtro2 = [14, 16, 16, 16, 14, 16]
filtro3 = [16, 18, 18, 18, 16, 18]
filtro4 = [18, 20, 20, 20, 18, 20]

log_dir = "D:\Marco Salazar Tesis\TensorBoard3"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Unidades LST100
hidden_units = 128

# Configuración del entorno GPU
with tf.device(gpu_device):
    # Definir la arquitectura de la red neuronal
    model = set_neural_network_architecture(input_size, num_classes, filtro3, filtro3, filtro3,
                                            filtro3, filtro3, filtro3, hidden_units)
    
    # Cargar el modelo guardado
    #model = tf.keras.models.load_model('D:/Marco Salazar Tesis/result_20230618_033334/modelo_20230618_033334.h5')

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=initial_learn_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
# Convertir las etiquetas en valores numéricos
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)


# Obtener el número de clases
num_classes = len(label_encoder.classes_)

# Convertir las etiquetas en one-hot encoding
y_train_encoded = to_categorical(y_train_encoded, num_classes=num_classes)
y_val_encoded = to_categorical(y_val_encoded, num_classes=num_classes)


# Configuración de las callbacks
lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * learn_rate_drop_factor if epoch % learn_rate_drop_period == 0 else lr)
early_stopping = EarlyStopping(monitor='val_loss', patience=validation_patience)

# Entrenamiento del modelo
history = model.fit(x_train, y_train_encoded, epochs=max_epochs, batch_size=mini_batch_size, 
                    validation_data=(x_val, y_val_encoded), shuffle=False, verbose=1,
                    callbacks=[lr_scheduler, early_stopping,tensorboard_callback])

#################GUARDADO DEL MODELO#################

# Obtener la fecha y hora actual
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Nombre de la carpeta con la fecha y hora
folder_name = f"result_{current_datetime}"

# Crear la carpeta
os.makedirs(folder_name)

# Nombre del archivo del modelo con la fecha y hora
model_filename = f"{folder_name}/modelo_{current_datetime}.h5"

# Guardar el modelo
model.save(model_filename)

# Crear y guardar gráfica de pérdida
loss_df = pd.DataFrame(history.history)[['loss', 'val_loss']]
loss_df.plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(f"{folder_name}/loss_plot.png")  # Guardar la gráfica de pérdida en un archivo dentro de la carpeta

# Crear y guardar gráfica de precisión
accuracy_df = pd.DataFrame(history.history)[['accuracy', 'val_accuracy']]
accuracy_df.plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig(f"{folder_name}/accuracy_plot.png")  # Guardar la gráfica de precisión en un archivo dentro de la carpeta


# Predecir las etiquetas para los datos de validación
y_val_pred = model.predict(x_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Obtener las etiquetas verdaderas en formato numérico
y_val_true_encoded = label_encoder.transform(y_val)

# Crear la matriz de confusión
confusion_mat = confusion_matrix(y_val_true_encoded, y_val_pred_classes)

# Crear el gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Verdaderas')
plt.title('Matriz de Confusión')

# Guardar la imagen de la matriz de confusión
plt.savefig(f"{folder_name}/matriz_confusion.png")

print("Matriz de Confusión guardada como imagen correctamente.")


# Imprime el resumen del modelo para ver la cantidad de parámetros
model.summary()

# Calcula el número total de parámetros del modelo
num_params = model.count_params()
print("Número total de parámetros del modelo:", num_params)
