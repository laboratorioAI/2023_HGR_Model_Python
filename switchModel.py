import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import pandas as pd
from Shared import Shared
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime 
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to prepare data
def prepareData(repetitionData, samplingF):
    factorConversion = 50 / samplingF
    window = 1.5
    guessMat = np.zeros((len(repetitionData), 12))
    labels = []
    for i in range(len(repetitionData)):
        sample = repetitionData["idx_"+str(i+1 )]
        emg = Shared.preprocess_signal(sample["emg"])
        if str(sample["gestureName"]) in ['waveIn', 'waveOut', 'open', 'pinch', 'fist', 'relax']:
            labels.append("Static")
        else:
            labels.append("Dynamic")
        quat = sample["quaternion"]
        arregloQuats = np.array([quat[key] for key in quat])
        #if(i > Shared.numGestureRepetitions):
        #    inicio = sample['groundTruthIndex'][0]
        #else:
        inicio = 0
        emgFinishIdx = inicio + window * samplingF
        qStart = int(np.ceil(inicio * factorConversion))
        qFinish = int(np.floor(inicio * factorConversion) + 75)
        quatSignal_transpose = np.transpose(arregloQuats)
        emg_transpose = np.transpose(emg)
        if emgFinishIdx > len(emg_transpose) or qFinish > len(quatSignal_transpose):
            emgRms = energy(emg_transpose[inicio:])
            quatRms = energy(quatSignal_transpose[qStart:])
        else:
            emgRms = energy(emg_transpose[inicio:int(emgFinishIdx)])
            quatRms = energy(quatSignal_transpose[qStart:qFinish])
        rmsValues = np.concatenate([emgRms, quatRms])
        guessMat[i, :] = rmsValues

    labels = pd.Categorical(labels, categories=['Static', 'Dynamic'], ordered=True)
    return {'guesses': guessMat, 'labels': labels}


# Energy function
def energy(signal):
    energy = []
    for i in range(signal.shape[1]):
        energy.append(np.sum(np.abs((signal[1:, i] * np.abs(signal[1:, i])) - (signal[:-1, i] * np.abs(signal[:-1, i])))))
    return energy


# DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = "D:\Marco Salazar Tesis\DatasetJSON"
trainingDir = 'JSONtesting'

# GET THE USERS DIRECTORIES
users, trainingPath = Shared.get_users(dataDir, trainingDir)


# Getting users data    
window = 1.5
numGuesses = len(users) * 180  # 180 from 12 gestures * 15 repetitions
guessMatrix = []
labels = []
testingGuessMat = []
testingLabels = []

for user in users:
    # Get user samples
    trainingSamples, validationSamples, emgSamplingRate, deviceType, userName = Shared.get_training_testing_samples(trainingPath, user)
    samplingF = emgSamplingRate
    factorConversion = 50 / samplingF
    datosEntrenamiento = prepareData(trainingSamples, samplingF)
    guessMatrix.extend(datosEntrenamiento["guesses"])
    labels.extend(datosEntrenamiento["labels"])
    #datosTesting = prepareData(validationSamples, samplingF)
    #testingGuessMat.extend(datosTesting["guesses"])
    #testingLabels.extend(datosTesting["labels"])

# Randomize matrices
guessMatrixR, labelsR = shuffle(guessMatrix, labels, random_state=0)
#testingGuessMatR, testingLabelsR = shuffle(testingGuessMat, testingLabels, random_state=0)
#testingGuessMat.extend(guessMatrix)
#testingLabels.extend(labels)

X_train, X_test, y_train, y_test = train_test_split(guessMatrix, labels, test_size=0.2, random_state=42)
print(X_train[0].shape)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Creamos el encoder para las etiquetas
label_encoder = LabelEncoder()

# Codificamos las etiquetas de entrenamiento
y_train_encoded = label_encoder.fit_transform(y_train)

y_test_encoded = label_encoder.fit_transform(y_test)

# Creamos el clasificador k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=4)  # Puedes ajustar el valor de 'n_neighbors'

# Entrenamos el clasificador k-NN
knn_classifier.fit(X_train, y_train)

# Hacemos predicciones en el conjunto de prueba
y_pred = knn_classifier.predict(X_test)

# Evaluamos la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Obtener la matriz de confusión
confusion_mat = confusion_matrix(y_test, y_pred)

# Crear una figura y un eje para el gráfico
fig, ax = plt.subplots(figsize=(8, 6))

# Crear el mapa de calor de la matriz de confusión
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", ax=ax)

# Configurar etiquetas y títulos utilizando el mapeo inverso de etiquetas
labels = label_encoder.inverse_transform(np.unique(y_test_encoded))
ax.set_xticklabels(labels, rotation=45)
ax.set_yticklabels(labels, rotation=0)
ax.set_xlabel("Predicción")
ax.set_ylabel("Valor Real")
ax.set_title("Matriz de Confusión")

# Mostrar el gráfico
plt.show()

# Guardar el gráfico como imagen
fig.savefig("matriz_confusion.png")

# Guardamos el modelo en un archivo
joblib.dump(knn_classifier, 'modelo_knn.joblib')
"""
# Creating the model
model = LogisticRegression(max_iter=10000)
model.fit(guessMatrixR, labelsR)

# Obtener la fecha y hora actual
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Nombre de la carpeta con la fecha y hora
folder_name = f"resultSwitch_{current_datetime}"

# Crear la carpeta
os.makedirs(folder_name)

# Nombre del archivo del modelo con la fecha y hora
model_filename = f"{folder_name}/modelo_{current_datetime}.pickle"

# Guardar el modelo en un archivo
with open(model_filename, 'wb') as archivo:
    pickle.dump(model, archivo)

# Calcular las probabilidades de predicción para los datos de entrenamiento y prueba
train_probs = model.predict_proba(guessMatrix)[:, 1]
test_probs = model.predict_proba(testingGuessMat)[:, 1]

# Graficar la distribución de las probabilidades de predicción para los datos de entrenamiento y prueba
plt.figure(figsize=(10, 6))
sns.histplot(train_probs, kde=True, color='blue', label='Static')
sns.histplot(test_probs, kde=True, color='orange', label='Dynamic')
plt.xlabel('Probabilidad de predicción')
plt.ylabel('Frecuencia')
plt.title('Distribución de las probabilidades de predicción')
plt.legend()
plt.savefig(f"{folder_name}/DistribucionProba.png")

# Calcular y mostrar la precisión del modelo en los datos de entrenamiento y prueba
train_accuracy = model.score(guessMatrix, labels)
test_accuracy = model.score(testingGuessMat, testingLabels)
print(f'Exactitud en datos de entrenamiento: {train_accuracy:.4f}')
print(f'Exactitud en datos de prueba: {test_accuracy:.4f}')

# Testing
# Total errors
errTrain = log_loss(labels, model.predict_proba(guessMatrix))
errTest = log_loss(testingLabels, model.predict_proba(testingGuessMat))

# Compute validation values
trainClassfTotal = (1 - errTrain) * 100
testingClassfTotal = (1 - errTest) * 100

print("Training",trainClassfTotal)
print("Testing",testingClassfTotal)

# Obtener las predicciones del modelo en los datos de prueba
predictions = model.predict(testingGuessMat)

# Calcular la matriz de confusión
cm = confusion_matrix(testingLabels, predictions, labels=['Static', 'Dynamic'])

# Crear un mapa de calor para visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.savefig(f"{folder_name}/matriz_confusion.png")
"""