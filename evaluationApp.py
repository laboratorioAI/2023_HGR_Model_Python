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
from Shared import Shared
import joblib
import time
import json
import sys
from collections import Counter

# Crear la estructura de datos inicial
reales = {
    "testing": {}
}

def prepareData(repetitionData, samplingF, data, userName):
    factorConversion = 50 / samplingF
    window = 1.5
    guessMat = np.zeros((len(repetitionData), 12))
    contador = 0
    for i in range(len(repetitionData)):
        numSample = "idx_"+str(i)
        sample = repetitionData["idx_"+str(i+1)]
        emg = Shared.preprocess_signal(sample["emg"])
        quat = sample["quaternion"]
        arregloQuats = np.array([quat[key] for key in quat])
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
        # Add to dataframe
        #print(data)
        #print( pd.Series([userName, emg_transpose, quatSignal_transpose, i, rmsValues, samplingF], index=data.columns))
        data.loc[len(data)] = [userName, emg, arregloQuats, numSample, rmsValues, samplingF]
        contador += 1
    return {'guesses': guessMat}

# Energy function
def energy(signal):
    energy = []
    for i in range(signal.shape[1]):
        energy.append(np.sum(np.abs((signal[1:, i] * np.abs(signal[1:, i])) - (signal[:-1, i] * np.abs(signal[:-1, i])))))
    return energy

# Generate Spectogram EMG
def generateFramesEMG(signal, emg_sampling_rate, userName, repetition):
    # Allocate space for the results
    num_windows = int(np.floor((signal.shape[1]-Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1)
    columnas = ['Spectograms', 'UserName', 'Repetition' ]
    dataSpectrogram = pd.DataFrame(columns=columnas)
    #print(num_windows)
    # Creating frames
    for i in range(num_windows):
        # Get signal data to create a frame
        translation = (i)*Shared.WINDOW_STEP_LSTM
        start = 0 + translation
        end = Shared.FRAME_WINDOW + translation
        frame_signal = signal[:,start:end]
        spectrograms = Shared.generate_spectrograms(frame_signal, emg_sampling_rate)
        #print(spectrograms)
        dataSpectrogram.loc[i] = [spectrograms, userName, repetition]
    return dataSpectrogram, num_windows

# Generate Spectogram QUAT
def generateQFrames(quatSignal, userName, repetition):
    quat_sampling_f = 50
    # Allocate space for the results
    #print(quatSignal)
    num_windows = int(np.floor((quatSignal.shape[1]- (Shared.FRAME_WINDOW_T * quat_sampling_f)) / 
                               (Shared.WINDOW_STEP_T * quat_sampling_f)) + 1)
    
    columnas = ['Spectograms', 'UserName', 'Repetition' ]
    dataSpectrogram = pd.DataFrame(columns=columnas)
    #print(num_windows)
    # Creating frames
    for i in range(num_windows):
        # Get signal data to create a frame
        translation = int((i)*Shared.WINDOW_STEP_T * quat_sampling_f)
        start = 0 + translation
        end = int(Shared.FRAME_WINDOW_T * quat_sampling_f) + translation
        # Get Spectrogram of the window
        quatSignal_transpose = np.transpose(quatSignal)
        frame_signal = quatSignal_transpose[start:end]
        quat_spectrograms = Shared.generate_quat_spectrogram(frame_signal, quat_sampling_f)
        #print(quat_spectrograms)
        dataSpectrogram.loc[i] = [quat_spectrograms, userName, repetition]
    return dataSpectrogram, num_windows

# Calculo puntos
def calculateVectorOfTimePoints(numWindows, samplingF, typeSignal):
    vectorOfTimePoints = []
    if(typeSignal == "Dynamic"):
        for i in range(numWindows):
            translation = int((i)*Shared.WINDOW_STEP_T * samplingF)
            start = 0 + translation
            timestamp = start + int((Shared.FRAME_WINDOW_T * samplingF) / 2)
            vectorOfTimePoints.append(timestamp)
    else:
        for i in range(numWindows):
            translation = (i)*Shared.WINDOW_STEP_LSTM
            start = 0 + translation
            timestamp = start + int(Shared.FRAME_WINDOW / 2)
            vectorOfTimePoints.append(timestamp)
    return vectorOfTimePoints

def set_no_gesture_use(with_no_gesture):
    if with_no_gesture:
        classes = ["fist", "relax", "open", "pinch", "waveIn", "waveOut", "up", "down", "left", "right", "forward", "backward"]
    else:
        classes = ["fist", "open", "pinch", "waveIn", "waveOut", "up", "down", "left", "right", "forward", "backward"]
    return classes

# FUNCTION TO POST PROCESS THE SAMPLE
def postprocessSample(labels, classes):

    if Shared.POSTPROCESS in ['1-1', '2-1', '1-2']:

        # Check the first label
        right = labels[0][0] == 'relax'
        if Shared.POSTPROCESS == '1-2':
            right = labels[0][1] == classes or labels[0][2] == 'relax'
        current = labels[0][0] == classes
        if right and current:
            labels[0][0] = 'relax'

        # Set start and finish for middle labels
        start = 1 if Shared.POSTPROCESS == '1-1' else 2
        finish = len(labels[0]) - 1 if Shared.POSTPROCESS == '1-1' else len(labels[0]) - 2
        if Shared.POSTPROCESS == '2-1':
            start = 2
        elif Shared.POSTPROCESS == '1-2':
            finish = len(labels[0]) - 3

        # Check for misclassified labels
        for i in range(start, finish + 1):

            # Check left-current-right classes
            left = labels[0][i - 1] == classes
            right = labels[0][i + 1] == classes
            if Shared.POSTPROCESS == '2-1':
                left = labels[0][i - 1] == classes or labels[0][i - 2] == classes
            elif Shared.POSTPROCESS == '1-2':
                right = labels[0][i + 1] == classes or labels[0][i + 2] == classes
            current = labels[0][i] != classes

            # Replace the class if matches the criterion
            if left and right and current:
                labels[0][i] = classes

            # Replace the class if matches the criterion
            if not left and not right and not current:
                labels[0][i] = 'relax'

        # Check the last label
        left = labels[0][len(labels[0]) - 2] == 'relax'
        if Shared.POSTPROCESS == '2-1':
            left = labels[0][len(labels[0]) - 2] == 'relax' or labels[0][len(labels[0]) - 3] == 'relax'
        current = labels[0][len(labels[0]) - 1] == classes

        # Replace the class if matches the criterion
        if left and current:
            labels[0][len(labels[0]) - 1] = 'relax'

    # Set wrong labels to relax
    for i in range(len(labels[0])):
        if labels[0][i] != classes:
            labels[0][i] = 'relax'

    # Transform to categorical
    labels = [pd.Categorical(labels[0], categories=Shared.setNoGestureUse(True))]

    return labels


# DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = "D:\Marco Salazar Tesis\DatasetJSON"
trainingDir = 'JSONTesting'

# GET THE USERS DIRECTORIES
users, trainingPath = Shared.get_users(dataDir, trainingDir)

#Create DataFrame
columnas = ['User','EMG','Quaternion', 'RepetitionNumber', 'Energy', 'SamplingRate']
data = pd.DataFrame(columns=columnas)

testingGuessMat = []

for user in users:
    # Get user samples
    trainingSamples, validationSamples, emgSamplingRate, deviceType, userName = Shared.get_training_testing_samples(trainingPath, user)
    samplingF = emgSamplingRate
    factorConversion = 50 / samplingF
    datosTesting = prepareData(validationSamples, samplingF, data, userName)
    #datosTesting = prepareData(validationSamples, samplingF)
    testingGuessMat.extend(datosTesting["guesses"])

#data.to_json('archivoTodo.json')

# Ruta al archivo del modelo guardado
model_filename = "D:/Marco Salazar Tesis/modelo_knn.joblib"

# Cargar el modelo Switch
#with open(model_filename, 'rb') as archivo:#
#    model = pickle.load(archivo)
loaded_knn_model = joblib.load('modelo_knn.joblib')

# Realizar predicciones en el dataframe
#print(data['Energy'])
energy_values = np.array(data['Energy'].values.tolist())

start_time_switch = time.time()
predictions = loaded_knn_model.predict(energy_values)


end_time_switch = time.time()
execution_time_switch = end_time_switch - start_time_switch

num_values_energy = energy_values.shape[0]

execution_time_switch_value = execution_time_switch / num_values_energy

data['type'] = predictions

conteo_valores = data['type'].value_counts()

print(conteo_valores)

#print(data['type'])

# Cargar el modelo Dinamico
modelD = tf.keras.models.load_model('D:/Marco Salazar Tesis/resultDLast_20230806_161316/modeloD_20230806_161316.h5')
# Carga el scaler Dinamico desde el archivo
scalerD = joblib.load('D:/Marco Salazar Tesis/scalerDynamicTestingAll.pkl')
# Mapea los índices a etiquetas correspondientes
mapeoD = {
    0: 'backward',
    1: 'down',
    2: 'forward',
    3: 'left',
    4: 'relax',
    5: 'right',
    6: 'up'
}

# Cargar el modelo Static
modelS = tf.keras.models.load_model('D:/Marco Salazar Tesis/resultSLast_20230806_144454/modelo_20230806_144454.h5')
# Carga el scaler Dinamico desde el archivo
scalerS = joblib.load('D:/Marco Salazar Tesis/scalerStaticTestingAll.pkl')
# Mapea los índices a etiquetas correspondientes
mapeoS = {
    0: 'fist',
    1: 'relax',
    2: 'open',
    3: 'pinch',
    4: 'waveIn',
    5: 'waveOut'
}

# Iterar por cada fila del dataframe
data.to_json('data.json', orient='records', lines=True)
#print(data)
numusuarios = 0
for index, row in data.iterrows():
    user_name = row['User'] 
    repetition_number = row['RepetitionNumber']
    predicts = []
    times = []
    predicted_classes = []
    # Create a separate dictionary for each user
    user_data = {
        "class": {},
        "vectorOfTimePoints": {},
        "vectorOfProcessingTime": {},
        "vectorOfLabels": {}
    }
    if user_name not in reales["testing"]:
        numusuarios+=1
        reales["testing"][user_name] = user_data
    print("Usuario ", numusuarios, " este usuario es: ", user_name, " de ", 43, " en repeticion: ", repetition_number , end="\r")
    sys.stdout.flush()
    if(row['type'] == "Static"):
        EMGValue = row['EMG']
        samplingRate = row['SamplingRate']
        start_time = time.time()
        frames, numWindows = generateFramesEMG(EMGValue, samplingRate, user_name, repetition_number)
        frames["Spectograms"] = frames["Spectograms"].apply(lambda x: scalerS.transform(x.reshape(-1, 1)).flatten().reshape((13,24,8)))
        samplingF = samplingRate
        spectroValues = frames['Spectograms'].values
        spectroValues = np.array(spectroValues)
        spectroValues_reshaped = []
        for valor in spectroValues:
            arreglo_reshaped = np.array(valor).reshape((13, 24, 8))
            spectroValues_reshaped.append(arreglo_reshaped)
        spectroValues = np.array(spectroValues_reshaped)
        prediction = modelS.predict(spectroValues, verbose=0)
        for predict in prediction:
            predicted_classes.append(np.argmax(predict)) 
        predicciones = [mapeoS[indice] for indice in predicted_classes]
        valores_filtrados = [valor for valor in predicciones if valor != "relax"]
        if valores_filtrados:
            valor_mas_comun = Counter(valores_filtrados).most_common(1)[0][0]
        else:
            valor_mas_comun = "relax"
        predicciones = [valor if valor == "relax" else valor_mas_comun for valor in predicciones]
        nueva_predicciones = []
        en_gesto = False
        for valor in predicciones:
            if valor != "relax":
                if en_gesto:
                    nueva_predicciones.append(valor_mas_comun)
                else:
                    nueva_predicciones.append(valor)
                en_gesto = True
            else:
                en_gesto = False
                nueva_predicciones.append(valor)
        end_time = time.time()
        execution_time = end_time - start_time
        num_predictions = prediction.shape[0]
        total_time = execution_time_switch_value + (execution_time / num_predictions)
        times = [total_time] * num_predictions
        translation = Shared.WINDOW_STEP_LSTM
    else:
        QuatValue = row['Quaternion']
        samplingRate = row['SamplingRate']
        repetition_number = row['RepetitionNumber']
        frames, numWindows = generateQFrames(QuatValue, user_name, repetition_number)
        samplingF = 50
        start_time = time.time()
        frames["Spectograms"] = frames["Spectograms"].apply(lambda x: scalerD.transform(x.reshape(-1, 1)).flatten().reshape((13,24,4)))
        spectroValues = frames['Spectograms'].values
        spectroValues = np.array(spectroValues)
        spectroValues_reshaped = []
        for valor in spectroValues:
            arreglo_reshaped = np.array(valor).reshape((13, 24, 4))
            spectroValues_reshaped.append(arreglo_reshaped)
        spectroValues = np.array(spectroValues_reshaped)
        prediction = modelD.predict(spectroValues, verbose=0)
        for predict in prediction:
            predicted_classes.append(np.argmax(predict))
        predicciones = [mapeoD[indice] for indice in predicted_classes]
        valores_filtrados = [valor for valor in predicciones if valor != "relax"]
        if valores_filtrados:
            valor_mas_comun = Counter(valores_filtrados).most_common(1)[0][0]
        else:
            valor_mas_comun = "relax"
        # Reemplaza los valores en predicciones por el valor más común (excepto "relax")
        predicciones = [valor if valor == "relax" else valor_mas_comun for valor in predicciones]
        
        nueva_predicciones = []
        en_gesto = False
        for valor in predicciones:
            if valor != "relax":
                if en_gesto:
                    nueva_predicciones.append(valor_mas_comun)
                else:
                    nueva_predicciones.append(valor)
                en_gesto = True
            else:
                en_gesto = False
                nueva_predicciones.append(valor)
        end_time = time.time()
        execution_time = end_time - start_time
        num_predictions = prediction.shape[0]
        total_time = execution_time_switch_value + (execution_time / num_predictions)
        times = [total_time] * num_predictions
        translation = Shared.WINDOW_STEP_T * 50
    vectorOfTimePoints = calculateVectorOfTimePoints(numWindows, samplingF, row['type'])

    num_repetition = str(repetition_number)
    reales["testing"][user_name]["class"].update({num_repetition: valor_mas_comun})
    reales["testing"][user_name]["vectorOfLabels"].update({num_repetition: nueva_predicciones})
    reales["testing"][user_name]["vectorOfProcessingTime"].update({num_repetition: times})
    reales["testing"][user_name]["vectorOfTimePoints"].update({num_repetition: vectorOfTimePoints})
archivo_json = "prediccionTrainingSamples1.json"
with open(archivo_json, "w") as file:
    json.dump(reales, file, indent=4)
