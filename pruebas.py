import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_training_testing_samples(data_dir, user):
        file_path = os.path.join(data_dir, user, user + '.json')
        with open(file_path) as f:
            user_data = json.load(f)
        # Extract samples
        print(data_dir,user,file_path)
        print(user_data)
        emg_sampling_rate = user_data['generalInfo']['samplingFrequencyInHertz']
        training_samples = user_data['trainingSamples']
        testing_samples = user_data['testingSamples']
        device_type = user_data['generalInfo']['deviceModel']
        return training_samples, testing_samples, emg_sampling_rate, device_type


frame_ground_truth=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

frame_ground_truth = np.array(frame_ground_truth)

total_ones = np.sum(frame_ground_truth == 1)

from scipy import signal
window = signal.hamming(24)

print(window)

#print("Total ones: ", total_ones)

#with open('DatastoresLSTM\\training\\Myo Armband\\waveIn\\user_032-train-0-[390-765].json', 'r') as f:
#    data = json.load(f)
"""
with open('SeñalEMGFiltrada.json', 'r') as f:
    data = json.load(f)

grafica = data[0]


df = pd.DataFrame(grafica)

print(df)

my_plot = df.plot()
plt.show()


señal = df[0].to_numpy()

f, t, Sxx = signal.spectrogram(señal, 0.2)

# Graficar espectrograma
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.show()

"""
#df = pd.DataFrame.from_dict(data['sequenceData'])

df2 = pd.read_json('dato5.json')

#print(np.fromstring(df['Spectograms'][0]))

spectrogram = np.array(df2["Spectograms"][19])

print(spectrogram[:,:,0].shape)

# Seleccionar el canal a graficar
channel = 0

plt.pcolormesh(23, 12, spectrogram[:,:,0], shading='gouraud')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.show()
