import json
import pandas as pd
import matplotlib.pyplot as plt

with open("DatasetJSON\JSONtraining\user_032\user_032.json") as archivo:
    datos = json.load(archivo)

print(datos)


grafica = datos['testingSamples']['idx_167']['emg']['ch5']


print(grafica)

df = pd.DataFrame(grafica)

my_plot = df.plot()
plt.show()