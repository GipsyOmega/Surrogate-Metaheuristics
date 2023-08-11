import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("MH-Parameters2.csv")
print(df.columns)
y = df['Bat']
y_cuckoo = df['CSA']
y_pso = df['PSO']
y_hho = df['HHO']
y_woa = df['WOA']
y_gwo = df['GWO']

print(y_pso)
I = 81
x = np.linspace(0, I, I)  # X-axis points  # Y-axis points

plt.plot(x, y, label="BA")  # Plot the chart
plt.plot(x, y_cuckoo, label="CSA")
plt.plot(x, y_pso, label='PSO')
plt.plot(x, y_hho, label='HHO')
plt.plot(x, y_gwo, label='GWO')
plt.plot(x, y_woa, label='WOA')
plt.legend()

plt.show()
