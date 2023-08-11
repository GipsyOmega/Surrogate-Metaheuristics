''' Importing the required dependencies  '''
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from operator import truediv
import os
import math
from numpy import random
# import Workbench
import numpy as np
import surrogate
import CuckooSearchModified
from PSOSurrogate import PSO
from HHOSurrogate import HHO
from GWOSurrogate import GWO
from WOASurrogate import WOA
import time

pop_size = 5  # no of particles
pa = 0.1  # probability of discovering of alien egg
iterations = 50
d = 2  # dimensions
X = []  # create population array
dp1 = [0.879769816, 0.6694603527]
dp2 = [0.8380485831, 0.7578896747]
dp3 = [0.8634688274, 0.5516441819]
dp4 = [0.786453414, 0.6530175954]
dp5 = [0.8939850756, 0.7378509629]
bound = [(0, 1), (0, 1)]
# bound = [(-5, 5), (-5, 5)]
beta = 1.5
upper = 1
lower = 0
dim = 2

# 12:22


def initialize():
    global X
    X = np.stack((dp1, dp2, dp3, dp4, dp5))
    return X

''' Main Optimisation Class '''
class Bat():
    def __init__(self, pop_size=pop_size, dim=dim, iterations=iterations, lower=lower, upper=upper):
        self.pop_size = pop_size
        self.iterations = iterations

        self.X = initialize()  # Position
        self.Xs = self.X.copy()  # np.zeros((pop_size, dim))  # Solutions
        # print(self.X)
        self.V = np.zeros((pop_size, dim))  # Velocity
        self.f = np.random.randn(pop_size, 1)  # Frequency
        self.A = np.random.rand(pop_size, 1)  # Loudness
        self.r = np.random.rand(pop_size, 1)  # Pulse Width
        self.dim = dim  # Dimension

        self.Fmin = 0
        self.LB = [0]*self.dim
        self.UB = [0]*self.dim
        self.Fitness = [0]*self.pop_size
        self.pbest = [0]*self.pop_size
        self.lower = lower   # X domain
        self.upper = upper   # X domain
        self.gbest = 0

        # Hyperparameters
        self.beta = np.random.rand()  # random
        self.alpha = 0.9
        self.gamma = 0.9
        self.y = []

    def fitness(self, X):
        X = X.reshape(1, 2)
        # print(X.shape)
        obj = surrogate.obj(X)
        return obj

    def best_bat(self):
        i = 0
        j = 0
        for i in range(self.pop_size):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.dim):
            self.pbest[i] = self.Xs[j][i]
        self.Fmin = self.Fitness[j]
        self.y.append(self.Fmin)

    def bat_position(self):
        for i in range(self.dim):
            self.LB[i] = self.lower
            self.UB[i] = self.upper

        for i in range(self.pop_size):
            self.f[i] = 0
            # print(self.Xs[i])
            self.Fitness[i] = self.fitness(self.Xs[i])
        self.best_bat()

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def run(self):
        self.bat_position()
        t = 0
        y = []
        while (t < self.iterations):
            '''
            Initialising Frequency Max and Min
            '''
            a, b = tuple(np.random.randint(0, 10, 2))
            fmax = max(a, b)
            fmin = min(a, b)

            for i in range(self.pop_size):
                self.f[i] = fmin + (fmax - fmin)*self.beta
                for j in range(self.dim):
                    self.V[i][j] = self.V[i][j] + \
                        (self.Xs[i][j] - self.pbest[i])*self.f[i]
                    self.X[i][j] = self.Xs[i][j] + self.V[i][j]
                    self.X[i][j] = self.simplebounds(
                        self.X[i][j], self.LB[j], self.UB[j])

            for i in range(self.pop_size):
                #index = 0
                if (np.random.rand() > self.r[i]):
                    # freq = np.where(self.f == self.f.max())
                    for j in range(self.dim):
                        self.X[i][j] = self.pbest[j] + \
                            (np.random.uniform(-1, 1, 1)) * \
                            self.A[j]  # LOOK CLOSELY
                        self.X[i][j] = self.simplebounds(
                            self.X[i][j], self.LB[j], self.UB[j])

                    Fnew = self.fitness(self.X[i])

                    if (np.random.rand() < self.A[i]) and (Fnew <= self.Fitness[i]):
                        self.r[i] = self.r[i] * \
                            (1 - np.exp(-self.gamma*t))   # R increases
                        self.A[i] = self.A[i]*self.alpha   # Loudness decreases

                        for j in range(self.dim):
                            self.Xs[i][j] = self.X[i][j]
                        self.Fitness[i] = Fnew
                    '''
                    else:
                    freq = self.f.max()
                    for i in self.pop_size:
                        if self.f[i] == freq:
                            index = i
                    self.pbest[i] = self.X[index]

                    '''
                    if Fnew <= self.Fmin:
                        for j in range(self.dim):
                            self.pbest[j] = self.X[i][j]
                        self.Fmin = Fnew

            t += 1
            self.y.append(self.Fmin)
            # print(self.Fmin)

        return self.y


iterations = 81
bat = Bat(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

pso = PSO(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

hho = HHO(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

gwo = GWO(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

woa = WOA(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

''' Experimentation '''
N, I = 10, 81
start = time.time()
y = np.zeros((N, I))
y_hho = np.zeros((N, I))
y_pso = np.zeros((N, I))
y_cuckoo = np.zeros((N, I))
y_gwo = np.zeros((N, I))
y_woa = np.zeros((N, I))
for i in range(N):
    z1 = np.array(bat.run())
    z2 = np.array(hho.run())
    z3 = np.array(pso.run())
    z4 = np.array(CuckooSearchModified.cuckooSearch())
    z5 = np.array(gwo.run())
    z6 = np.array(woa.run())

    for k in range(I):
        y[i][k] = z1[k]
        y_hho[i][k] = z2[k]
        y_pso[i][k] = z3[k]
        y_cuckoo[i][k] = z4[k]
        y_gwo[i][k] = z5[k]
        y_woa[i][k] = z6[k]

y = np.sum(y, axis=0)/N
y = np.expand_dims(y, axis=1)
# print(y.shape)

y_hho = np.sum(y_hho, axis=0)/N
y_hho = np.expand_dims(y_hho, axis=1)

y_pso = np.sum(y_pso, axis=0)/N
y_pso = np.expand_dims(y_pso, axis=1)

y_cuckoo = np.sum(y_cuckoo, axis=0)/N
y_cuckoo = np.expand_dims(y_cuckoo, axis=1)

y_gwo = np.sum(y_gwo, axis=0)/N
y_gwo = np.expand_dims(y_gwo, axis=1)

y_woa = np.sum(y_woa, axis=0)/N
y_woa = np.expand_dims(y_woa, axis=1)

ends = time.time()
diff = ends - start
print("Time Taken:", diff)
#y = bat.run()
#y_pso = pso.run()
#y_hho = hho.run()
#y_cuckoo = CuckooSearchModified.cuckooSearch()
final = np.concatenate([y, y_cuckoo, y_gwo, y_woa, y_hho, y_pso], axis=1)
df = pd.DataFrame(final, columns=['Bat', 'CSA', 'GWO', 'WOA', 'HHO', 'PSO'])
df.to_csv("MH-Parameters.csv", index_label=False)

x = np.linspace(0, I, I)  # X-axis points  # Y-axis points

''' Plotting the graphs '''

plt.plot(x, y, label="BA")  # Plot the chart
plt.plot(x, y_cuckoo, label="CSA")
plt.plot(x, y_pso, label='PSO')
plt.plot(x, y_hho, label='HHO')
plt.plot(x, y_gwo, label='GWO')
plt.plot(x, y_woa, label='WOA')


plt.legend()
plt.show()  # display
