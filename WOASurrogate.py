from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
import numpy as np
import surrogate
import math

pop_size = 5  # no of particles
pa = 0.1  # probability of discovering of alien egg
iterations = 51
d = 2  # dimensions
X = []  # create population array
dp1 = [0.879769816, 0.6694603527]
dp2 = [0.8380485831, 0.7578896747]
dp3 = [0.8634688274, 0.5516441819]
dp4 = [0.786453414, 0.6530175954]
dp5 = [0.8939850756, 0.7378509629]
bound = [(0, 1), (0, 1)]
# bound = [(-5, 5), (-5, 5)]
upper = 1
lower = 0
dim = 2

# 12:22

# NEW FMIN = 0.015114015783183277


def initialize():
    global X
    X = np.stack((dp1, dp2, dp3, dp4, dp5))
    return X


class WOA():
    def __init__(self, pop_size=pop_size, dim=dim, iterations=iterations, lower=lower, upper=upper):
        self.pop_size = pop_size
        self.iterations = iterations

        self.X = initialize()  # Position
        self.Xs = self.X.copy()  # np.zeros((pop_size, dim))  # Solutions
        #self.pbest = np.zeros((pop_size))
        self.dim = dim  # Dimension

        self.Fmin = 100
        self.LB = [0]*self.dim
        self.UB = [0]*self.dim
        self.Fitness = [0]*self.pop_size
        self.lower = lower   # X domain
        self.upper = upper   # X domain
        self.gbest = 0

        # Hyperparameters

        self.y = []
        self.b = 0.5
        self.l = 0.5

        self.r = random.random()

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

        # BEST FITNESS

        self.gbest = self.Xs[j]
        self.Fmin = self.Fitness[j]  # GBEST INITIALIZE/ BEST FITNESS
        self.y.append(self.Fmin)

    def bat_position(self):
        for i in range(self.dim):
            self.LB[i] = self.lower
            self.UB[i] = self.upper

        for i in range(self.pop_size):
            self.Fitness[i] = self.fitness(self.Xs[i])
        self.best_bat()

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def random_k(self, i):
        k = random.randint(0, self.pop_size-1)
        while(k == i):
            if k == self.pop_size-1:
                k -= 1
            else:
                k += 1
        return k

    def run(self):
        self.bat_position()
        t = 0
        while (t < self.iterations):

            self.a = 2 * (1 - t/self.iterations)
            self.C = 2 * random.random()
            self.A = 2 * random.random() * self.a - self.a
            self.p = random.random()

            for i in range(self.pop_size):
                if self.p < 0.5:
                    if abs(self.A) < 1:
                        D = abs(self.C*self.Xs[i] - self.gbest)
                        self.X[i] = self.gbest - self.A*D
                    elif abs(self.A) >= 1:
                        k = self.random_k(i)
                        D = abs(self.C*self.Xs[k] - self.Xs[i])
                        self.X[i] = self.Xs[k] - self.A*D

                elif self.p >= 0.5:
                    D = abs(self.gbest - self.Xs[i])
                    self.X[i] = D*np.exp(self.b*self.l) * \
                        np.cos(2*3.14*self.l) + self.gbest

            for i in range(self.pop_size):
                Fnew = self.fitness(self.gbest)
                self.Fitness[i] = self.fitness(self.X[i])

                if self.Fitness[i] < self.fitness(self.gbest):
                    self.gbest = self.X[i]
                    Fnew = self.fitness(self.gbest)

                self.Xs[i] = self.X[i]
                if Fnew <= self.Fmin:
                    self.Fmin = Fnew

            # / self.Fmin
            t += 1
            self.y.append(self.Fmin)
            # print(self.Fmin)

        return self.y


# pso = GWO(pop_size=pop_size, dim=dim,
#           iterations=iterations, lower=lower, upper=upper)

# y = pso.run()
