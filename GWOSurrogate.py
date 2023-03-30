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


class GWO():
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

    def best_wolves(self):
        ls = self.Fitness.copy()
        key = []
        for _ in range(3):
            j = np.argmin(ls)
            ls = np.delete(ls, j)
            key.append(j)
        X_a, X_b, X_c = self.Xs[key[0]], self.Xs[key[1]], self.Xs[key[2]]
        return X_a, X_b, X_c, key

    def run(self):
        self.bat_position()
        t = 0
        while (t < self.iterations):

            self.a = 2*(1 - t/self.iterations)
            self.c = 2 * random.random()
            self.A = 2 * random.random() * self.a - self.a

            X_a, X_b, X_c, key = self.best_wolves()
            self.new = []
            for i in range(self.pop_size):
                if i not in key:
                    self.new.append(i)

            for k in self.new:  # Omega Wolves index
                Y1 = X_a - (2 * random.random() * self.a - self.a) * \
                    abs(2*random.random()*X_a - self.Xs[k])
                Y2 = X_b - (2 * random.random() * self.a - self.a) * \
                    abs(2*random.random()*X_b - self.Xs[k])
                Y3 = X_c - (2 * random.random() * self.a - self.a) * \
                    abs(2*random.random()*X_c - self.Xs[k])
                self.X[k] = (Y1 + Y2 + Y3)/3

            for i in range(self.pop_size):
                Fnew = self.fitness(self.gbest)
                self.Fitness[i] = self.fitness(self.X[i])

                if self.Fitness[i] < self.fitness(X_a):
                    self.gbest = self.X[i]
                    Fnew = self.fitness(self.gbest)

                self.Xs[i] = self.X[i]

                if Fnew <= self.Fmin:
                    self.Fmin = Fnew

            # / self.Fmin
            t += 1
            self.y.append(self.Fmin)
            #print(self.Fmin)

        return self.y


# pso = HHO(pop_size=pop_size, dim=dim,
#           iterations=iterations, lower=lower, upper=upper)

# y = pso.run()
