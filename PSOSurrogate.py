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
beta = 1.5
upper = 1
lower = 0
dim = 2

# 12:22


def initialize():
    global X
    X = np.stack((dp1, dp2, dp3, dp4, dp5))
    return X


class PSO():
    def __init__(self, pop_size=pop_size, dim=dim, iterations=iterations, lower=lower, upper=upper):
        self.pop_size = pop_size
        self.iterations = iterations

        self.X = initialize()  # Position
        self.Xs = self.X.copy()  # np.zeros((pop_size, dim))  # Solutions
        self.V = np.random.rand(pop_size, dim)  # Velocity
        #self.pbest = np.zeros((pop_size))
        self.dim = dim  # Dimension

        self.c1r1, self.c2r2 = 2*random.random(), 2*random.random()
        self.weight = 0.1

        self.Fmin = 100
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
        for i in range(self.pop_size):
            self.pbest[i] = self.Xs[i]

        self.gbest = self.pbest[j]
        self.Fmin = self.Fitness[j]  # GBEST INITIALIZE
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

    def run(self):
        self.bat_position()
        t = 0
        while (t < self.iterations):
            for i in range(self.pop_size):
                for j in range(self.dim):
                    self.V[i][j] = self.weight*np.array(self.V[i][j]) + self.c1r1 * \
                        (self.pbest[i][j] - self.Xs[i][j]) + \
                        self.c2r2*(self.gbest[j] - self.Xs[i][j])
                    self.X[i][j] = self.Xs[i][j] + self.V[i][j]
                    self.X[i][j] = self.simplebounds(
                        self.X[i][j], self.LB[j], self.UB[j])

            for i in range(self.pop_size):
                #index = 0
                Fnew = self.fitness(self.gbest)
                if self.fitness(self.pbest[i]) > self.fitness(self.X[i]):
                    self.pbest[i] = self.X[i]
                self.Fitness[i] = self.fitness(self.pbest[i])
                if self.fitness(self.pbest[i]) < self.fitness(self.gbest):
                    self.gbest = self.pbest[i]
                    Fnew = self.fitness(self.gbest)

                for j in range(self.dim):
                    self.Xs[i][j] = self.X[i][j]
                #self.Fitness[i] = Fnew

                if Fnew <= self.Fmin:
                    self.Fmin = Fnew
            t += 1
            self.y.append(self.Fmin)
            # print(self.Fmin)

        return self.y


pso = PSO(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

y = pso.run()
