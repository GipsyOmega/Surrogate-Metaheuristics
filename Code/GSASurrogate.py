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
beta = 1.5
upper = 1
lower = 0
dim = 2

# 12:22

# NEW FMIN = 0.015114015783183277


def initialize():
    global X
    X = np.stack((dp1, dp2, dp3, dp4, dp5))
    return X


class HHO():
    def __init__(self, pop_size=pop_size, dim=dim, iterations=iterations, lower=lower, upper=upper):
        self.pop_size = pop_size
        self.iterations = iterations

        self.X = initialize()  # Position
        self.Xs = self.X.copy()  # np.zeros((pop_size, dim))  # Solutions
        #self.pbest = np.zeros((pop_size))
        self.V = np.zeros((pop_size, dim))
        self.F = np.zeros((pop_size, dim))

        self.m = np.zeros((pop_size))
        self.M = np.zeros((pop_size))
        self.A = np.zeros((pop_size))
        self.dim = dim  # Dimension

        self.Fmin = 100
        self.LB = [0]*self.dim
        self.UB = [0]*self.dim
        self.Fitness = [0]*self.pop_size
        self.lower = lower   # X domain
        self.upper = upper   # X domain
        self.gbest = 0
        self.epsilon = 0.02

        # Hyperparameters
        self.beta = 1.5  # random
        self.G0 = 3
        self.alpha = 0.7
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

        self.worst = max(self.Fitness)  # WORST FITNESS
        self.best = min(self.Fitness)  # BEST FITNESS

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

    def eucludian_distance(self, X, Y):
        x1, x2 = X
        y1, y2 = Y
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return dist.astype(int)

    def run(self):
        self.bat_position()
        t = 0
        while (t < self.iterations):

            self.G = self.G0*(1/self.iterations)**self.alpha
            for i in range(self.pop_size):
                self.m[i] = (self.Fitness[i] - self.worst) / \
                    (self.best - self.worst)

            for i in range(self.pop_size):
                self.M[i] = self.m[i]/sum(self.m)
                # j = 0  # j!=i

                for j in range(self.pop_size):
                    if j != i:
                        X, Y = self.Xs[i], self.Xs[j]
                        d = self.eucludian_distance(X, Y)
                        self.F[i] = self.G*self.M[i]*self.M[j] * \
                            (np.sum(self.Xs[i] - self.Xs[j]))/2 / \
                            (d + self.epsilon)

                        self.A[i] = np.sum(
                            random.random()*np.array(self.F[i]))/self.M[i]
                        self.V[i][j] = random.random()*self.V[i][j] + self.A[i]

                        self.X[i][j] = self.Xs[i][j] + self.V[i][j]
                        self.X[i][j] = self.simplebounds(
                            self.X[i][j], self.LB[j], self.UB[j])

            for i in range(self.pop_size):
                #index = 0
                Fnew = self.fitness(self.gbest)

                self.Fitness[i] = self.fitness(self.X[i])

                if self.Fitness[i] < self.fitness(self.gbest):
                    self.gbest = self.X[i]
                    Fnew = self.fitness(self.gbest)

                for j in range(self.dim):
                    self.Xs[i][j] = self.X[i][j]

                if Fnew <= self.Fmin:
                    self.Fmin = Fnew

            self.worst = max(self.Fitness)
            self.best = min(self.Fitness)  # / self.Fmin
            t += 1
            self.y.append(self.Fmin)
            # print(self.Fmin)

        return self.y


pso = HHO(pop_size=pop_size, dim=dim,
          iterations=iterations, lower=lower, upper=upper)

y = pso.run()
