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
        self.dim = dim  # Dimension

        self.Fmin = 100
        self.LB = [0]*self.dim
        self.UB = [0]*self.dim
        self.Fitness = [0]*self.pop_size
        self.lower = lower   # X domain
        self.upper = upper   # X domain
        self.gbest = 0

        # Hyperparameters
        self.beta = 1.5  # random
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

        self.gbest = self.Xs[j]
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

    def levyflight(self, beta):
        sigma = (math.gamma(1-beta)*math.sin(3.14*beta/2)) / \
            math.gamma((1+beta)/2)*beta*(2**(beta-1)/2)
        sigma = sigma**1/beta
        func = 0.01*random.random()*sigma
        v = random.random()
        func = 0.01*func/abs((v**1/beta))
        return func

    def run(self):
        self.bat_position()
        t = 0
        while (t < self.iterations):

            self.a = 2*(1 - t/self.iterations)
            self.q = random.random()

            self.E0 = 2*random.random() - 1
            self.J = 2*(1 - random.random())

            self.E = 2*self.E0*(1-t/self.iterations)

            for i in range(self.pop_size):
                if self.E >= 1:
                    if self.q >= 0.5:
                        for j in range(self.dim):
                            k = random.randint(0, self.pop_size)
                            self.X[i][j] = self.Xs[k][j] - random.random() * \
                                abs(self.Xs[k][j] - 2 *
                                    random.random()*self.Xs[i][j])

                    else:
                        self.Xm = np.sum(np.array(self.Xs),
                                         axis=0)/self.pop_size
                        self.Xm = list(self.Xm.reshape(2,))
                        for j in range(self.dim):
                            self.X[i][j] = (self.gbest[j] - self.Xm[j]) - random.random()*(
                                self.lower + random.random()*(self.upper - self.lower))

                elif self.E < 1:
                    if (self.r >= 0.5 and self.E >= 0.5):
                        for j in range(self.dim):
                            X_diff = self.gbest[j] - self.Xs[i][j]
                            self.X[i][j] = X_diff - self.E * \
                                abs(self.J*self.gbest[j] - self.Xs[i][j])

                    elif (self.r >= 0.5 and self.E < 0.5):
                        for j in range(self.dim):
                            X_diff = self.gbest[j] - self.Xs[i][j]
                            self.X[i][j] = self.gbest[j] - self.E * abs(X_diff)

                    elif (self.r < 0.5 and self.E >= 0.5):
                        # INSIDE THE DIM LOOP
                        for j in range(self.dim):
                            Y1 = self.gbest[j] - self.E * \
                                abs(self.J*self.gbest[j] - self.Xs[i][j])

                        Z1 = Y1 + random.random()*self.levyflight(self.beta)
                        FY, FZ = self.Xs.copy(), self.Xs.copy()
                        FY[i], FZ[i] = Y1, Z1
                        if(self.fitness(self.Xs[i]) > self.fitness(FY[i])):
                            self.X[i] = FY[i]
                        if(self.fitness(self.Xs[i]) > self.fitness(FZ[i])):
                            self.X[i] = FZ[i]

                    elif (self.r < 0.5 and self.E < 0.5):
                        # INSIDE THE DIM LOOP
                        self.Xm = np.sum(np.array(self.Xs),
                                         axis=0)/self.pop_size
                        self.Xm = list(self.Xm.reshape(2,))
                        for j in range(self.dim):
                            Y2 = self.gbest[j] - self.E * \
                                abs(self.J*self.gbest[j] - self.Xm[j])

                        Z2 = Y2 + random.random()*self.levyflight(self.beta)
                        FY, FZ = self.Xs.copy(), self.Xs.copy()
                        FY[i], FZ[i] = Y2, Z2
                        if(self.fitness(self.Xs[i]) > self.fitness(FY[i])):
                            self.X[i] = FY[i]
                        if(self.fitness(self.Xs[i]) > self.fitness(FZ[i])):
                            self.X[i] = FZ[i]

                for j in range(self.dim):
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
            t += 1
            self.y.append(self.Fmin)
            # print(self.Fmin)

        return self.y


# pso = PSO(pop_size=pop_size, dim=dim,
#           iterations=iterations, lower=lower, upper=upper)

# y = pso.run()
