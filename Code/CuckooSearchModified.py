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

n = 5  # no of particles
pa = 0.8  # probability of discovering of alien egg
max_iterations = 81
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


# 12:22


def initialize():
    global X
    X = np.stack((dp1, dp2, dp3, dp4, dp5))
    return X


def cuckooSearch():
    global X, pa, l
    X = initialize()
    l = []
    z = fitness(X)
    for t in range(0, max_iterations):  # main loop
        '''
        # declare constants
        u = random.normal(n) * sigma
        v = random.normal(n)
        s = u / (abs(v) ** (1 / b))
        '''
        sigma = (math.gamma(1-beta)*math.sin(3.14*beta/2)) / \
            math.gamma((1+beta)/2)*beta*(2**(beta-1)/2)
        sigma = sigma**1/beta
        func = 0.01*random.random()*sigma
        v = random.random()
        s = 0.01*func/abs((v**1/beta))

        # calculate fitness value for each particle

        #print("Value of Fitness = " + str(z))
        # calculate best position
        best = X[z.argmin()]

        # position update 1
        Xnew = X.copy()
        #print("Xnew copied")
        # print(Xnew)
        for i in range(n):
            Xnew[i, :] += np.random.randn(d) * 0.1 * s * (Xnew[i, :] - best)
        checkBounds(Xnew)
        # print(Xnew)
        fnew = fitness(Xnew)
        X[fnew < z, :] = Xnew[fnew < z, :]
        #  check bounds
        checkBounds(X)

        #  position update 2
        Xnew = X.copy()
        Xold = X.copy()
        for i in range(n):
            d1, d2 = np.random.randint(0, 3, 2)
            for j in range(d):
                r = np.random.rand()
                if r < pa:
                    Xnew[i, j] += np.random.rand() * (Xold[d1, j] - Xold[d2, j])
        checkBounds(Xnew)
        fnew = fitness(Xnew)
        X[fnew < z, :] = Xnew[fnew < z, :]
        #print("UPDATE 2 COMPLETE")

        #  check bounds
        checkBounds(X)

        # print("Best solution after iteration  " +
        #       str(t+1) + " = " + str(z.min()))
        l.append(z.min())
        # df = pd.DataFrame(l)
        # df.to_csv(r"results1.csv")
        # print(best)
        z[fnew < z] = fnew[fnew < z]

    return l


# fitness function
def fitness(X):
    # x = X[:, 0]
    # y = X[:, 1]
    # z = X[:, 2]
    # r = (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73) + (z - 1) ** 3
    # r = (1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2
    # return r
    # df = pd.DataFrame(X.round(4))
    # print(df)
    # df.to_csv(r"Parameters.csv", index=False,
    #           header=False)
    #
    # obj = Workbench.read_results()
    obj = surrogate.obj_original(X)
    # l = []
    # for i in range(n):
    #     a = input("Enter Cd/Cl: \n")
    #     l.append(a)
    # obj = np.array(l).astype(float)
    # print(type(obj))

    return obj


def checkBounds(X):
    for i in range(d):
        xmin, xmax = bound[i]
        X[:, i] = np.clip(X[:, i], xmin, xmax)


# cuckooSearch()

# x = np.linspace(0, 200, 100)  # X-axis points
# y = l  # Y-axis points


# plt.plot(x, y)  # Plot the chart
# plt.show()  # display
