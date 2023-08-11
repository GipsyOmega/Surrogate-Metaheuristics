import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from smt.surrogate_models import RBF
import pandas as pd

df = pd.read_csv(r"surrogateTest - 1000mm.csv")
height = [i for i in df["height_n"]]
angle = [i for i in df["angle_n"]]
obj = [i for i in df["obj"]]
obj = np.array(obj)


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


x = merge(height, angle)
x = np.array(x)
sm = RBF(d0=5, print_global=False)
sm.set_training_values(x, obj)
sm.train()


def obj(X):
    # obj = []
    # for i in range(5):
    #     l = []
    #     l.append(list(X[i]))
    #     pred = sm.predict_values(np.array(l))
    #     print(np.array(l).shape)
    #     obj.append(pred[0][0])
    # obj = np.array(obj)

    pred = sm.predict_values(np.array(X))
    obj = pred[0][0]
    return obj


def obj_original(X):
    obj = []
    for i in range(5):
        l = []
        l.append(list(X[i]))
        pred = sm.predict_values(np.array(l))
        # print(np.array(l).shape)
        obj.append(pred[0][0])
    obj = np.array(obj)

    # pred = sm.predict_values(np.array(X))
    # obj = pred[0][0]
    return obj


# num = 100
# x = np.linspace(0.0, 4.0, num)
# y = sm.predict_values(x)
#
# print(sm.predict_values(np.array([0.5])))
# plt.plot(xt, yt, "o")
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(["Training data", "Prediction"])
# plt.show()
# sampler = qmc.LatinHypercube(d=2)
# sample = sampler.random(n=5)
# l_bounds = [-15, -60]
# u_bounds = [20, 30]
# a = qmc.scale(sample, l_bounds, u_bounds)
# xt = np.array([[0.0,0.0], [1.0,0.5], [2.0,1.0], [3.0,1.5], [4.0,2]])
# xt = np.array([0.0,1.0,2.0,3.0,4.0,0.0])
# yt = np.array([0.0, 1.0, 0.3, 0.5, 1.0,0.0])
