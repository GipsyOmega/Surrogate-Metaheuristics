import numpy as np
import random


def random_k(i):
    k = 4

    while(k == i):
        if k == 4:
            k -= 1
        else:
            k += 1
    return k


k = random_k(4)
print(k)
