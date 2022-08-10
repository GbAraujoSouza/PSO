import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_particulas = 10
d = {"p{}".format(x): [] for x in range(num_particulas)}

iter_1 = [1,2,3,4,5]
iter_2 = [0,0,0,0,0]

for i in d:
    d[i].append(iter_1)
    d[i].append(iter_2)

df = pd.DataFrame(d['p1'], columns=['coord_{}'.format(i) for i in range(5)])
print(df)
