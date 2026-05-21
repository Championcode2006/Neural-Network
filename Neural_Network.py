import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) #To make sure there is no pattern in the dataset like all 0's lable together

print(m, n)