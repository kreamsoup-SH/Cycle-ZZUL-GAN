# Load Library

import tensorflow as tf
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from PIL import Image

# Input Pipeline
def makeset(datasets):
    X_list = [] # numpy 배열, (입력 사이즈, 100, 100)

    for i in datasets:
        X = Image.open(i)
        X = X.convert("RGB")
        X = X.resesize((100,100))
        X = np.array(X)
        X_list.append(X)
    X_train, X_test = train_test_split(X_list, test_size=0.33, random_state=42)
    return X_train, X_test

X_output = glob.glob('./celeba/*.jpg')
train_input1, test_input1 = makeset(input1_output)

matrix = np.ones((2,3,4))
print(matrix.shape[0])


