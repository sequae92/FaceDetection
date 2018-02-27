import os
import numpy as np

if __name__ == "__main__":

    dict = {}
    mean = np.zeros((2,2))
    cov = np.zeros((2,2))

    clusters = 2

    for i in range(clusters):
        dict[i] = [mean,cov]
        print("===========================================")
        print(dict[i][0])
        print("================Demarcator=================")
        print(dict[i][1])