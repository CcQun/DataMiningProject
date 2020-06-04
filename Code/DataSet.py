import pandas as pd
import numpy as np
import os

dataRoot = "D:\\CodingProject\\PyCharmProject\\DataMiningProject\\Data\\csv"

def dataset():

    datas = []
    labels = []
    for dataCsv in os.listdir(dataRoot):
        df = pd.read_csv(os.path.join(dataRoot, dataCsv))
        datas.append(df.iloc[:, :-1].values)
        labels.append(df.iloc[:, -1].values.reshape((-1, 1)))

    data = np.vstack((datas[0], datas[1]))
    label = np.vstack((labels[0], labels[1]))
    for i in range(2, len(datas)):
        data = np.vstack((data, datas[i]))
        label = np.vstack((label, labels[i]))
    return data,label