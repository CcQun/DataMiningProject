import numpy as np
import pandas as pd
import os

dataRoot = "D:\\CodingProject\\PyCharmProject\\DataMiningProject\\Data\\txt"


def getDataFromTxt(path):
    f = open(path, "r")
    line = f.readline()
    line = line[:-1]
    while line != 'Time (seconds) and Data Channels':
        line = f.readline()
        line = line[:-1]
    data = np.array([])
    lastSpeed = 5.00
    totalCycle = 0
    linenum = 0
    while line:
        linenum += 1;
        line = f.readline()
        line = line[:-1]
        numbers = line.split()
        if len(numbers) == 0:
            break
        if float(numbers[1]) - lastSpeed > 1.0:
            totalCycle += 1
            appendData = []
            for i in range(408):
                line = f.readline()
                line = line[:-1]
                tsv = line.split()
                if (len(tsv) == 0):
                    break
                appendData = np.append(appendData, float(tsv[2]))
                if i == 407:
                    lastSpeed = float(tsv[1])
            if totalCycle == 1:
                data = np.append(data, appendData)
            elif appendData.shape[0] == 408:
                data = np.vstack((data, appendData))
            else:
                break
        else:
            lastSpeed = float(numbers[1])
    return data


label = 0
for dataDir in os.listdir(dataRoot):
    data = []
    dataDir = os.path.join(dataRoot, dataDir)
    for dataTxt in os.listdir(dataDir):
        data.append(getDataFromTxt(os.path.join(dataDir, dataTxt)))
    X = data[0]
    for i in range(1, len(data)):
        X = np.vstack((X, data[i]))
    Y = np.full(X.shape[0], int(label))
    label += 1
    print(X.shape, Y.shape)
    df = pd.DataFrame(np.hstack((X, Y.reshape((-1, 1)))))
    output_csv_path = os.path.join(dataRoot.replace('txt', 'csv'), os.path.basename(dataDir) + '.csv')
    df.to_csv(dataDir.replace('txt', 'csv'), ',', index=False, header=False)
