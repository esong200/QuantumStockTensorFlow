from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn import preprocessing
import numpy as np
import json
from pandas import DataFrame
import random
import os


def dropOut(data, solution, percent: int = 3):
    random.seed = 123
    testData = []
    testAns = []
    while len(testData) / len(data) <= percent / 100:
        toDropIndex = random.randint(0, len(data) - 1)
        testData.append(data.pop(toDropIndex))
        testAns.append(solution.pop(toDropIndex))
    return testData, testAns


def intoPercent(values: list):
    return [(values[i + 1] - values[i]) / values[i] if values[i] != 0 else values[i + 1] for i in
            range(len(values) - 1)]


def unpack(*items):
    elements = []
    for item in items:
        if type(item) == list:
            elements += unpack(*item)
    return elements + [b for b in items if not type(b) is list]


def formatData(dailyOpen, dailyClose, dailyHigh, dailyLow, dailyVolume, daysofData: int = 40, daysAfter: int = 14):
    dataX = []
    y = []
    for i in range(len(dailyOpen)):
        dataDay = []
        for j in range(len(dailyOpen[i]) - (daysofData + daysAfter)):
            adjustData = [dailyVolume[i][j + k] for k in range(daysofData)]
            adjustData = [a / sum(adjustData) for a in adjustData]
            dataDay = [
                [dailyOpen[i][j + k], dailyClose[i][j + k], dailyHigh[i][j + k], dailyLow[i][j + k], adjustData[k]] for
                k in range(daysofData)]

            dataX.append(unpack(*dataDay))
            y.append(percentChange(*[dailyClose[i][j + k] for k in range(daysAfter)]))
            # print(len(dataX), len(y))
            # print(np.asanyarray(y).shape)
    return dataX, y


def percentChange(*changes):
    change = 1
    for day in changes:
        change *= (1 + day)
    return advice((change - 1) * 100)


def advice(percent: int):
    if percent < -5:
        return [1, 0, 0, 0, 0]
    elif percent < -2:
        return [0, 1, 0, 0, 0]
    elif percent < 2:
        return [0, 0, 1, 0, 0]
    elif percent < 5:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
file_name = "dataFortune500Raw2019-2-10.txt"
file_directory = os.path.join(THIS_FOLDER, file_name)
testComps = 30

dailyOpen = []
dailyClose = []
dailyHigh = []
dailyLow = []
dailyVolume = []
comps = 0

with open(file_directory, 'r') as file:
    data = json.load(file)
    print('Opening Json')

print('Json successfully opened')

for company, priceData in data.items():
    if company != 'DJIA':
        comps += 1
        split = 1
        compOpen = []
        compClose = []
        compHigh = []
        compLow = []
        compVolume = []
        for dataDay in priceData:
            split*= dataDay[-1]
            compOpen.append(dataDay[0]*split)
            compClose.append(dataDay[1]*split)
            compHigh.append(dataDay[2]*split)
            compLow.append(dataDay[3]*split)
            compVolume.append(dataDay[-2])
        dailyOpen.append(intoPercent(compOpen))
        dailyClose.append(intoPercent(compClose))
        dailyHigh.append(intoPercent(compHigh))
        dailyLow.append(intoPercent(compLow))
        dailyVolume.append(compVolume)

print('Done formatting')