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

#14 Day Period Tests

X, y = formatData(dailyOpen, dailyClose, dailyHigh, dailyLow,dailyVolume)
testX, testy = dropOut(X,y)

print("done")

X = preprocessing.scale(np.asarray(X))
X_scale = preprocessing.scale(X)
y = np.asarray(y)
testX = preprocessing.scale(np.asarray(testX))
testy = np.asarray(testy)
clf = DecisionTreeRegressor(max_depth= None, min_samples_split = 2, random_state = 0).fit(X,y)
clfE = ExtraTreeRegressor(max_depth=None, min_samples_split=2, random_state=0).fit(X,y)

scores = cross_val_score(clf, X, y, cv = 5)
scoresE = cross_val_score(clfE, X, y, cv = 5)
print('Training Decision',scores.mean())
print('Training Extra', scoresE.mean())

unseen = cross_val_score(clf, testX, testy, cv = 5)
unseenE = cross_val_score(clfE, testX, testy, cv = 5)
print('New Data Decision', unseen.mean())
print('New Data Extra', unseenE.mean())

defaultPrdict = clf.predict(testX)
#defaultPrdictLog = clf.predict_proba(testX)
extraPrdict = clfE.predict(testX)
#extraPrdictLog = clfE.predict_proba(testX)
defaultTrain = clf.predict(X)
extraTrain = clfE.predict(X)

#print(defaultPrdictLog)

print(clfE.n_outputs_)

print(X.shape)
print(extraPrdict)
print(defaultPrdict)
print(testy)

dCompare = [(list(defaultPrdict[i]).index(1), list(testy[i]).index(1)) for i in range(len(testy))]
eCompare = [(list(extraPrdict[i]).index(1) if 1 in list(extraPrdict[i]) else None, list(testy[i]).index(1)) for i in
            range(len(testy))]

dTracing = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
eTracing = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
for (pre, act) in dCompare:
    try:
        dTracing[act][pre] += 1
    except:
        dTracing[act][pre] = 1

for (pre, act) in eCompare:
    try:
        eTracing[act][pre] += 1
    except:
        eTracing[act][pre] = 1

print('Tracing from actual to predict decision Tree: \n', dTracing)
print('Tracing from actual to predict extra Tree: \n', eTracing)

dCasting = {}
eCasting = {}
for (pre, act) in dCompare:
    dCasting[pre] = {} if pre not in dCasting else dCasting[pre]
    dCasting[pre][act] = dCasting[pre][act] + 1 if act in dCasting[pre] else 1

for (pre, act) in eCompare:
    eCasting[pre] = {} if pre not in eCasting else eCasting[pre]
    eCasting[pre][act] = eCasting[pre][act] + 1 if act in eCasting[pre] else 1

print('Going form predict to actual Decision Tree: \n', dCasting)
print('Going form predict to actual Extra Tree: \n', eCasting)

# Visualizing the decisions First from actual -> predict

print('Visualization of Actual -> Predicted')

for act, preds in dTracing.items():
    # print('When the actual value is {} for Default Trees'.format(act))
    predictions = [key for key in sorted(preds)]
    values = [preds[key] for key in sorted(preds)]
    values = [a / sum(values) for a in values]
    df = DataFrame({str(act): predictions, 'predictions': values})
    ax = df.plot.bar(x=str(act), y='predictions')

for act, preds in eTracing.items():
    # print('When the actual value is {} for Extra Trees'.format(act))
    predictions = [key for key in sorted(preds, key=lambda x: x if type(x) is int else -1)]
    values = [preds[key] for key in sorted(preds, key=lambda x: x if type(x) is int else -1)]
    values = [a / sum(values) for a in values]
    df = DataFrame({str(act): predictions, 'predictions': values})
    ax = df.plot.bar(x=str(act), y='predictions')

print('Visualization of Predicted -> Actual')

for act, preds in sorted(dCasting.items()):
    # print('When the actual value is {} for Default Trees'.format(act))
    predictions = [key for key in sorted(preds)]
    values = [preds[key] for key in sorted(preds)]
    values = [a / sum(values) for a in values]
    df = DataFrame({str(act): list(predictions), 'actual': list(values)})
    ax = df.plot.bar(x=str(act), y='actual')

for act, preds in sorted(eCasting.items(), key=lambda x: x[0] if type(x[0]) is int else -1):
    # print('When the actual value is {} for Default Trees'.format(act))
    predictions = [key for key in sorted(preds)]
    values = [preds[key] for key in sorted(preds)]
    values = [a / sum(values) for a in values]
    df = DataFrame({str(act): list(predictions), 'actual': list(values)})
    ax = df.plot.bar(x=str(act), y='actual')

# From the Training Data
dTCompare = [(list(defaultTrain[i]).index(1), list(y[i]).index(1)) for i in range(len(y))]
eTCompare = [(list(extraTrain[i]).index(1) if 1 in list(extraTrain[i]) else None, list(y[i]).index(1)) for i in
             range(len(y))]

dTTracing = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
eTTracing = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
for (pre, act) in dTCompare:
    try:
        dTTracing[act][pre] += 1
    except:
        dTTracing[act][pre] = 1

for (pre, act) in eTCompare:
    try:
        eTTracing[act][pre] += 1
    except:
        eTTracing[act][pre] = 1

print('Tracing from actual to predict decision Tree: \n', dTTracing)
print('Tracing from actual to predict extra Tree: \n', eTTracing)