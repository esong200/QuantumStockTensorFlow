import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import json
import os

print("Done initialization")

print("Current Working Directory:" + os.getcwd())

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
file_name = "dataFortune500Raw2019-2-10.txt"

file_directory = os.path.join(THIS_FOLDER, file_name)
DAYS = 30
DAYSDATA = 30
num_comps = 0
JSONOFDATA = {}
compiledData = []
avg50Day = []
openp = []
close = []
high = []
low = []
volume = []
splitCoef = []
DJIAOpenP = []
DJIAChange = []
vsDow = []
comps_excluded = []

def resultInterpEasy(results):  # if we can't seperate into five, see if we can just do sell, buy, hold
    change = results.copy()
    for i in range(len(change)):
        if change[i][0] < -.02:
            change[i] = [1, 0, 0]
        elif change[i][0] < 0.2:
            change[i] = [0, 1, 0]
        else:
            change[i] = [0, 0, 1]

    change = [change[i].index(1) for i in range(len(change))]

    return change


def resultInterp(results):
    change = results.copy()
    for i in range(len(change)):
        if change[i][0] < -.05:
            change[i] = [1, 0, 0, 0, 0]
        elif change[i][0] < -.02:
            change[i] = [0, 1, 0, 0, 0]
        elif change[i][0] < .02:
            change[i] = [0, 0, 1, 0, 0]
        elif change[i][0] < .05:
            change[i] = [0, 0, 0, 1, 0]
        else:
            change[i] = [0, 0, 0, 0, 1]

    change = [change[i].index(1) for i in range(len(change))]

    return change


def getLargest(item):
    while len(set(item)) > 1:
        item = [item[i] for i in range(len(item)) if item[i] > item[i - 1]]
    return item[0]


'''def resultInterp(results):
    change = results.copy()
    #print(results)
    for i in range(len(change)):
        if change[i][0] <=-.1:
            change[i] = [1,0,0,0,0,0,0,0]
        elif change[i][0] <=-.05:
            change[i] = [0,1,0,0,0,0,0,0]
        elif change[i][0] <=-.02:
            change[i] = [0,0,1,0,0,0,0,0]
        elif change[i][0] <= 0:
            change[i] = [0,0,0,1,0,0,0,0]
        elif change[i][0] <= .02:
            change[i] = [0,0,0,0,1,0,0,0]
        elif change[i][0] <= .05:
            change[i] = [0,0,0,0,0,1,0,0]
        elif change[i][0] <= .1:
            change[i] = [0,0,0,0,0,0,1,0]
        else:
            change[i] = [0,0,0,0,0,0,0,1]
    change = [change[i].index(1) for i in range(len(change))]

    return change
'''

def dumpJSON(trainingType='Fourtune500+50DayAvg'):
    fileName = 'data' + trainingType + str(DAYS) + '.txt'

    with open(fileName, 'w+') as outfile:
        print('dumpingJson')
        json.dump(JSONOFDATA, outfile)
def dumpJSON(name):
    fileName = name

    with open(fileName, 'w+') as outfile:
        print('dumping JSON')
        json.dump(JSONOFDATA, outfile)


def add50DayAvg(data):
    change = data.copy()
    try:
        for i in range(len(change)):
            avg50Day = sum([data[i + days][1] for days in range(1, 51)]) / 50
            change[i].append(avg50Day)
    except:
        pass

    return change


'''
Calcualtion of BBands
'''


def twentyAvg(data):
    change = data.copy()
    avg = []
    try:
        for i in range(len(change)):
            avg50Day = sum([data[i + days] for days in range(1, 21)]) / 20
            avg.append(avg50Day)
    except:
        pass

    return avg


def standDev(data, days=20):
    change = data.copy()
    try:
        for i in range(len(change)):
            mean = 0
            sumed = 0
            std = 0
            for j in range(days):
                sumed += change[i + j]

            mean = sumed / days

            for j in range(days):
                std += (change[i + j] - mean) ** 2

            std /= days
            change[i] = std ** (1 / 2)
    except:
        pass

    return change


def bbandHigh(data):
    copy = [data.copy()[i][0] for i in range(len(data))]
    mid = twentyAvg(copy)
    standardDev = standDev(copy)
    bbands = [mid[i] + 2 * standardDev[i] for i in range(len(mid))]

    return bbands


def bbandLow(data):
    # print(len(data))
    change = [data.copy()[i][0] for i in range(len(data))]
    mid = twentyAvg(change)
    # print('change', change)
    # print(mid)
    standardDev = standDev(change)
    # print('standardDev', standardDev)
    bbands = [mid[i] - 2 * standardDev[i] for i in range(len(mid))]
    # print('bband from calcs', bbands)

    return bbands


'''
Calculation of CCI
'''


def CCI(data):
    TP = typicalPrice(data)
    SMAofTP = twentyAvg(TP)
    meanDeviation = MeanDeviation(TP, SMAofTP)
    cci = []

    for i in range(len(SMAofTP)):
        try:
            cci.append((TP[i] - SMAofTP[i]) / (.015 * meanDeviation[i]))
        except:
            cci.append(0)

    return cci


def typicalPrice(data):  # data is in form [[open, close, high, low]...]

    return [(data[i][1] + data[i][2] + data[i][3]) / 3 for i in range(len(data))]


def MeanDeviation(typPrice, movingAverage):
    meanDeviations = []
    for i in range(len(movingAverage)):
        absSum = 0
        for j in range(20):
            absSum += abs(typPrice[i + j] - movingAverage[i])

        # print(absSum)
        meanDeviations.append(absSum / 20)

    return meanDeviations


'''
Aroon Calculations
'''


def AroonUp(data):
    closePrices = [data[i][1] for i in range(len(data))]
    sinceHigh = SinceHigh(closePrices)

    aroonUp = [(25 - sinceHigh[i]) / 25 for i in range(len(sinceHigh))]

    return aroonUp


def SinceHigh(data):
    sinceLastHigh = []
    for i in range(len(data)):
        try:
            twentyFiveMax = max([data[i + days] for days in range(25)])
            sinceLastHigh.append([data[i + days] for days in range(25)].index(twentyFiveMax))
        except:
            break

    return sinceLastHigh

def AroonDown(data):
    closePrices = [data[i][1] for i in range(len(data))]
    sinceLow = SinceLow(closePrices)

    aroonDown = [(25 - sinceLow[i]) / 25 for i in range(len(sinceLow))]

    return aroonDown


def SinceLow(data):
    sinceLastHigh = []
    for i in range(len(data)):
        try:
            twentyFiveMax = min([data[i + days] for days in range(25)])
            sinceLastHigh.append([data[i + days] for days in range(25)].index(twentyFiveMax))
        except:
            break

    return sinceLastHigh


'''
Exponential Moving Average
'''


def EMA(data, days=DAYS):
    multiplyer = (2 / (days + 1))
    # print(multiplyer)
    firstAverage = sum([data[i][1] / days for i in range(-days, 0)])
    EMA = [firstAverage]

    for i in range(-(days + 1), -(len(data) + 1), -1):
        # print((data[i][1]-EMA[0])*multiplyer+EMA[0])
        EMA.insert(0, (data[i][1] - EMA[0]) * multiplyer + EMA[0])

    return EMA


'''
MACD
'''


def MACD(dataf):
    ema26 = EMA(dataf, days=26)
    ema12 = EMA(dataf, days=12)

    MACDLine = [ema12[i] - ema26[i] for i in range(min(len(ema26), len(ema12)))]

    signalLine = MACDEMA(MACDLine, 9)

    MACD = [MACDLine[i] - signalLine[i] for i in range(min(len(MACDLine), len(signalLine)))]
    # print(max(MACD), min(MACD))
    # print(max(ema26), min(ema26))
    # print(max(ema12), min(ema12))
    # print(max(signalLine), min(signalLine))
    # print('MACDLine', max(MACDLine), min(MACDLine))

    return MACD


def MACDEMA(data, days):
    multiplyer = (2 / (days + 1))

    EMA = [sum([data[i] / days for i in range(-days, 0)])]

    for i in range(-(days + 1), -(len(data) + 1), -1):
        EMA.insert(0, (data[i] - EMA[0]) * multiplyer + EMA[0])

    return EMA


'''
STOCH
'''


def FastSTOCH(data):
    fastKs = []
    for i in range(len(data) - 14):
        recentClose = data[i][1]
        highest = max([data[i + j][2] for j in range(15)])
        lowest = min([data[i + j][3] for j in range(15)])

        # print((recentClose-lowest)/(highest-lowest))
        try:
            fastKs.append((recentClose - lowest) / (highest - lowest))
        except:
            fastKs.append(1)

    ds = [(fastKs[i] + fastKs[i + 1] + fastKs[i + 2]) / 3 for i in range(len(fastKs) - 3)]

    # print(max(fastKs), min(fastKs))
    # print(max(ds), min(ds))
    return fastKs, ds



def percent(data, days=1) -> list:
    adjust = []
    for i in range(len(data) - days):
        try:
            adjust.append(data[i] / data[i + days] - 1)
        except:
            adjust.append(1)
    return adjust


def formatData(openP, close, high, low, volume, splitCoef, avg50Day, vsDow, dowChange,
               bbandLow, bbandHigh, cci, aroonUp, aroonDown, EMA, MACD, fastK, fastD):
    adjOpen = [min(change, 1) for change in percent(openP)]
    adjClose = [min(change, 1) for change in percent(close)]
    adjHigh = [min(change, 1) for change in percent(high)]
    adjLow = [min(change, 1) for change in percent(low)]
    adjVolume = [min(change, 1) for change in percent(volume)]
    adj50Day = [avg50Day[i] / openP[i] - 1 for i in range(len(openP))]
    adjVsDow = [min(change, 1) for change in percent(vsDow)]
    adjDowChange = [min(change, 1) for change in percent(DowChange)]
    adjLow = [min(1, 1 - (bbandLow[i] / openP[i])) for i in range(min(len(bbandLow), len(openP)))]
    # print(1-(bbandLow[0]/openP[0]))
    # print('bband',bbandLow[0])
    # print('price',openP[0])
    # print(min(1,1-(bbandLow[0]/openP[0])))
    adjHigh = [min(1, bbandHigh[i] / openP[i] - 1) for i in range(min(len(bbandLow), len(openP)))]
    adjCCI = [cci[i] / 100 for i in range(len(cci))]
    adjEMA = [EMA[i] / openP[i] - 1 for i in range(min(len(openP), len(EMA)))]
    adjMACD = [MACD[i] / 10 for i in range(len(MACD))]

    data = []

    for i in range(DAYS - 1, len(openP) - DAYS):
        fiveDay = []
        for days in range(DAYS):
            fiveDay.append([adjOpen[i + days], adjClose[i + days], adjHigh[i + days], adjLow[i + days],
                            adjVolume[i + days], splitCoef[i + days], adj50Day[i + days],
                            adjVsDow[i + days], adjDowChange[i + days], adjLow[i + days], adjHigh[i + days],
                            adjCCI[i + days], aroonUp[i + days], aroonDown[i + days], adjEMA[i + days],
                            adjMACD[i + days], fastK[i + days], fastD[i + days]])

        data.append(fiveDay)

    # print(len(data))
    return data[DAYSDATA - 1:]


def getResult(openPrice, days=DAYSDATA - 1):
    change = percent(openPrice, days)[:-days - days - 1]
    result = []
    for item in change:
        result.append([item])
    return result


def addToJSON(data, dataType, ticker):
    if dataType == 'trainingData':
        try:
            JSONOFDATA[ticker]['trainingData'] = data
        except:
            JSONOFDATA[ticker] = {}
            JSONOFDATA[ticker]['trainingData'] = data

    elif dataType == 'trainingResult':
        try:
            JSONOFDATA[ticker]['trainingResult'] = data
        except:
            JSONOFDATA[ticker] = {}
            JSONOFDATA[ticker]['trainingResult'] = data

    elif dataType == 'testData':
        try:
            JSONOFDATA[ticker]['testData'] = data
        except:
            JSONOFDATA[ticker] = {}
            JSONOFDATA[ticker]['testData'] = data

    elif dataType == 'testResult':
        try:
            JSONOFDATA[ticker]['testResult'] = data
        except:
            JSONOFDATA[ticker] = {}
            JSONOFDATA[ticker]['testResult'] = data

with open(file_directory) as json_file:
    print('opening Json ')
    allData = json.load(json_file)

compiledData = []
avg50Day = []
openp = []
close = []
high = []
low = []
volume = []
splitCoef = []
DJIAChange = []
vsDow = []
compiledData = []
avg50Day = []
openp = []
close = []
high = []
low = []
volume = []
splitCoef = []
DJIAOpenP = []
DJIAChange = []
vsDow = []
for ticker, data in allData.items():
    if ticker == 'DJIA':
        print(len(data))
        for i in range(len(data)):
            DJIAOpenP.append(data[i][0])
            try:
                DJIAChange.append(data[i][0] / data[i + 1][0])
            except:
                DJIAChange.append(1)
try:
    for ticker, data in allData.items():
        if ticker != 'DJIA':
            data = data[:len(DJIAOpenP)]
            DowChange = []
            compiledData = []
            avg50Day = []
            openp = []
            close = []
            high = []
            low = []
            volume = []
            splitCoef = []
            vsDow = []
            with50Day = add50DayAvg(data)
            bandLow = bbandLow(data)

            bandHigh = bbandHigh(data)
            # print(ticker)
            cci = CCI(data)
            aroonUp = AroonUp(data)
            aroonDown = AroonDown(data)
            EMAcalc = EMA(data)
            # print(len(EMAcalc))
            MACDcalc = MACD(data)
            # print(max(EMAcalc), min(EMAcalc))
            fastK, fastD = FastSTOCH(data)
            for i in range(len(with50Day)):
                try:
                    avg50Day.append(with50Day[i][6])
                    openp.append(with50Day[i][0])
                    close.append(with50Day[i][1])
                    high.append(with50Day[i][2])
                    low.append(with50Day[i][3])
                    volume.append(with50Day[i][4])
                    splitCoef.append(with50Day[i][5])
                    vsDow.append(data[i][0] / DJIAOpenP[i])
                    DowChange.append(DJIAChange[i])
                except:
                    pass
            result = getResult(openp)
            dataFormated = formatData(openp, close, high, low, volume, splitCoef, avg50Day, vsDow, DowChange,
                                      bandLow, bandHigh, cci, aroonUp, aroonDown,
                                      EMAcalc, MACDcalc, fastK, fastD)
            # print('size of dataformated', len(dataFormated), len(dataFormated[0]), len(dataFormated[0][0]))
            # print(dataFormated)

            testResult = result.pop(0)
            testData = dataFormated.pop(0)
            # print(testResult)
            # print(testData)

            addToJSON(dataType='trainingResult', data=result, ticker=ticker)
            addToJSON(dataType='trainingData', data=dataFormated, ticker=ticker)
            addToJSON(dataType='testData', data=testData, ticker=ticker)
            addToJSON(dataType='testResult', data=testResult, ticker=ticker)
            # end loop
except:
    pass
    comps_excluded.append(ticker)


print('Done with data formatting')
print('Number of companies: ', num_comps)
print('Companies excluded:')
for i in comps_excluded:
    print(i)
if len(comps_excluded) == 0:
    print("No companies excluded")
print ("Test length: ", len(testData))

#dumpJSON(file_name + 'FORMATTED.txt')
#print('Formatted file written')

#Begin Keras

for ticker, data in JSONOFDATA.items():
    print(data['testData'][0])
    break

trainingResults = []
trainingData = []
testResults = []
testData = []
for ticker, data in JSONOFDATA.items():
    trainingResults += data['trainingResult']
    trainingData += data['trainingData']
    testResults.append(data['testResult'])
    testData.append(data['testData'])

trainingResults = resultInterpEasy(trainingResults)
testResults = resultInterpEasy(testResults)

print(trainingResults.count(1), len(trainingResults))

print('transfering to arrays')
trainingResults = np.asanyarray(trainingResults)
trainingData = np.asanyarray(trainingData)
testResults = np.asanyarray(testResults)
testData = np.asanyarray(testData)

print('Starting Keras')
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, len(testData[0][0]))),
    keras.layers.Dense(128, activation=tf.nn.softplus),
    keras.layers.Dense(64, activation=tf.nn.softplus),
    keras.layers.Dense(32, activation=tf.nn.softplus),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainingData, trainingResults, epochs=125)

test_loss, test_acc = model.evaluate(testData, testResults)

print('Test accuracy:', test_acc)
predictions = model.predict(testData)
print('Predictions:')
print(predictions)
print('Training Results:')
print(trainingResults)
print('Test Re  sults:')
print(testResults)

model.save_weights('AAPL30DayWeights.h5')