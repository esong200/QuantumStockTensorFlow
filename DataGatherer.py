import pandas_datareader
import datetime
import numpy as np
import json
import time

today = datetime.datetime.today()

startDate = '1700-12-10'
endDate = str(today.year) + '-' + str(today.month) + '-' + str(today.day)


def collectData(company, startDate=startDate, endDate=endDate):
    panelData = pandas_datareader.data.DataReader(company, 'av-daily-adjusted', startDate, endDate,
                                                  access_key='M7NCSUIT1QIC6BRR')

    high = list(panelData['high'])
    openP = list(panelData['open'])
    close = list(panelData['close'])
    low = list(panelData['low'])
    volume = list(panelData['volume'])
    splitCoef = list(panelData['split coefficient'])
    return (openP, close, high, low, volume, splitCoef)


def organizeData(openP, close, high, low, volume, splitCoef):
    data = []
    for i in range(len(openP)):
        data.append([openP[i], close[i], high[i], low[i], volume[i], splitCoef[i]])
    print(data)

    return data


def addToFile(ticker, data, Type='DJIA+AAPLRaw'):
    fileName = 'data' + Type + endDate +  '.txt'
    stored = {}
    try:
        with open(fileName, 'r') as json_file:
            stored = json.load(json_file)
    except:
        pass
    finally:
        stored[ticker] = data

    with open(fileName, 'w+') as outfile:
        json.dump(stored, outfile)

tickers = ['DJIA','AAPL']
tickersALL = [
    'DJIA', 'WMT', 'XOM', 'MCK', 'UNH', 'CVS', 'GM', 'F', 'T', 'GE', 'ABC',
    'VZ', 'CVX', 'COST', 'FNMA', 'KR', 'AMZN', 'HPQ', 'CAH', 'ESRX',
    'JPM', 'BA', 'MSFT', 'BAC', 'WFC', 'HD', 'C', 'PSX', 'IBM', 'VLO',
    'ANTM', 'PG', 'GOOGL', 'CMCSA', 'TGT', 'JNJ', 'MET', 'ADM', 'MPC', 'FMCC',
    'PEP', 'UTX', 'AET', 'LOW', 'UPS', 'AIG', 'PRU', 'INTC', 'HUM', 'DIS', 'CSCO',
    'PFE', 'SYY', 'FDX', 'CAT', 'LMT', 'KO', 'HCA', 'ETE', 'TSN', 'AAL',
    'DAL', 'JCI', 'BBY', 'MRK', 'GS', 'HON', 'ORCL', 'MS', 'CI', 'UAL', 'ALL', 'INTL',
    'CHS', 'AXP', 'GILD', 'GD', 'TJX', 'COP', 'NKE', 'INT', 'MMM', 'MDLZ', 'EXC', 'FOXA',
    'DE', 'T', 'SO', 'TWX', 'AVT', 'M', 'EPD', 'TRV', 'PM', 'RAD', 'TECD', 'MCD',
    'QCOM', 'SHLD', 'COF', 'DUK', 'HAL', 'NOC', 'ARW', 'RTN', 'PAGP', 'ABBV', 'CNC', 'CYH',
    'ARNC', 'IP', 'EMR', 'UNP', 'AMGN', 'USB', 'SPLS', 'DHR', 'WHR', 'AFL', 'AN',
    'PGR', 'DG', 'THC', 'LLY', 'LUV', 'PAG', 'MAN', 'KSS', 'SBUX', 'PCAR', 'CMI',
    'MO', 'XRX', 'KMB', 'HIG', 'KHC', 'LEA', 'FLR', 'ACM', 'FB', 'JBL', 'CTL', 'GIS'


    ,'SO','NEE','TMO','AEP','PCG','NGL','BMY','G','NUE','PNC','MU','CL','FCX','CAG',
    'GPS','BHI','BK','DLTR','WFM','PPG','GPC','IEP','PFGC','OMC','DISH','FE','MON','AES',
    'KMX','NRG','WDC','MAR','ODP','JWN','KMI','ARMK','DVA','MOH','WCG','CBS','V',
    'LNC','ECL','K','CHRW','TXT','L','ITW','SNX','HFC','LAKE','DVN','PBF','YUM',
    'TXN','CDW','WM','MMC','CHK','PH','OXY','JCP','ED','CTSH','VFC','AMP','LB',
    'JEC','PFG','ROST','BBBY','CSX','LVS','LUK','D','X','LLL','EIX','ADP','FDC',
    'BLK','WRK','VOYA','SHW','HLT','RRD','SWK','XEL','MUSA','CBG','DHI','EL','PX',
    'BIIB','STT','UNM','RAI','GPI','HSIC','HRI','NSC','RGA','PEG','BBT','DTE','AIZ',
    'GLP','HUN','BDX','SRE','AZO','NAV','DFS','QVCA','GWW','BAX','SYK','APD',
    'WNR','UHS','OMI','CHTR','AAP','MA','AMAT','EMN','SAH','ALLY','CST','EBAY',
    'LEN','GME','RS','HRL','CELG','GNW','PYPL','PCLN','MGM','ALV','FNF','RSG',
    'GLW','UNVR','MOS','CORE','HDS','CCK','EOG','VRTV','APC','LH','FOXA','STI',
    'CAR','LVLT','TEN','UNFI','DF','CPB','MHK','BWA','PVH','BLL','ORLY','ES',
    'BEN','MAS','LAD','KKR','OKE','NEM','PPL','SPTN','PWR','XPO','RL','IPG',
    'STLD','WCC','DGX','BSX','AGCO','FL','HSY','CNP','WMB','DKS','LYV','WRB',
    'LKQ','AVP','DRI','KND','WY','CASY','SEE','FITB','DOV','HII','NFLX','DDS',
    'EME','JONE','AKS','UGI','EXPE','CRM','TRGP','APA','SPR','EXPD','AXE','FIS',
    'ABG','HES','R','TEX','SYMC','SCHW','CPN','CMS','ADS','JBLU','DISCA','TRN',
    'SANM','NCR','FTI','ERIE','ROK','DPS','IHRT','TSCO','JBHT','CMC','OI','AFG',
    'NTAP','OSK','AEE','AMRK','BKS','DAN','STZ','LPNT','ZBH','HOG','PHM','NWL',
    'AVY','JLL','WEC','MRO','TA','URI','HRG','ORI','WIN','DK','PKG','Q','HBI',
    'RLGY','MAT','MSI','SJM','RF','CE','CLX','INGR','GEN','BTU','ALK','SEB','FTR',
    'APH','WYN','KELYA','WU','VC','AJG','HST','ASH','NSIT','MKL','ESND','OC',
    'SPGI','RJF','NI','ABM','CFG','BAH','SPGI','UFS','COL','LRCX','FISV','SE',
    'NAVI','BIG','TDS','FAF','NVR','CINF','BURL']

startTime = time.time()
numberCompsDone = 0

for comp in tickers:
    openP, close, high, low, volume, splitCoef = collectData(comp)

    dailySplit = organizeData(openP, close, high, low, volume, splitCoef)

    addToFile(comp, dailySplit)
    print('finished collecting', comp)
    print()
    numberCompsDone += 1
    if numberCompsDone % 5 == 0:
        time.sleep(60 - (min(time.time() - startTime, 60)))
        startTime = time.time()