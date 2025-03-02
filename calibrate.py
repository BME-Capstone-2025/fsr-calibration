import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as lin
from sklearn.model_selection import KFold
import random

def calibrate_fsr(dat, mode='calibrate'):
    integ = []
    locav = []
    last = 0

    nz = False
    nz_ids = []

    for x in range(dat.shape[0]):
        if dat['FSR'][x] < 50:
            integ.append(0)
            locav.append(0)
            if nz:
                nz = False
                nz_ids.append((last, x))
            last = x

        else:
            integ.append(np.trapz(dat['FSR'][:x+1], dat['T'][:x+1]) / 1e5)
            locav.append(x - last)
            if not nz:
                nz = True

    dat['FSR_I'] = integ
    dat['FSR_LA'] = locav

    dat['FSR2'] = dat['FSR'] ** 2
    dat['FSR_I2'] = dat['FSR_I'] ** 2
    dat['FSR_LA2'] = dat['FSR_LA'] ** 2
    
    X = dat[['FSR', 'FSR_I', 'FSR_LA', 'FSR2', 'FSR_I2', 'FSR_LA2']]
    y = dat['LC']

    if mode == 'calibrate':
        lr = lin.LinearRegression(fit_intercept=False)
        lr.fit(X, y)
        return lr
    
    elif mode == 'train-test':
        kf = KFold(n_splits=5, random_state=808)
        models = []
        for i, (train_index, test_index) in enumerate(kf.split(nz_ids)):
            X_tr = pd.concat([X.loc[nz_ids[x][0]:nz_ids[x][1]] for x in train_index])
            X_ts = pd.concat([X.loc[nz_ids[x][0]:nz_ids[x][1]] for x in test_index])

            y_tr = pd.concat([y.loc[nz_ids[x][0]:nz_ids[x][1]] for x in train_index])
            y_ts = pd.concat([y.loc[nz_ids[x][0]:nz_ids[x][1]] for x in test_index])

            lr = lin.LinearRegression(fit_intercept=False)
            lr.fit(X_tr, y_tr)
            print(f'Split: {i}')
            print(f'Train: {train_index}')
            print(f'Test: {test_index}')
            print(f'Train accuracy: {lr.score(X_tr, y_tr)}')
            print(f'Test accuracy: {lr.score(X_ts, y_ts)}')
            models.append(lr)
        
        return models
    