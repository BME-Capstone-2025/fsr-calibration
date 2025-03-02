import pandas as pd
from calibrate import calibrate_fsr

dat = pd.read_csv('out_data.csv')
x = calibrate_fsr(dat, mode='calibrate')
print(x.coef_)