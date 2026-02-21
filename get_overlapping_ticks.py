# this script finds the companies that exist in both the MAEC dataset and the
# fraud data set and prints their tickers to a CSV file

import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime

# load fraud data
fd = pd.read_csv('final_fraud_data.csv')
uni_fraud = list(np.unique(fd['ticker'])) # unique fraud tickers

# load earnings call data
folder = 'MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction/MAEC_Dataset'
command = 'ls %s'%folder
subfolder_names = os.popen(command).readlines()
num_cut = 9
maec_tickers = []
for sub_name in subfolder_names:
    this_sub_name = sub_name.replace("\n","")
    maec_tickers.append(this_sub_name[num_cut:])
uni_maec = np.unique(maec_tickers) # unique maec tickers

# find the tickers common to both the fraud and earnings data sets
common_tickers = pd.DataFrame(set(uni_maec).intersection(uni_fraud))

save_path = 'data/common_tickers.csv'
common_tickers.to_csv(save_path, index=False)
