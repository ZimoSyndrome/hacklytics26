## this script creates the binary output matrix of shape (n_samples X n_time)
##  where ntime is no. of quarters to look into the future from the date of the earnings call
##        n_samples is the number of earnings calls

import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt

try:
    os.mkdir('data')
except FileExistsError:
    pass

# loop through all earnings calls
#   find the company name and date corresponding to that earnings call
#       that date is the t=0 for this potential incident
#       if that company name corresponds to an incident in the fraud data
#           encode the occurrence of that incident as a one-hot vector where the location corresponds to the quarter it occurred in

folder = 'MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction/MAEC_Dataset'
command = 'ls %s'%folder
all_earnings_calls = os.popen(command).readlines()
num_samples = len(all_earnings_calls); # print(f'num_samples = {num_samples}')
fraud_data = pd.read_csv('final_fraud_data.csv')
fraud_tickers = list(np.unique(fraud_data['ticker']))

num_time = 16 # number of quarters we look in the future from the earnings call date
y = np.zeros((num_samples,num_time))
print(f'y.shape = {y.shape}')

all_times_between_call_and_incident = []
for si, earnings_call_pre in enumerate(all_earnings_calls):
    earnings_call_id = earnings_call_pre.replace("\n","")
    date = earnings_call_id[:8] # yyyymmdd
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    earnings_call_date = datetime(year=year,month=month,day=day)
    earn_tik = earnings_call_id[9:]
    t0 = earnings_call_date # t=0 for this potential incident
    if earn_tik in fraud_tickers:
        all_fraud_cases_this_call = fraud_data[fraud_data['ticker'] == earn_tik] # all fraudelent events corresponding to this earnings call
        fraud_dates_this_call = list(all_fraud_cases_this_call['date']) # all fraud dates corresponding to this call
        all_quarters_from_call_to_fraud = [] # list to store the number of quarters from call to fraud, for this fraud incident
        for fraud_date_str in fraud_dates_this_call:
            year = int(fraud_date_str[:4])
            month = int(fraud_date_str[5:7])
            day = int(fraud_date_str[8:10])
            tik_date_dt = datetime(year=year,month=month,day=day) # convert incident to date time object for comparison with call date
            time_since_earnings_call = (tik_date_dt - t0) # time delta object
            if time_since_earnings_call.days > 0: # the fraud should happen after the call
                # days_between_call_and_incident.append(time_since_earnings_call.days)
                quarters_from_call_to_incident = 4*(time_since_earnings_call.days / 365.25)
                all_quarters_from_call_to_fraud.append(quarters_from_call_to_incident)
                all_times_between_call_and_incident.append(quarters_from_call_to_incident) # aggregate list across all samples for plotting
            if len(all_quarters_from_call_to_fraud) > 0:
                for q_to_save in all_quarters_from_call_to_fraud:
                    if int(q_to_save) < num_time:
                        y[si,int(q_to_save)] = 1


plt.hist(all_times_between_call_and_incident,density=True,bins=15)
plt.xlabel('time between earnings call and incident (quarters)')
plt.ylabel('prob density')
plt.savefig('times.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

save_path = 'data/y.npy'
np.save(save_path,y)
print('saved y to %s'%save_path)



