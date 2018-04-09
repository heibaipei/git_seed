import scipy.io as sio
import os
import numpy as np
import pandas as pd
start=1
end=2
dataset_dir="seed_data/"
for j in range(start,end):	
    data_dir= dataset_dir+'s'+format(j, '02d')
    record_list = [task for task in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,task))]
    print(record_list)
    for record in record_list:
        trials = sio.loadmat(data_dir+"/"+record)
        newtrials = trials.copy()
        for trial in trials.keys():
            if "__" in trial:
                del newtrials[trial]
            for i in range(1,16):
                flag = "eeg"+str(i)
                if flag in trial:
                    newtrials[str(i+100)] = newtrials[trial]
                   # del newtrials[trial]
        for trial in trials.keys():
            if "eeg" in trial:
                del newtrials[trial]
        print(record, "--", sorted(newtrials.keys()))
        sio.savemat(data_dir+"/"+record,newtrials)