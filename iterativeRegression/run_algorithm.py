from __future__ import division
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.linear_model import LinearRegression, RANSACRegressor
from collections import Counter
import copy
from db import DB

import config
import algorithm as al
import db_job
import visualize

import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey


data_warehouse = config.data_warehouse
model_version = config.model_version
window_size = config.window_size
moving_window = config.moving_window
threshold = config.threshold
model_version = config.model_version
base_model = config.base_model
std_threshold = config.std_threshold

try:
    acnt_ids_to_process = db_job.get_account_ids_of_new_weights(db_job.get_newest_weight_id_entire())
except:
    acnt_ids_to_process = []

# if this is the initial run, take the latest users (account_ids) who weighed in
if len(acnt_ids_to_process) == 0:
    acnt_ids_to_process = db_job.first_x_weight_ids(50) # initial run is set to have latest 50 ids

import time
print('{} accounts to process..'.format(len(acnt_ids_to_process)))
mins_list = []
for i, acnt_id in enumerate(acnt_ids_to_process):
    print(i,acnt_id)
    start = time.time()
    model = al.IterativeRegression(window=window_size, threshold=threshold, std_threshold=std_threshold, model=base_model, model_version=model_version)
    num_processed = db_job.count_num_processed_only_filtered2_False(acnt_id)
    featureDB_data = db_job.get_latest_rows_and_filtered2_false(acnt_id)
    if num_processed <= window_size:
        kairos_data = db_job.get_all_from_kairos_per_acnt(acnt_id)
        df = model.fit_predict1(kairos_data, num_processed, moving_window = moving_window)
        db_job.write_to_featureDB(df)
        # print('model.temp_model', model.temp_model)
        # print("{}th 1st case/{} rows appended".format(i, len(df)))
    elif num_processed > window_size:
        kairos_data2 = db_job.get_only_new_from_kairos_per_acnt(acnt_id)
        df = model.fit_predict2(kairos_data2, featureDB_data, num_processed, moving_window = moving_window)
        db_job.write_to_featureDB(df)
        # print('model.temp_model', model.temp_model)
        print("{}th 2nd case/{} rows appended".format(i, len(df)))
    end = time.time()
    minutes = (end-start)/60
    mins_list.append(minutes)
    print("{} mins| on average {} mins".format(minutes, np.mean(mins_list)))
print "finished"
