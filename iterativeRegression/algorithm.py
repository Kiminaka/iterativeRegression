from __future__ import division
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.linear_model import LinearRegression, RANSACRegressor
from collections import Counter
import copy

import config
import db_job
import visualize
import warnings
warnings.filterwarnings('ignore')

model_version = config.model_version
window_size = config.window_size
moving_window = config.moving_window
threshold = config.threshold
model_version = config.model_version
base_model = config.base_model
std_threshold = config.std_threshold

class IterativeRegression():
    """
    Iterative regression combined with threshold method and conditioanl logics
    """
    def __init__(self, window=window_size, threshold=threshold, std_threshold=std_threshold, model=base_model, model_version=model_version):
        self.window = window
        self.threshold = threshold
        self.std_threshold = std_threshold
        self.model = model
        self.model_version = model_version
        self.temp_model = None
        self.start_date = None
        self.current_date = None
        self.account_id = None

    def create_training_set(self, data):
        """
        Use threshold method and conditional logics to prepare training data to feed into regression model
        """
        count = 0
        total_rows = 0
        new_data = pd.DataFrame(columns=[data.columns])
        i_checker = 0
        reported_weight = db_job.get_reported_weight(data.account_id[0])
        init_weight = False
        # consider various cases to create the initial weight data
        try:
            try:
                first_3_confirmed_median = np.median(data[data.confirmed == True][data.value > 100][-3:].value)
                if str(first_3_confirmed_median) == 'nan':
                    first_3_confirmed_median = 10000000
            except:
                first_3_confirmed_median = 10000000
            try:
                first_conirmed_weight = data[data.value > 100][data.confirmed == True].value[0]
                if str(first_conirmed_weight) == 'nan':
                    first_conirmed_weight = 10000000
            except:
                first_conirmed_weight = 10000000
            try:
                first_weight = data[data.value > 100].value[0]
                if str(first_weight) == 'nan':
                    first_weight = 10000000
            except:
                first_weight = 10000000
            if reported_weight != 0:
                comp_r3 = abs(first_3_confirmed_median - reported_weight)
                comp_r1 = abs(first_conirmed_weight - reported_weight)
                comp_rf = abs(first_weight - reported_weight)
                comp_list = [comp_r3,comp_r1, comp_rf]
                closest_val = np.min(comp_list)
                closest_val_index = np.argmin(comp_list)
                if closest_val < 20:
                    if closest_val_index == 0:
                        init_weight = first_3_confirmed_median
                    elif closest_val_index == 1:
                        init_weight = first_conirmed_weight
                    elif closest_val_index == 2:
                        init_weight = first_weight
                else:
                    init_weight = None

                    # print('using reported weight!')
            else:
                try:
                    try:
                        init_weight = np.median(data[data.confirmed == True][data.value > 100][-3:].value)
                    except:
                        init_weight = data[data.confirmed == True][data.value > 100].value[0]
                except:
                    try:
                        init_weight = data[data.value > 100].value[0]
                    except:
                        init_weight = None
            # init_weight = np.median(data[data.confirmed == True][data.value > 100][-3:].value)
        except:
            # init_weight = np.median(data[data.confirmed == True][:3].value)
            init_weight = None
        if str(init_weight) == 'nan':
            init_weight = None
    # -----------------------------------------base case -------------------------------------------------------------
        # compare weight(t) with weight(t-1)
        for i in range(len(data)):
            ii = i + i_checker #keep track of True case
            # create df with window size
            total_rows +=1 #for checking
            if count < self.window:
                new_data.loc[i] = data.iloc[i] #appending a row from original df

            # condition for the first weighin
            if i == 0:
                if count > self.window:
                    break
                elif init_weight == None:
                    new_data['filtered2'][i] = True
                    i_checker += -1
                elif init_weight < 100:
                    new_data['filtered2'][i] = True
                    i_checker += -1
                elif abs(data['value'][i] - init_weight) > self.threshold:
                    new_data['filtered2'][i] = True
                    i_checker += -1
                    continue
                elif abs(data['value'][i] - init_weight) <= self.threshold:
                    new_data['filtered2'][i] = False
                    i_checker = 0
                    ii = i
                    count +=1
                    continue

            # condition for data after the first weighin
            if i > 0:
                try:
                    if count > self.window-1:
                        break
                    if init_weight < 100:
                        new_data['filtered2'][i] = True
                        i_checker += -1
                    elif data['filtered2'][ii-1] == False:
                        if abs(data['value'][i] - data['value'][ii-1]) > self.threshold:
                            new_data['filtered2'][i] = True
                            i_checker += -1
                            continue
                        elif abs(data['value'][i] - data['value'][ii-1]) <= self.threshold:
                            new_data['filtered2'][i] = False
                            i_checker = 0
                            ii = i
                            count +=1
                            continue
                    elif data['filtered2'][ii-1] == True:
                        if abs(data['value'][i] - init_weight) > self.threshold:
                            new_data['filtered2'][i] = True
                            i_checker += -1
                            continue
                        elif abs(data['value'][i] - init_weight) <= self.threshold:
                            new_data['filtered2'][i] = False
                            i_checker = 0
                            ii = i
                            count +=1
                            continue
                except KeyError: # if the first weighin is True, code breaks. To avoid failing, make exception
                    try:
                        if abs(data['value'][i] - init_weight) > self.threshold:
                            new_data['filtered2'][i] = True
                            i_checker += -1
                            continue
                        elif abs(data['value'][i] - init_weight) <= self.threshold:
                            new_data['filtered2'][i] = False
                            i_checker = 0
                            ii = i
                            count +=1
                            continue
                    except TypeError:
                        new_data['filtered2'][i] = True
                        i_checker += -1
                        continue

    # --------------------------------case where 1 or less inlier found OR where first 5 rows are all outliers------------------------------
        if (len(new_data) > self.window-1 and Counter(new_data[:20].filtered2)[False] < 5) or \
        (len(new_data) > self.window-1 and Counter(new_data.filtered2[:5])[True] > 4):
            count = 0
            total_rows = 0
            new_data = pd.DataFrame(columns=[data.columns])
            i_checker = 0
            new_data = pd.DataFrame(columns=[data.columns])

            # print 'Case applied!!!',i,ii
            for i in range(len(data)):
                ii = i + i_checker #keep track of True case
                total_rows +=1
                if count < self.window:
                    new_data.loc[i] = data.iloc[i] #appending a row from original df

                if i == 0:
                    median = np.median(data[data.value > 100].value[:self.window]) # use median as initial comparison instead of confirmed initial weight
                    if str(median) == 'nan':
                        median = True

                    if count > self.window:
                        break
                    elif abs(data['value'][i] - median) > self.threshold:
                        new_data['filtered2'][i] = True
                        i_checker += -1
                        continue
                    elif abs(data['value'][i] - median) <= self.threshold:
                        new_data['filtered2'][i] = False
                        i_checker = 0
                        ii = i
                        count +=1
                        continue
                if i > 0:
                    try:
                        if count > self.window-1:
                            break
                        elif data['filtered2'][ii-1] == False:
                            if abs(data['value'][i] - data['value'][ii-1]) > self.threshold:
                                new_data['filtered2'][i] = True
                                i_checker += -1
                                continue
                            elif abs(data['value'][i] - data['value'][ii-1]) <= self.threshold:
                                new_data['filtered2'][i] = False
                                i_checker = 0
                                ii = i
                                count +=1
                                continue
                        elif data['filtered2'][ii-1] == True:
                            if abs(data['value'][i] - median) > self.threshold:
                                new_data['filtered2'][i] = True
                                i_checker += -1
                                continue
                            elif abs(data['value'][i] - median) <= self.threshold:
                                new_data['filtered2'][i] = False
                                i_checker = 0
                                ii = i
                                count +=1
                                continue
                    except KeyError:
                        if abs(data['value'][i] - median) > self.threshold:
                            new_data['filtered2'][i] = True
                            i_checker += -1
                            continue
                        elif abs(data['value'][i] - median) <= self.threshold:
                            new_data['filtered2'][i] = False
                            i_checker = 0
                            ii = i
                            count +=1
                            continue

    # ----------------------------------------- check detrended std -------------------------------------------------------------
        median = np.median(new_data.value)
        # using linear least-squares to detrend
        detrended_new_data = pd.DataFrame(scipy.signal.detrend(new_data[new_data['filtered2'] == False].value), columns=['value'])
        std = np.std(detrended_new_data.value)
    # ----------------------------------------- Case where std > 3 -------------------------------------------------------------
        if std > 3:
            # print 'More than 3 STD!'
            count = 0
            total_rows = 0
            i_checker = 0

            for i,x in enumerate(new_data.value):
                ii = i + i_checker #keep track of True case
                total_rows +=1
                if i == 0:
                    median = np.median(new_data.value[:5])
                    if count > self.window-1:
                        break
                    elif abs(x - median) > self.std_threshold:
                        i_checker += -1
                        continue
                    elif abs(x - median) <= self.std_threshold:
                        i_checker = 0
                        ii = i
                        count +=1
                        new_data['filtered2'][i] = False
                        continue
                if i > 0:
                    try:
                        if count > self.window-1:
                            break
                        elif new_data['filtered2'][ii-1] == False:
                            if abs(new_data['value'][i] - new_data['value'][ii-1]) > self.std_threshold:
                                i_checker += -1
                                continue
                            elif abs(new_data['value'][i] - new_data['value'][ii-1]) <= self.std_threshold:
                                new_data['filtered2'][i] = False
                                i_checker = 0
                                ii = i
                                count +=1
                                continue
                        elif new_data['filtered2'][ii-1] == True:
                            median = np.median(new_data.value[:5])
                            if abs(new_data['value'][i] - median) > self.std_threshold:
                                i_checker += -1
                                continue
                            elif abs(new_data['value'][i] - median) <= self.std_threshold:
                                new_data['filtered2'][i] = False
                                i_checker = 0
                                ii = i
                                count +=1
                                continue
                    except KeyError:
                        median = np.median(new_data.value[:5])
                        if abs(new_data['value'][i] - median) > self.std_threshold:
                            i_checker += -1
                            continue
                        elif abs(new_data['value'][i] - median) <= self.std_threshold:
                            i_checker = 0
                            ii = i
                            count +=1
                            continue


            detrended_new_data = pd.DataFrame(scipy.signal.detrend(new_data[new_data['filtered2'] == False].value), columns=['value'])
            std = np.std(detrended_new_data.value)
        new_data['processed'] = True
        new_data = new_data.reset_index().drop('index', axis=1)
        return new_data

    # run RANSAC #########################################################################################################################

    def create_data_format1(self, kairos_data, num_processed):
        self.account_id = kairos_data.account_id[0]
        self.start_date = kairos_data.weighed_at[0]
        # if num_processed == 0:
        col = [u'id', u'account_id', u'weighed_at', u'value', u'confirmed', u'manual',u'filtered']
        data = kairos_data[col]
        data = data.rename(columns = {'id':'weight_id'})
        data['filtered2'] = False
        data['processed'] = False

        # else:
        #     col = [u'id', u'account_id', u'weighed_at', u'value', u'confirmed', u'manual',u'filtered' ,u'filtered2', u'value_prediction_w16', 'feature_col', u'processed']
        #     old_data = kairos_data[:num_processed]
        #     new_data = kairos_data[num_processed:]
        #     new_data['processed'] = False
        #
        #     data = pd.concat([old_data, new_data])
        #     data['filtered2'] = False
        #     data = data[col]
        #     data = data.rename(columns = {'id':'weight_id'})


        data['model_version'] = self.model_version
        data['feature_col'] = 0
        data['value_prediction_w16'] = None
        for i, date in enumerate(data.weighed_at):
            if i == 0:
                data['feature_col'][i] = 0
            else:
                data['feature_col'][i] = data.feature_col[i-1] + self.create_x(data.weighed_at.values[i-1], data.weighed_at.values[i])

        data = data.reset_index().drop('index', axis=1)
        return data


    def create_data_format2(self, kairos_data, featureDB_data, num_processed):
        self.account_id = featureDB_data.account_id.values[0]
        self.start_date = featureDB_data.weighed_at[0]
        col = [u'id', u'account_id', u'weighed_at', u'value', u'confirmed', u'manual',u'filtered']
        new_data = kairos_data[col]
        new_data = new_data.rename(columns = {'id':'weight_id'})
        new_data['filtered2'] = False
        new_data['processed'] = False
        new_data['feature_col'] = 0
        new_data['value_prediction_w16'] = None
        new_data['model_version'] = self.model_version

        for i, date in enumerate(new_data.weighed_at):
            if i == 0:
                new_data['feature_col'][i] = featureDB_data.feature_col.values[-1] + self.create_x(featureDB_data.weighed_at.values[-1], new_data.weighed_at.values[i])
            else:
                new_data['feature_col'][i] = new_data.feature_col[i-1] + self.create_x(new_data.weighed_at.values[i-1], new_data.weighed_at.values[i])

        data = pd.concat([featureDB_data, new_data])
        data = data.reset_index().drop('index', axis=1)
        return data.drop('id', axis=1)

    def fit_predict1(self, kairos_data, num_processed, moving_window=10):
        """
        Prepare training sets. And fit and predict for each point afterwards.
        Used with a senario where there is no existing data with corresponding  account id in database
        """
        data = self.create_data_format1(kairos_data, num_processed)
        # Use the conditioned threshold method defined as init_df to feed linear regression.
        data_init_original = self.create_training_set(data)
        if len(data_init_original[data_init_original.filtered2 == False]) < self.window:
            return data_init_original

        init_last_index = data_init_original[data_init_original.filtered2 == False].index[-1]
        #--------------------------------After first data input--------------------------------
        for i in range(init_last_index+1,len(data.value)):
            # Update the dataset with new labeled data and remove outlier
            data_init = data_init_original[data_init_original.filtered2 == False][data_init_original.value > 100]
            detrended_std = np.std(scipy.signal.detrend(data_init.value))
            # Use latest 10 points so that the model can detect non-linearity trend
            X = np.array(data_init.feature_col[-moving_window:])
            X = X.reshape((len(X),1))
            y = data_init.value[-moving_window:]
            y = y.reshape((len(y),1))

            # Create model
            model = self.model
            try:
                model.fit(X, y)
            except:
                model = LinearRegression()
                model.fit(X, y)
                # print 'Using Linear Regression!'
            self.temp_model = model
            pred_y = model.predict(data.feature_col[i]) #next point
            current_1 = data_init.weighed_at.values[-2]
            current = data.weighed_at.values[i]
            new_data = data.ix[i]
            self.current_date = new_data.weighed_at
            diff = self.calc_diff_days(current_1, current)
            label = self.filtered_checker(data['value'][i], pred_y, detrended_std, diff) # check in/outlier
            new_data['filtered2'] = label
            new_data['processed'] = True
            new_data['value_prediction_w16'] = self.predict_week16()
            data_init_original = data_init_original.append(new_data)
            data_init_original = data_init_original.reset_index().drop('index', axis=1)
        return data_init_original

    def fit_predict2(self, kairos_data, featureDB_data, num_processed, moving_window=20):
        """
        Fit and predict for each point.
        Used with a senario where there are existing data with corresponding account id in database
        """
        data = self.create_data_format2(kairos_data, featureDB_data, num_processed).reset_index()
        for i in data[data.processed == False].index:
            # Update the dataset with new labeled data and remove outlier

            data_init = data[data.processed == True][data.value > 100][data.filtered2 == False]
            detrended_std = np.std(scipy.signal.detrend(data_init.value))
            # Use latest 10 points so that the model can detect non-linearity trend
            X = np.array(data_init.feature_col[-moving_window:])
            X = X.reshape((len(X),1))
            y = data_init.value[-moving_window:]
            y = y.reshape((len(y),1))

            # Create model
            model = self.model
            try:
                model.fit(X, y)
            except:
                model = LinearRegression()
                model.fit(X, y)
                # print 'Using Linear Regression!'
            pred_y = model.predict(data.feature_col[i]) #next point

            self.temp_model = model
            current_1 = data_init.weighed_at.values[-2]
            current = data.weighed_at.values[i]
            self.current_date = data.weighed_at[i]
            diff = self.calc_diff_days(current_1, current)
            label = self.filtered_checker(data['value'][i], pred_y, detrended_std, diff) # check in/outlier


            data['filtered2'][i] = label
            data['processed'][i] = True

            data['value_prediction_w16'][i] = self.predict_week16()

            try:
                del data['index']
            except:
                data
        return data[len(featureDB_data):]

    def calc_diff_days(self, x_1, x):
        x = x.astype('datetime64[D]')
        x_1 = x_1.astype('datetime64[D]')
        diff = abs(x - x_1)
        days = diff / np.timedelta64(1, 'D')
        return days

    def create_x(self, x_1, x):
        # calculate differene in days between x and x-1 rows
        days = self.calc_diff_days(x_1, x)
        if days == 0:
            return 0
        elif days < 0:
            return 0
        else:
            return days

    def filtered_checker(self, obs_y, pred_y, std, diff):
        # label inlier or outlier
        if std > 3:
            if abs(obs_y - pred_y) > 14 + (diff * .2):
                return True
            else:
                return False
        else:
            if abs(obs_y - pred_y) > self.threshold + (diff * .2):
                return True
            else:
                return False

    def predict_week16(self):
        # predict weight at week 16
        diff = abs(self.current_date - self.start_date)
        date_til_w16 = 112 - diff.days # 112 = 16 wks

        if date_til_w16 >= 0:
            pred_y_w16 = self.temp_model.predict(112)
            if pred_y_w16 > 0:
                return pred_y_w16[0][0]
            else:
                return None
        else:
            # return 0
            pred_y_w16 = self.temp_model.predict(diff.days+30)
            if pred_y_w16 > 0:
                return pred_y_w16[0][0]
            else:
                return None
