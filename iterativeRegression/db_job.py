from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
from db import DB
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

import config

data_warehouse = config.data_warehouse
kairos = config.kairos

def get_newest_weight_id_entire():
    database = DB(profile='data_warehouse')
    q = database.query('select max(weight_id) from kim.weights_outlier')
    return q.values[0][0]

def get_newest_weight_id_per_acnt(account_id):
    database = DB(profile='data_warehouse')
    q = database.query('select max(weight_id) from kim.weights_outlier where account_id = {}'.format(account_id))
    return q.values[0][0]
# get_newest_id('kim.weights_outlier', 'data_warehouse')


def get_account_ids_of_new_weights(newest_id_from_featureDB):
    # search new weigh-ins and return account_ids corresponding the data points
    database = DB(profile='kairos')
    q = database.query('select account_id from public.weights where id > {};'.format(newest_id_from_featureDB))
    return list(set(q['account_id'].values))
# get_account_ids_of_new_weights(get_newest_id('kim.weights_outlier', 'data_warehouse'))

def first_x_weight_ids(num_weight_ids):
    database = DB(profile='kairos')
    q = database.query('SELECT account_id from weights order by id DESC limit {};'.format(num_weight_ids))
    return list(set(q['account_id'].values))

def count_num_processed_only_filtered2_False(account_id):
    database = DB(profile='data_warehouse')
    q = database.query(
    """
    SELECT count(*)
    FROM (SELECT max(id) as max_id FROM kim.weights_outlier GROUP BY weight_id) max_ids
    JOIN kim.weights_outlier wo on max_ids.max_id = wo.id
    WHERE account_id = {}
    AND filtered2 = False;
    """.format(account_id))
    return q.values[0][0]


def get_latest_rows_and_filtered2_false(account_id, window = 10):
    database = DB(profile='data_warehouse')
    # database = DB(profile='kairos')
    q = database.query(
    """
    SELECT id, weight_id, account_id, weighed_at, value, confirmed, manual, filtered, filtered2, value_prediction_w16, feature_col ,processed, model_version
    FROM (SELECT max(id) as max_id from kim.weights_outlier GROUP BY weight_id) max_ids
      JOIN kim.weights_outlier wo on max_ids.max_id = wo.id
    WHERE id BETWEEN
        (
        SELECT min(lastX.id)
        FROM (
          SELECT id
           FROM (SELECT max(id) as max_id from kim.weights_outlier GROUP BY weight_id) max_ids
             JOIN kim.weights_outlier wo on max_ids.max_id = wo.id
          WHERE account_id = {0}
                and filtered2 = False
          ORDER BY id DESC
          limit {1}
             ) lastX
        )
    AND
        (
        SELECT max(lastX.id)
        FROM (
          SELECT id
           FROM (SELECT max(id) as max_id from kim.weights_outlier GROUP BY weight_id) max_ids
             JOIN kim.weights_outlier wo on max_ids.max_id = wo.id
          WHERE account_id = {0}
                and filtered2 = False
          ORDER BY id DESC
          limit {1}
             ) lastX
        )
    AND
        account_id = {0}
    ORDER BY id ASC;
    """.format(account_id,window)
    )
    return q

def get_all_from_kairos_per_acnt(account_id):
    database = DB(profile='kairos')
    q = database.query('SELECT * FROM public.weights WHERE account_id = {};'.format(account_id))
    return q

def get_only_new_from_kairos_per_acnt(account_id):
    newest_weight_id_from_featureDB = get_newest_weight_id_per_acnt(account_id)
    database = DB(profile='kairos')
    q = database.query('SELECT * FROM public.weights WHERE account_id = {} and id > {};'.format(account_id, newest_weight_id_from_featureDB))
    return q

def write_to_featureDB(df, if_exists = 'append'):
    try:
        df.drop('index', axis=1)
    except:
        df
    engine = create_engine(URL(**data_warehouse))
    count = 0
    result = None
    while result is None:
        try:
            df.to_sql(name='weights_outlier', con=engine, if_exists = if_exists, index=False, schema='kim')
            result = 1
        except:
            pass
         
def get_reported_weight(account_id):
    database = DB(profile='kairos')
    try:
        q = database.query(
        """
        SELECT rd.weight  AS risk_data_weight
          FROM registration_program_applications rpa
          JOIN risk_data rd
            ON rpa.risk_data_id = rd.id
          JOIN risk_assessments ra
            ON rd.risk_assessment_id = ra.id
         WHERE rpa.account_id is not NULL
           AND rpa.account_id = {}
        """.format(account_id))
        return q.values[0][0]
    except IndexError:
        return 0

# get_account_ids_of_new_weights(8738664)
