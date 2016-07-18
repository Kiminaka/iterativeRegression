from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from db import DB
import config
import algorithm as al
import db_job

def compare_plots(df):
    df.sort(columns='weight_id',  inplace=True)
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(2,2,1)
    plt.plot_date(df[df['filtered2'] == True]['weighed_at'],df[df['filtered2'] == True]['value'], fmt='ro', label='outlier', alpha=.3)
    plt.plot_date(df[df['filtered2'] == False]['weighed_at'],df[df['filtered2'] == False]['value'], fmt='yo', label='inlier', alpha=.3)
    plt.plot_date(df[df['value_prediction_w16'] != 0]['weighed_at'],df[df['value_prediction_w16'] != 0]['value_prediction_w16'], fmt='b-', label='Predicted value at week 16 on date X')
    ax1.legend(loc='best',ncol=2, bbox_to_anchor=(1, -.4))
    try:
        closest_112 = np.min(df[df['feature_col'] >= 112].weighed_at.values)
        plt.axvline(x=closest_112, color='black', linestyle='--')
    except ValueError:
        fig

    plt.ylabel("Weight")
    plt.xticks(rotation='vertical')
    plt.title("iterative RANSAC")


    ax2 = fig.add_subplot(2,2,2, sharey=ax1)
    plt.plot_date(df['weighed_at'],df['value'], fmt='ro', label='outlier',)
    plt.plot_date(df[df['filtered'] == False]['weighed_at'],df[df['filtered'] == False]['value'], fmt='yo', label='inlier')
    plt.ylabel("Weight")
    plt.xticks(rotation='vertical')
    plt.title("14 pound threshold (with user/health coach inputs)")
    ax2.legend(loc='best',ncol=2, bbox_to_anchor=(1, -.4))


    plt.show()
data_warehouse = config.data_warehouse
kairos = config.kairos

if __name__ == "__main__":
    database = DB(profile='data_warehouse')
    q = database.query('select * from kim.weights_outlier')
    for acnt_id in list(set(q.account_id)):
        print acnt_id,'--------------------------------------------------------------------'
        compare_plots(q[q.account_id == acnt_id])
