# -*- coding: utf-8 -*-
# encoding = uft-8
"""
@Author: Alfred Gao
"""

import numpy as np
import pandas as pd
import warnings


def get_result(result):
    """Double the one week result to two week.
    Args:
        @result: The pandas dataframe of result. Type: pandas.DataFrame
    Return:
        @df: The results dataframe. Type: pandas.DataFrame
    """
    if(len(result.shape)==1):
        df = pd.DataFrame(result,columns=[0])
    else:
        df = pd.DataFrame(result,columns=['col_'+str(i) for i in range(result.shape[1])])
    df.insert(0,'shop_id',[i for i in range(1,2001)])
    df = pd.merge(df,df,on='shop_id')
    return df


def select_days_range(start_date, end_date, data_pandas):
    """Select days range between two dates.

    Args:
        @start_date: The start date of select. Type: datetime
        @end_date: The end date of select.  Type: datetime
        @data_pandas: The data's dataframe.  Type: pandas.DataFrame

    Returns:
        @days_range: A dataframe to coressponding data fetched.
    """
    days_range = data_pandas.loc[:,start_date:end_date]
    return days_range


def find_null(data_pandas):
    """Find the rows contains null value in data frame.

    Args:
        @data_pandas: The data's dataframe. Type: DataFrame

    Return:
        @null_rows: The list contained the index of rows contained null value.
    """
    null_rows = pd.isnull(data_pandas).any(1).nonzero()[0]
    return null_rows


def smooth_null(null_rows, data_pandas_need_smooth):
    """Smooth the null value in data frame. In this function the null value is \
        filled by the mean of row which the null value located.

    Args:
        @null_rows: The list contained the index of rows contained null value. Type: List
        @data_pandas_need_smooth: The data frame need to smooth the null value. Type: pandas.Dataframe

    Return:
        @data_pandas_need_smooth: The null has been smoothed.

    Raise:
        Error: The id list of shop_id whose null value cannot be smoothed. It will happen when\
        all value are null and cannot get mean.
    """
    for null_index in null_rows:
        data_pandas_need_smooth.iloc[null_index] = data_pandas_need_smooth \
            .iloc[null_index].fillna(data_pandas_need_smooth.iloc[null_index].mean(),axis=0)
    null_recheck = find_null(data_pandas_need_smooth)
    if len(null_recheck) != 0:
        warnings.warn("There exist the value cannot be smoothed")
        print ("The shop id is/are: {}").format(null_recheck)
    return data_pandas_need_smooth


def transfer_2dataframe(numpy_arr, col_name='col_'):
    """Transfer Numpy array to dataframe

    Args:
        @numpy_arr: The numpy array that need to be transfered. Type: Numpy.array
        @col_name: The col_name. The default value is 'col_'

    Return:
        @df: The dataframe that tranfer to.
    """
    if(len(numpy_arr.shape)==1):
        df = pd.DataFrame(numpy_arr,columns=['col_0'])
    else:
        df = pd.DataFrame(numpy_arr,columns=[col_name+str(i) for i in range(numpy_arr.shape[1])])
    return df


def cal_offline_score(pre_y, real_y):
    """Calculte the offline score according to the score function on Tianchi.

    Args:
        @pre_y: The prediction value
        @real_y: The true value

    Return:
        @L: The offline score of the prediction.

    Raise:
        Error: When the shapes of prediction and real are different.
    """
    score_sum = 0.
    assert pre_y.shape == real_y.shape
    for shop_index in range(pre_y.shape[0]):
        for day in range(pre_y.shape[1]):
            c_it = pre_y[shop_index][day]
            c_itg = real_y[shop_index][day]
            # score = (pre_y[shop_index][day]-real_y[shop_index][day])/\
            # (pre_y[shop_index][day]+real_y[shop_index][day])
            if c_itg + c_it == 0:
                c_itg = 1.
                c_it = 1.
            score = abs(cit - citg / c_itg + c_it)
            score_sum = score_sum + score
    nT = pre_y.shape[0]*pre_y.shape[1]
    L = score_sum / nT
    return L
