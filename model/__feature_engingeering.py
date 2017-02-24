# -*- coding: utf-8 -*-
# encoding = uft-8
"""
@Author: Alfred Gao
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from __utils import *


shop_date_pay = pd.read_csv('../dataset/preprocessed/shop_date_dapay.csv')
shop_info = pd.read_csv('../dataset/preprocessed/shop_info_0221.csv')
week_1 = select_days_range('2016-09-20','2016-09-26',shop_date_pay)
week_2 = select_days_range('2016-10-11','2016-10-17',shop_date_pay)
week_3 = select_days_range('2016-10-18','2016-10-24',shop_date_pay)
week_4 = select_days_range('2016-10-25','2016-10-31',shop_date_pay)
#----------------------------------Data Clean-----------------------------------
#
#
#------------------------------Train data feature-------------------------------
#Use the week1 week2 week3 as the training set and week4 as y label
#Create the raw data frame of train_feature_raw
train_feature_raw = week_1.join(week_2.join(week_3))
#Find the rows contained the null value in training data feature
train_null_value = find_null(train_feature_raw)
#Smooth the null value
train_feature = smooth_null(train_null_value, train_feature_raw)
#Handle the Null value manually.
train_feature.iloc[809] = [7,5,4,6,9,6,11,7,5,4,6,9,6,11,7,5,4,6,9,6,11]

#------------------------------Train data label---------------------------------
#Use the week2 week3 week4 as the testing data set.
#Find the rows contained the null value in training data feature
label_null_value = find_null(week_4)
#Smooth the null value
train_label = smooth_null(label_null_value, week_4)
#Handle the Null value manually.
train_label.iloc[1706] = train_label.iloc[1706].fillna(train_feature.iloc[1706].mean(), axis=0)
train_label.iloc[1823] = train_label.iloc[1823].fillna(train_feature.iloc[1823].mean(), axis=0)
print len(find_null(train_label))


#----------------------------Construct the feature------------------------------
# train_feature.reset_index()
train_sum = train_feature.sum(axis=1)
print train_sum
train_mean = train_feature.mean(axis=1)
train_std = train_feature.std(axis=1)
train_max = train_feature.max(axis=1)
train_min = train_feature.min(axis=1)
train_median = train_feature.median(axis=1)
train_mad = train_feature.mad(axis=1)
train_var = train_feature.var(axis=1)
week_end = ['2016-09-24','2016-09-25','2016-10-15','2016-10-16','2016-10-22','2016-10-23']
train_weekend_ration = train_feature[week_end].sum(axis=1)/train_sum
train_day_pay = shop_info['per_pay']
train_city_level = shop_info['city_level']
poly = PolynomialFeatures(2, interaction_only = True, include_bias=False)
# print poly.fit_transform(train_feature).shape

train_feature = transfer_2dataframe(poly.fit_transform(train_feature))
train_feature['sum'] = train_sum
train_feature['mean'] = train_mean
train_feature['var'] = train_var
train_feature['weekend'] = train_weekend_ration
train_feature['day_pay'] = train_day_pay
train_feature['city_level'] = train_city_level


# print train_feature
train_feature.to_csv('../train/train_feature.csv')
train_label.to_csv('../train/train_label.csv')
# print train_weekend_ration
train_feature['week1_mean'] = train_feature.loc[:,'col_0':'col_6'].mean(axis=1)
train_feature['week2_mean'] = train_feature.loc[:,'col_7':'col_13'].mean(axis=1)
train_feature['week3_mean'] = train_feature.loc[:,'col_14':'col_20'].mean(axis=1)

#------------------------------START--170222------------------------------------
train_feature = pd.read_csv('../train/train_feature.csv')
train_feature = train_feature.rename(columns = {'Unnamed: 0':'shop_id'})
train_feature['week1_mean'] = train_feature.loc[:,'col_0':'col_6'].mean(axis=1)
train_feature['week2_mean'] = train_feature.loc[:,'col_7':'col_13'].mean(axis=1)
train_feature['week3_mean'] = train_feature.loc[:,'col_14':'col_20'].mean(axis=1)
train_feature['shop_id'] = train_feature['shop_id']+1
shop_info = pd.read_csv('../dataset/preprocessed/shop_info_0221.csv')
shop_info_cat = shop_info.loc[:,'cate_1':'cate_3']
shop_info_cat['shop_id'] = shop_info['id']
cat_statis = train_feature.merge(shop_info_cat,on='shop_id')
cat_null = cat_statis['cate_3'].isnull().nonzero()

#Create cate_3_week1_mean,cate_3_week2_mean,cate_3_week3_mean three features.
train_feature['cate_3_week1_mean'] = 0
train_feature['cate_3_week2_mean'] = 0
train_feature['cate_3_week3_mean'] = 0


#Fillna in cate_3 if it is null. Fill the value using 'cate_2'.
for index in cat_null:
    cat_statis['cate_3'].iloc[index] = cat_statis['cate_2'].iloc[index].values


#Count the mean group by 'cate_3'.
train_three_names = list(cat_statis.loc[:,'col_0':'col_20'].columns) + ['cate_3']
train_three_week = cat_statis[train_three_names]
cat_mean = train_three_week.groupby('cate_3').mean()
cat_mean_week1 = cat_mean.loc[:,'col_0':'col_6'].mean(axis=1).to_dict()
cat_mean_week2 = cat_mean.loc[:,'col_7':'col_13'].mean(axis=1).to_dict()
cat_mean_week3 = cat_mean.loc[:,'col_14':'col_20'].mean(axis=1).to_dict()


for index in range(2000):
    train_feature['cate_3_week1_mean'].iloc[index] = cat_mean_week1[cat_statis['cate_3'].iloc[index]]
    train_feature['cate_3_week2_mean'].iloc[index] = cat_mean_week2[cat_statis['cate_3'].iloc[index]]
    train_feature['cate_3_week3_mean'].iloc[index] = cat_mean_week3[cat_statis['cate_3'].iloc[index]]


train_feature.to_csv('train_feature_0222.csv')
#--------------------------------END--170222------------------------------------


test_feature_raw = week_2.join(week_3.join(week_4))
test_null_value = find_null(test_feature_raw)
test_feature = smooth_null(test_null_value, test_feature_raw)


test_sum = test_feature.sum(axis=1)
print test_sum
test_mean = test_feature.mean(axis=1)
test_std = test_feature.std(axis=1)
test_max = test_feature.max(axis=1)
test_min = test_feature.min(axis=1)
test_median = test_feature.median(axis=1)
test_mad = test_feature.mad(axis=1)
test_var = test_feature.var(axis=1)
week_end = ['2016-10-15','2016-10-16','2016-10-22','2016-10-23','2016-10-29','2016-10-30']
test_weekend_ration = test_feature[week_end].sum(axis=1)/test_sum
test_day_pay = shop_info['per_pay']
test_city_level = shop_info['city_level']
poly = PolynomialFeatures(2, interaction_only = True, include_bias=False)
# print poly.fit_transform(train_feature).shape

test_feature = transfer_2dataframe(poly.fit_transform(test_feature))
test_feature['sum'] = test_sum
test_feature['mean'] = test_mean
test_feature['var'] = test_var
test_feature['weekend'] = test_weekend_ration
test_feature['day_pay'] = test_day_pay
test_feature['city_level'] = test_city_level
# print train_feature
test_feature.to_csv('../train/test_feature.csv')
# print train_weekend_ration


#----------------------------------START--170222--------------------------------
test_feature = pd.read_csv('../train/test_feature.csv')
test_feature['week1_mean'] = test_feature.loc[:,'col_0':'col_6'].mean(axis=1)
test_feature['week2_mean'] = test_feature.loc[:,'col_7':'col_13'].mean(axis=1)
test_feature['week3_mean'] = test_feature.loc[:,'col_14':'col_20'].mean(axis=1)
test_feature = test_feature.rename(columns = {'Unnamed: 0':'shop_id'})
test_feature['shop_id'] = test_feature['shop_id']+1
cat_statis_test = test_feature.merge(shop_info_cat,on='shop_id')
cat_null_test = cat_statis_test['cate_3'].isnull().nonzero()

test_feature['cate_3_week1_mean'] = 0
test_feature['cate_3_week2_mean'] = 0
test_feature['cate_3_week3_mean'] = 0


for index in cat_null:
    cat_statis_test['cate_3'].iloc[index] = cat_statis_test['cate_2'].iloc[index].values


test_three_names = list(cat_statis_test.loc[:,'col_0':'col_20'].columns) + ['cate_3']
test_three_week = cat_statis_test[test_three_names]
cat_mean_test = test_three_week.groupby('cate_3').mean()
cat_mean_week1_test = cat_mean_test.loc[:,'col_0':'col_6'].mean(axis=1).to_dict()
cat_mean_week2_test = cat_mean_test.loc[:,'col_7':'col_13'].mean(axis=1).to_dict()
cat_mean_week3_test = cat_mean_test.loc[:,'col_14':'col_20'].mean(axis=1).to_dict()


for index in range(2000):
    test_feature['cate_3_week1_mean'].iloc[index] = cat_mean_week1_test[cat_statis_test['cate_3'].iloc[index]]
    test_feature['cate_3_week2_mean'].iloc[index] = cat_mean_week2_test[cat_statis_test['cate_3'].iloc[index]]
    test_feature['cate_3_week3_mean'].iloc[index] = cat_mean_week3_test[cat_statis_test['cate_3'].iloc[index]]


test_feature.to_csv('test_feature_0222.csv')
#----------------------------------END--170222----------------------------------
#----------------------------------START--170223--------------------------------
#Just Append the prediction to the last 2 week to be the new prediction set.
