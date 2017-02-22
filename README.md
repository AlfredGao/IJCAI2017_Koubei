# IJCAI2017_Koubei
This repository is created for IJCAI2017 main track competition -- Customer Forecast on Koubei
## 02/07/17 Thanks for @kuhung's contributions
- 大家的代码首先push到自己的branch下，然后再merge到master，这样确保master是最高score分的solution :)

### 现在的思路有二：
- A.通过对店家的日销售额的统计，利用滑动时间窗口进行解决。但此方法比较普通，容易想到；且容易丢失user信息。
- B.通过用户、店家的信息，建立起一个推荐系统。用户发生购买，就打标记为1，反之为0。最后再对数据量进行统计。缺点是工程量大，电脑开销大。
(当然，可以最后对两个结果进行融合）

## 02/21/17 Random Forest Score: 0.084326
- Grid Search 出的最佳参数：{'n_jobs': -1, 'min_samples_leaf': 2, 'n_estimators': 1200, 'min_samples_split': 2, 'random_state': 1, 'criterion': 'mse', 'max_features': 237, 'max_depth': 25}
- 采用的Feature：
0. Poly=2 的每店每日销售额。 对于缺失的数据做了两点平滑
1. train_feature['sum']: 时间窗内的销售额总数
2. train_feature['mean']: 时间窗内的销售额均值
3. train_feature['var']: 时间窗内的销售额方
4. train_feature['weekend']: 时间窗内的周末的客流量占总数的比例
5. train_feature['day_pay']: shop_list表中提供的店人均消费额
6. train_feature['city_level']: 根据shop_list对店的城市信息做了分级
 
