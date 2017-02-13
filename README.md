# IJCAI2017_Koubei
This repository is created for IJCAI2017 main track competition -- Customer Forecast on Koubei
## 02/07/17 Thanks for @kuhung's contributions
- 大家的代码首先push到自己的branch下，然后再merge到master，这样确保master是最高score分的solution :)

### 现在的思路有二：
- A.通过对店家的日销售额的统计，利用滑动时间窗口进行解决。但此方法比较普通，容易想到；且容易丢失user信息。
- B.通过用户、店家的信息，建立起一个推荐系统。用户发生购买，就打标记为1，反之为0。最后再对数据量进行统计。缺点是工程量大，电脑开销大。
(当然，可以最后对两个结果进行融合）
