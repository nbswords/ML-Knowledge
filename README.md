---
tags: 'ML notes'
---

# 機器學習知識庫

面試導向的機器學習筆記

[Toc]

![](https://i.imgur.com/3FRxtwt.jpg)
[Img source](https://towardsdatascience.com/types-of-machine-learning-interviews-and-how-to-ace-them-51587a95f847)

## GPU
### 為什麼 GPU 可以加速矩陣運算?

[How do GPUs speed up Neural Network training?](https://www.youtube.com/watch?v=EKD1kEMNeeU)

1. GPU 的記憶體頻寬更寬，一次可以處理更多資料
2. GPU 提供平行化 (Parallelization) 處理
3. 雖然 GPU 的 cache 和 register 更小，但卻更多、更快，有許多的 streamlined processors (SM)

以矩陣運算的處理來說

1. 一次處理一個element -> 無法利用到任何一個優勢
![](https://i.imgur.com/pa4nqEJ.jpg)
2. 利用GPU Memory 來處理element -> 可利用到平行化的優勢
![](https://i.imgur.com/XgOJhFM.jpg)
3. 將矩陣運算拆成幾個 Block，以 Block 為單位進行運算 -> 可利用到所有優勢
![](https://i.imgur.com/24WW4NN.jpg)
![](https://i.imgur.com/poRKBQM.gif)

而 Block multiplication 的 Block size 跟 SM 的個數以及如何實作這些的細節會由 CUDA 幫忙處理


## Models

### Linear Regression
- 找殘差最小的 best fit of line (用 Least Squares Method 算出模型係數)
- 若結果再加上 Sigmoid 就會變成 Logistic regression

#### Assuption :
1. Linearity: The relationship between X and the mean of Y is linear.
2. Homoscedasticity: The variance of the residual is the same for any value of X.
3. Independence: Observations are independent of each other.
4. Normality: For any fixed value of X, Y is normally distributed.

#### Test for Significance

- Simple regression
  - t-test, F-test 皆可，結果會一樣
- multiple regression
  - t-test, F-test 的結果會不一樣
    - t-test : test for individual significance (測試單一變數)
      - 可能會因為遇上多重共線性問題 (Multicollinearity) 而無法準確使用
    - F-test : test for overall significance (測試所有變數)


#### Coefficient of Determination 

- 利用 $r^2 = SSR / SST$ (Coefficient of Determination )來評估
  - Adjusted $r^2$ 將變數的數量也納入考量，讓模型不會因為沒用的變數越來越多而越來越好 (會導致難以估計係數) 
  - $r^2_{adj} = 1 - (1-r^2)(\frac{n-1}{n-p-1})$

#### Error Term 𝜀 的假設
[Residuals Analysis ](https://www.theopeneducator.com/doe/Regression/residuals-analysis)
1. **normally distributed with zero mean** : The error 𝜀 is a normally distributed random variable with mean of zero
2. **constant (homogeneous) variance** : The variance of 𝜀, denoted by σ2, is the same for all values of the independent variable
3. **uncorrelated** : The values of 𝜀 are independent

#### Interval
- Prediction interval : 對單一個 given x 所預測出的那單一個新 y 會有的區間估計
- Confidence interval : 對單一個 given x 所可能預測出的那群 y 的平均值區間估計
- Prediction interval 的 margin of error 會比較大 (區間較大)

#### Residual Analysis : 確保 Error Term 假設是對的
- 用來估計觀察或預測到的誤差(residuals)與隨機誤差(stochastic error)是否一致
  - 假如沒有一致的話那麼 Interval 和 Significance 都會出錯
- 正常模型的擬合，殘差應該以0為中心並平均散佈在被擬合值點附近，而且是以對稱的形式呈現
  - 正常的殘差圖要表現出隨機性(random)和不可預測性(unpredictable)，殘差不應該包含任何可預測的資訊
- Simple regression 中針對 $\hat{y}$ 和 $\hat{x}$ 的殘差圖結果一樣
- Multiple regression 中則是只會針對 $\hat{y}$ 做殘差圖來確認模型假設是否符合

![](https://i.imgur.com/O1vi3k5.jpg)
![](https://i.imgur.com/fL2pDVm.jpg)

#### Standardization Residual Plot
1. 用來判斷 error term 是否為常態分布
2. 用來判斷 outlier : 若 standardization residual > +2 或是 < -2 就代表這個資料點是 outlier
   - 當遇上異常大的 outlier 的時候會必須改用**學生化殘差(studentized deleted residuals, 又稱t化殘差)**，不過通常就直接用學生化殘差來判斷了
      - studentized deleted residual 的做法是把待測點刪除掉後評估待測點與根據該估計模型預測結果之離差
      -  若 studentized deleted residual 絕對值偏高，表示這個資料點與其他點可能屬不同模型，即這一點是outlier
      - Studentized deleted residual 
        ![](https://i.imgur.com/mt1Bezv.jpg)


#### Outlier vs Leverage vs influential Points
[Outlier, Leverage, and Influential Points](https://www.theopeneducator.com/doe/Regression/outlier-leverage-influential-points)
- **Outlier point** 是相對於 x 或是 y 的異常資料點 (通常是y-outliers比較常見)
  - y-outlier : 通常殘差相較於其他資料點特別高的會是 y-outlier ($|Studentized Residual| > 2$)
  - x-outlier : 通常 diagonal element of the hat matrix, HI 值大於 2p/n 的會是 x-outlier 
    - p 是變數的數量，若為單變數迴歸就是只有 x 和 y，也就是 p=2
    - n 是 observation 的數量 
- **Influential point** 指的是一個足以影響整條迴歸線的資料點
  - 利用 DFFITS (Studentized Difference in fits) 或是 COOK distance 可以找出
    - 其中 DFFITS 代表的是移除第 $i_{th}$ 資料點之後，對單一個 fitted value 的影響
    - Cook's distance 代表的是移除第 $i_{th}$ 資料點之後，對所有 fitted value 的影響  
  ![](https://i.imgur.com/0rlh1v2.jpg)
  - 若 DFFITS 的絕對值在量級為小~中等的資料級下超過1就被判斷為 influential，在大規模的資料集則是以 $|DFFITS|>2\sqrt{\frac{p}{n}}$來判斷
  - 若 COOK distance 非常大也會被判斷為 influential (或是以 $D_i > \frac{4}{n}$、將$D_i$代入F(p, n-p)所產生的機率值是否大於50%來判斷)
    - 可以把 COOK distance 帶入 F分配 (1-FDIST(COOK distance, p , 自由度)之中來判斷這個資料點到底有多 influential，算出來的機率越高代表越 influenital    
![](https://i.imgur.com/Xvw62TM.jpg =200x100)

- **Leverage point** 指的是一個 x值非常異常但其 y 又 follow predicted regression line
  - 這種資料點會使得 p-value 變更小，也會使得r-square 變更大但卻不會影響到模型係數 (可能會導致高估模型)
  - diagonal element of the hat matrix, HI 可以用來找出 Leverage point
![](https://i.imgur.com/3OZFCvm.jpg)
![](https://i.imgur.com/dJSsErG.jpg)
![](https://i.imgur.com/LgJTETK.jpg)

### Logistic Regression
- 加上 Sigmoid 的 Linear Regression
  - 為甚麼是 Sigmoid?
    - 值域 [0,1] 與機率相符
    - 越高越平坦 (越來越難上升) 以及越低越平坦 (越來越難下降) 的 S型函數 (乙狀函數) 特性
![](https://i.imgur.com/IKG2yXq.jpg)
- 使用 Maximum Likelihood Estimation 來估計模型參數，而非最小平方法


#### Assuptions : [Explained](https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290)
1. **Appropriate Outcome Type :** 輸出應該是 Binary，不然就得使用  multinomial or ordinal logistic regression 
2. **The Observations are Independent :** 資料之間不互相影響 
3. **There is No Multicollinearity Among Explanatory Variables :**  變數之間沒有共線性關係，可用 Variance Inflation Factor (VIF) 來測，若 VIF > 5 或 10 可能就有共線性
4. **There are No Extreme Outliers :** 可用 Cook's distance 來測，看是否 > 4/N
5. **There is a Linear Relationship Between Explanatory Variables and the Logit of the Response Variable** : 變數和輸出的機率(logit)之間有線性關係，可用Box-Tidwell Test 或是畫圖測
6. **The Sample Size is Sufficiently Large :** 出現頻率最低的變數至少要有10筆資料，也就是說需要的最小資料量會 = `10*變數數量 / 最低的期望機率`
   - if you have 3 explanatory variables and the expected probability of the least frequent outcome is 0.20, then you should have a sample size of at least (10*3) / 0.20 = 150

#### Comparison with Linear Regression

Differences
- Logistic regression 不需要 x 和 y 有線性關係，但仍需要 x 和預測結果的 log-oddsH有線性關係
- Logistic regression 不需要 Homoscedasticity 
- error terms (residuals) 不需要是常態分布

Similarities
- 無法處理 multicollinearity
- Observations are independent of each other

#### Metrics

- Cross Validation
  - ROC/AUC、Confusion Matrix (Precision/Recall)
- Concordance : 所有預測結果(0,1)成對的機率值中，真實為1的事件，1的機率高於0的機率佔所有成對資料的比例。
  - 理想中，該比例越高越好，即表示所有預測(0,1)的機率值，若真實為1，則1的預測機率理應都大於0的預測機率。（discordance則為相反結果的比例，tied則為機率無差別的結果比例，三比例相加應為100%）。
- Others [img source](https://taweihuang.hpd.io/2017/12/22/logreg101/)
![](https://i.imgur.com/3LrCw01.jpg)

### Ensemble : Bagging, Boosting, Stacking

- 單一個模型都稱為 Weak learner
- 多個模型組合在一起就稱為 Strong learner
- 組合的方式可分為 homogeneous weak learners 跟  heterogeneous weak learners
  - heterogeneous : stacking 
  - homogeneous : bagging and boosting

#### Bagging : Reducing Variance
從訓練資料中隨機抽取(取後放回)樣本訓練多個分類器(要多少個分類器自己設定)，每個分類器的權重一致最後用投票方式(Majority vote)得到最終結果，而這種抽樣的方法在統計上稱為bootstrap

- 又稱 Bootstrap aggregation
- 優點 : 原始訓練樣本中若有 noise data 時，透過 Bagging 的抽樣就有機會不讓 noise data被訓練到，所以可以降低模型的不穩定性和overfitting的可能
- Example : Random forest、KNN
  - Deep decision tree 就是一種 High variance 但 low bias 的模型，因此適合使用 Bagging

Boostrapping : 取後放回的隨機抽樣方法
![](https://i.imgur.com/Rs2Danb.png)

[Workflow of Bagging](https://medium.com/ml-research-lab/bagging-ensemble-meta-algorithm-for-reducing-variance-c98fffa5489f)

![](https://i.imgur.com/3o5U36s.png)

#### Boosting : Reducing Bias

Boosting 是將很多個 weak classifier 先用相同權重初始化，選出其中 error 大的辨識資料 (難以辨識的資料)，並選出能對這些難以辨識的資料進行辨識的模型進行合成，變成一個 Strong classifier

- Boosting 的 Model 和 data 都是有權重的，而 Bagging 的 Model 權重都一樣且資料是隨機抽樣
- 優點 : 在訓練資料太少的時候可以避免 under fitting，也可以避免因為驗證資料太少而 over fitting
- 缺點是無法 done in parallel (unlike bagging)，但其衍伸模型改善了這個問題
- Example : Gradient Boosting, Adaboost, XGBoost
  - Shallow decision tree 就是一種 High Bias 但 low variance 的模型

[Workflow of Boosting](https://medium.com/ml-research-lab/boosting-ensemble-meta-algorithm-for-reducing-bias-5b8bfdce281)

![](https://i.imgur.com/CQt81iv.png)

#### Stacking : improve predictions
stacking 的方法就是訓練各種不同的 model, 然後會有第二層去綜合前面 model 的觀點得到一個新的結果

[Workflow of stacking](https://medium.com/ml-research-lab/stacking-ensemble-meta-algorithms-for-improve-predictions-f4b4cf3b9237)

![](https://i.imgur.com/fpR4BMW.png)

可分為
1. Algorithm Stacking
   - 以不同的model訓練後作為feature, 當作第二層model的feature跟原本的feature參雜在一起. 在這樣的思維下就是用不同人的觀點綜合之後可以得到一個比較沒有偏見的答案所得出的結果
2. Features Stacking
   - 切分不同的feature subset訓練後當作feature, 當作之後model learning的feature做不同的weak learner，之後再組合起來一起學習
3. Dataset Stacking
   - 切分不同的dataset訓練後當作不同層的training set
   - 這個思維跟 bagging 不一樣的點在於 bagging 想做到的事情是用同樣的 classifier 去訓練不同子集合的 dataset, 想避免的事情是某個演算法過度optimize當前訓練集的結果
   - 而在stacking裡面的做法其實是在第一層的weak learner的時候都用同一個subset,得到的結果去用在第二層的model裡面當作feature predictor, 再用另一個subset來做第二層leaner的訓練集。

**將各種不同的模型 stacking 起來的方法 :**
1. 訓練另一個模型來綜合，比如線性、NN
2. Averaging
![](https://i.imgur.com/tQFUAXu.png)
4. Weighted Averaging
![](https://i.imgur.com/7VbWeor.png)

### CNNs

[CS231n](https://cs231n.github.io/convolutional-networks/)
[ML Lecture 10: Convolutional Neural Network](https://www.youtube.com/watch?v=FrKWiRv254g)
#### Indutive bias of CNN: 
1.  Locality : Pixels near one another are related 相連的pixel是有關連的
2.  weight sharing : Different portions of an image should be processed identically regardless of their absolute location 一張圖片中的不同區域都會做一樣的處理 (同一個 Filter 在不同的圖片區域上的 weight 都是一樣的)
    
#### 為甚麼要做 Pooling?

方法通常有 Max Pooling, Mean Pooling, Stochastic-Pooling

1. 減少後續layer需要參數，加快系統運作的效率
2. 具有抗干擾的作用：圖像中某些像素在鄰近區域有微小偏移或差異時，對Pooling layer的輸出影響不大，結果仍是不變的 (translation invariance、rotation invariance、scale invariance)
   - 比如手寫辨識圖片的數字可能會有些微的不同，圖一的數字偏左，圖二的數字偏右一點，此時圖一到圖二只差一個單位，那麼就算把圖片從16x16做Pooling變成8x8也會得到一樣的特徵和位置 (也就是相同這種情況的圖片都會得到這樣的結果)
   - Translational invariance (平移不變性)
![](https://i.imgur.com/iMiB1W1.jpg)
   - Rotation invariance (旋轉不變性)
![](https://i.imgur.com/RLX7CIb.jpg)
   - Scale Invariance (尺度不變性)
![](https://i.imgur.com/VFONFf5.jpg)


#### 跟 Fully-connected layer 的差別 ?
[CNN vs MLP in image](https://medium.com/analytics-vidhya/cnn-convolutional-neural-network-8d0a292b4498)
- CNN 其實就是 FC layer 拿掉一些layer的結果
  - CNN 是只對跟 filter 同樣大小的區域(input)做 Fully connected 算 inner product
  - 而 FC layer 是對於整張圖片做 Fully connected 算 inner product

#### 圖片的輸出大小

公式:  $(W−F+2P)/S+1$
- $W$ : Input image size
- $F$ : Filter size
- $P$ : Amount of zero-padding
- $S$ : Stride size
Example : 
- Input 7x7 image and 3x3 Filter with Stride 1 and pad 0
- Will ouput $(7-3+2*0)/1 + 1 =$ 5x5 output image
  - 計算上需要注意若無法算出整數的 output image size 就無法使用這個參數 !

Convolution有多少個Filter，Output就會有多少個Filter (即使疊了兩個Convolution也是一樣，只是Filter的高會變)
- 第一個Conv filter中有9個參數
- 第二個Conv filter中有25*9=225個參數
![](https://i.imgur.com/v2z3WDr.jpg)

### RNNs

#### Indutive bias of RNN:
假設資料在時序上是有關係的，歷史資料會影響未來資料

### Attentions

#### ViT vs CNN 相關研究
[Differences of ViT and CNNs representation](https://medium.com/syncedreview/google-brain-uncovers-representation-structure-differences-between-cnns-and-vision-transformers-83b6835dbbac)


### GNN

[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)


### 模型壓縮

#### Parameter pruning

- 可分為 one-shot 和 iteration pruning
  - **One-shot** : Train -> 評估神經元、layer、kernel 的重要性 -> 去掉最不重要的 -> Fine-tuning -> 停止 pruning
    - 又稱靜態剪枝 
  - **Iteration** : Train -> 評估神經元、layer、kernel 的重要性 -> 去掉最不重要的 -> Fine-tuning -> 判斷是否繼續 pruning -> 若不需則停止 pruning
    - 又稱動態剪枝 
- 也分為 Structured 和 unstructured pruning
  - **Structured** : 直接去掉整個 channel、kernel、layer 的結構化資訊 (直接刪掉一個layer之類的)
     - 雖然壓縮率低但適合在硬體上運作 
  - **Unstructured** : 考慮每個 kernel 中的每個element，刪除其中不重要的參數
    - 也稱為 sparse pruning，因為其得到的權重矩陣會是稀疏的，需要一些硬體去特別加速
    - 雖然壓縮率高但不太適合在硬體上運作
![](https://i.imgur.com/psHYeKf.jpg)


Parameter pruning 基本流程
![](https://i.imgur.com/WsobBlU.jpg)
- 是否要 Pruning 是基於前面那個預訓練所得到的權重去判斷的
- 其中 Fine-truning 是指要微調模型以重新獲得因為丟失參數而失去的能力
  - 通常會將 prune 之後的網路中被保留下來的參數(被保留下來的參數就是重要參數)用來初始化這個修剪後的網路，再重新在訓練資料集上訓練

矛盾的點? [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)
   1. 模型過度參數化，只留下訓練資料集上有效的參數不代表放到現實就會有效
   2. 大模型的重要權重不一定有助於修剪後的小模型
   3. 修剪網路應該以整體架構為主而非利用繼承重要參數來主導其修剪

#### Lottery Ticket Hypothesis 

- 2019年發表的ICLR best paper[The Lottery Ticket Hypothesis： Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
- 提出只要對模型的訓練時所產生的子網路進行很好的重新初始化

什麼是彩票假設?
- 任何密集、隨機初始化的包含子網路(彩票網路)的前饋網路 ，當將其隔離訓練時，可以在相似的疊代次數內達到與原始網路相當的 test accuracy

如何得到彩票網路?
利用 iteration pruning 方式找
1. 隨機初始化神經網路
2. 訓練這個網路直到收斂
3. prune 掉部分權重
4. 將修剪後的網路利用步驟1的方式隨機初始化
5. 判斷步驟4的網路是否為彩票網路，訓練子網路來比較其 test accuracy

這個彩票(子)網路有什麼特徵?
- 在相同的迭代次數下可以達到跟原始網路一樣的結果
- 訓練上比原始網路更快泛化且 test accuracy 更快

**現況就是大家還是以傳統方法居多，但持續在研究如何有效地做出彩票網路** - 2021/8/4

#### knowledge distillation


## Training Model

Bias-Variance TTradeoff
![](https://i.imgur.com/vkaEmkn.jpg)

### Overfitting

Ovefitting : training error 低但 testing error 高

原因 : 
1. 模型過於複雜 (模型容量過高)
2. 每一種資料都太少，無法找出真正通用的特徵

如何避免或是解決 : 
1. 取得更多有用的資料
2. 減少層數、參數來降低模型複雜度
3. Early Stopping
4. 在不改參數和層數的情況下做 Dropout
5. 在不改參數和層數的情況下做 Regularization



### Regularization (正規化)
[Regularization 方法 : Weight Decay , Early Stopping and Dropout](https://hackmd.io/@allen108108/Bkp-RGfCE)

[Lasso vs Ridge vs Elastic Net | ML
](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/)

[Regularization in ML](https://towardsdatascience.com/types-of-regularization-in-machine-learning-eb5ce5f9bf50)
#### Weight decay(權重衰減) : Lasso and Ridge regression
- 就是 L1 / L2 Regularization
- 用來判斷模型的複雜度並加以懲罰
- 越大的weight懲罰就越大，越小的weight懲罰就越小
    - L1, Lasso 當n很小時至多只能選出n個變量，而且無法處理 group selection，也就是說當有一群參數之間的 pairwise correlations 很強的時候，Lasso 只會從這群參數中取一個出來 
    - L2, Ridge 減少模型複雜度但沒有減少參數的個數，係數永遠不會是0，不能用來做 feature reduction
- 為什麼加入 L1 / L2 就會造成 Weight decay ?
    - L2 在進行參數更新的時候會乘上(1−2ηλ)，其值小於1，必定會使參數遞減並往0靠近
- L1 + L2 的結合 = Elastic Net，解決兩者缺點
![](https://i.imgur.com/YAfg2Pk.jpg)
![](https://i.imgur.com/BnVE25T.jpg)
![](https://i.imgur.com/Gg9SZ8m.jpg)

<center class="half">
    <img src="https://i.imgur.com/n3ADxGz.png" width="400">
    <img src="https://i.imgur.com/aAc4tGl.png" width="400">
</center>

#### Early Stopping
  - 判斷何時需要停止訓練，圖中　$d^*_{vc}$ 為最佳停止時間 
    ![](https://i.imgur.com/K4Roa4W.png)
    
#### Dropout
  - 在每一層隨機停用某部分(p%)的神經元 
  - 當這些利用 p% 所訓練出來的權重要拿到 test data 評估的時候，因為 testing 的時候不使用 Dropout，所以 testing 時所有權重要乘上 (1-p)%
![](https://i.imgur.com/YFeWGB7.png)


#### 其他 regularization 方法

- Stochastic Depth
- Label Smoothing
- Layer Scale

### Batch Normalization

[batch normalization in 3 level](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338)
![](https://i.imgur.com/pppjMSE.png)
- Batch Normalization 的作法是對每一個 mini-batch 都進行正規化到平均值為0、標準差為1的常態分佈
  - **適合使用在 MLP 或是 CNN 之上，不適用在 RNN 或是 batch size 小的情況上**，因為其 batch 中的資訊並沒有辦法反映在全局的統計分布上
- 這樣可以把分散的數據統一，有助於減緩梯度消失以及解決 Internal Covariate Shift 的問題，同時可以加速收斂，並且有正則化的效果 (可以不使用Dropout)
  - γ and β 為控制 linear/affine transformation 的可學習參數，在 [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) 中可透過 `affine` = true / false 來開關
![](https://i.imgur.com/7eIU1Oc.jpg)

- 經過 Normalization 之後的數據在通過激活函數後，可以得到分佈較為平均的輸出
![](https://i.imgur.com/6zDexYJ.png)
![](https://i.imgur.com/NdzE2Rs.jpg)

優點 : 
1. 減緩梯度消失的問題
2. 解決 Internal Covariate Shift 的問題
3. 加速模型收斂
4. 具有正則化效果

缺點 :
來自 [NFNets](https://arxiv.org/abs/2102.06171) 的論點
1. 在計算平均數和標準差時，需要將其值保存在記憶體中，除了會增加記憶體使用量(計算成本)，也增加了網路評估梯度的時間
2. 造成模型訓練和推論時的差異，代表 BN 引入了必須調整的隱藏超參數
3. 破壞了 minibatch 的訓練資料間的獨立性，因此在 minibatch 中選擇哪些樣本變得很重要

因此造成
1. 難以進行分布式訓練 (因為 Train-test inconsistency 容易造成資料洩漏)
2. Batch size 不能太小，會很不穩定，通常BN只能用在大模型上，也因此導致許多需要大量 gpu 記憶體的 tasks( Detection, segmentation, video) 無法使用 BN
   - 此時可改用 [Group Normalization, 2018](https://arxiv.org/abs/1803.08494)

除此之外，在 [Rethinking “Batch” in BatchNorm, 2021](https://arxiv.org/pdf/2105.07576.pdf) 中也提出更多 BN 的缺點，比如 batch 的概念較模糊因此實作方法多、實作上容易有 bug 等等

#### 如何改善 Batch Norm 問題?

[Exploring Adaptive Gradient Clipping and NFNets](https://wandb.ai/ayush-thakur/nfnet/reports/Exploring-Adaptive-Gradient-Clipping-and-NFNets--Vmlldzo1MDc0NTQ)

- [NFNet, 2021](https://arxiv.org/abs/2102.06171) 中提出了自適應梯度修剪（Adaptive Gradient Clipping，AGC）方法，是基於梯度範數與參數範數的單位比例來裁切梯度
  - 利用 history of gradient norms 去設定 clipping value
  - choose a percentile $p$ instead of a absolute value as clipping threshold
  - AGC 可以訓練更大 Batch size 和大規模數據增強的非歸一化網絡，但訓練穩定性會對 λ 特別敏感
![](https://i.imgur.com/GY10SpH.jpg)


### Layer Normalization, Instance Normalization, Group Normalization, 
[In-layer normalization techniques for training very deep neural networks](https://theaisummer.com/normalization/)
![](https://i.imgur.com/ZkM0SdM.png)
![](https://i.imgur.com/dERQg9m.jpg)


#### Layer normalization
- 首次於 [2016年](https://arxiv.org/abs/1607.06450) 被提出
  - 一開始是用來處理 vector (大多是 RNN 的輸出)，但一直默默無名，直到 transformers 出現之後才又被積極討論
  - **除了 RNN 與 Attention 以外也適合使用在 batch size 較小的任務上**
- 跨越所有 channels 以及 spatial dimension 將每一個 feature 的 activations 正規化到 zero mean and unit variance
  - **獨立於 batch 之外 (Batch independent)** 為其最重要的特性
![](https://i.imgur.com/m6iSv4Q.png)

算法:
- γ and β 為控制 linear/affine transformation 的可學習參數，在 [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm) 中可透過 `elementwise_affine` = true / false 來開關
![](https://i.imgur.com/6x1Rj09.jpg)

#### Instance normalization
- 與 LN 一樣在 [2016](https://arxiv.org/abs/1607.08022) 被提出，IN 主要應用在特徵較密集的電腦視覺領域以及每一個 pixels 都有用處的演算法上 (例如 GAN)
  - **不建議被用在 (1) MLP 或 RNN，因為其一個 channel 上只有一個資料 (2) feature map 較小時**
- 只在單一個 channel、單一個樣本中每個 feature 的空間維度之中計算
  - **獨立於 channel 和不同樣本之外 (independent for each channel and sample)**
![](https://i.imgur.com/Oezntqt.png)
- 本質上就是在 normalize features，因此可用來改變一張圖片的風格 (透過可學習參數 $\gamma$ 和 $\beta$)

算法:
  - 其實就只是把 BN 中的 $\mu$ 和 $\sigma$ 去掉 $N$
  - γ and β 為控制 linear/affine transformation 的可學習參數，在 [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html#torch.nn.InstanceNorm1d) 中可透過 `affine` = true / false 來開關
![](https://i.imgur.com/2HlNxGT.jpg)

#### Group normalization
- 於 [2018](https://arxiv.org/abs/1803.08494) 被提出，應用在圖片分類、物件偵測、物件分割等視覺任務上
- GN 把channel 分成 num_group 個 group 並分開個別對其做 normalization (算出個別的 mean 和 var)
  - 一樣**獨立於 batch 之外**
  - num_group 為一個可調的超參數，num_group=1 就是 LN
![](https://i.imgur.com/KwOfHr7.png)
- 在各種不同的 batch size 下，GN 擁有比 BN 更穩定的準確率
![](https://i.imgur.com/HnLUCEv.jpg)

算法:
  - $G$ = num_group，超參數，論文中以 G=32 做為預設參數
  - $C/G$ =  number of channels per group
  - $S_i$ = 第 $i$ 個 Set
  - GN computes µ and σ along the (H, W) axes and along a group of C/G channels
  - γ and β 為控制 linear/affine transformation 的可學習參數，在 [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm) 中可透過 `affine` = true / false 來開關
![](https://i.imgur.com/KrqMLGd.jpg)
![](https://i.imgur.com/QXQIpCp.jpg)


### Learning Rate Schedules

[Pytorch optIM](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

[Guide to Pytorch learning rate scheduling](https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling)

#### LinearLR: 用兩個 multiplicative factor，一個作為起始，另一個作為終點來線性衰減學習率直到 total_inters
- `scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)`

#### ExponentialLR: 用一個固定的 Multiplicative factor `gamma` 來衰減每個 epoch 的學習率
- `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)`
![](https://i.imgur.com/0wYjKqa.png)

#### CosineAnnealing: 透過餘弦退火可以讓學習率先緩降->再驟降->最後緩降
- `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0)`
    - T_max: 最多迭代次數
    - eta_min: 最小學習率
![](https://i.imgur.com/Ay8msGd.png)
- 學習率一個週期(完成一次 epochs=50 的訓練)的變化
![](https://i.imgur.com/7GP73Bf.png)
- Implementation
```python=
global_step = min(global_step, decay_steps)
cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
decayed = (1 - alpha) * cosine_decay + alpha
decayed_learning_rate = learning_rate * decayed
```

#### CosineAnnealingWarmRestarts: 週期性的使用 cosine annealing
- `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0)`
   - T_0: First restart 後的迭代次數
   - T_mult: 在 restart 後增加迭代次數的 multiplicative factor，預設為1
   - eta_min: 最小學習率
![](https://i.imgur.com/iV5NzvD.png)

- 週期性的使用 cosine annealing 所形成的學習率變化

![](https://i.imgur.com/t21inCG.png)
#### CyclicLR: 根據 cyclical learning rate policy (CLR) 在一個設定好的區間內以 constant frequency 來循環學習率
- `torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr)`
- base_lr 和 max_lr 會決定學習率循環的區間
- 分為三種循環方式：triangular, triangular2, exp_range, 預設參數為mode='triangular'
1. triangular
![](https://i.imgur.com/lpGnSXV.png)
2. triangular2
![](https://i.imgur.com/7LQzvMy.png)
3. exp_range
![](https://i.imgur.com/duOZRRd.png)

#### OneCycleLR: 根據 1cycle learning rate policy 來從初始學習率退火到一個最大學習率，再從這個最大學習率退火到一個比初始學習率還小的最小學習率
- `torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)`
- total_steps = epochs * steps_per_epoch，要嘛就指定 step_per_epoch 要嘛就指定 total_steps
- 退火策略有兩種可以選: linear 和 consine，預設參數為 anneal_strategy='cos'
1. consine
![](https://i.imgur.com/rgV2mHl.png)
2. linear
![](https://i.imgur.com/dkhKYfj.png)

### Loss
#### Regression Loss
- $MSE, L2 loss = {\displaystyle\sum_{i=1}^{D}(x_i-y_i)^2}$ 
   - MSE 計算方便但遇到 outlier 會較不穩定 
- $RMSE = {\displaystyle\sqrt(MSE) = \sqrt(\sum_{i=1}^{D}(x_i-y_i)^2)}$
- ${\displaystyle {\mbox{MAE}} = \sum_{i=1}^{D}|x_i-y_i|}$
  - MAE 較能處理 outlier 但收斂速度較慢 
- $Huber loss = L_{\delta}=
    \left\{\begin{matrix}
        \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
        \delta ((y - \hat{y}) - \frac1 2 \delta) & otherwise
    \end{matrix}\right.$
  - It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.
  - HuberLoss 的存在就是希望能降低MSE對Outlier 的不穩定性，並提升 MAE 的收斂速度
 - ${\displaystyle {\mbox{MAPE}}={\frac {100}{n}}\sum _{t=1}^{n}\left|{\frac {A_{t}-F_{t}}{A_{t}}}\right|}$
     - $A_t$是實際值
     - $F_t$是預測值
     - 若有資料會等於0不可用
     - 主要是用來比較某兩者，考慮的是相對誤差
     - 比如兩間飲料店賣的飲料數，這樣才可以把真正的實際值也考慮進去，才不會把
其他還有 MSPE, MSLE...等等

##### Summary

MSE & MAE: 考慮的是絕對誤差，MSPE & MAPE: 考慮的是相對誤差
- MAE: 有異常值的情況，如果不想要這些異常值影響模型可以用
- MSE: 有一點異常值的情況，如果想要包含這些異常值可以用
- MSPE: 具有權重概念的MSE
- MAPE: 具有權重概念的MAE
- MSLE: 取對數(log)的MSE

#### Classification Loss

- $Cross Entropy = -{(y\log(p) + (1 - y)\log(1 - p))}-\sum_{c=1}^My_{o,c}\log(p_{o,c})$
    - M - number of classes
    - log - the natural log
    - y - binary indicator (0 or 1) if class label c is the correct classification for observation o
    - p - predicted probability observation o is of class c
- $Negative Loglikelihood = NLL(y) = -{\log(p(y))}\min_{\theta} \sum_y {-\log(p(y;\theta))}\max_{\theta} \prod_y p(y;\theta)$
- $Hinge loss = max(0, 1 - y \cdot \hat{y})$
  - Used in SVM
- KL/JS divergence
![](https://i.imgur.com/ZwmeYO9.jpg)


### Data Splitting
[Data Splitting for Model Evaluation](https://towardsdatascience.com/data-splitting-for-model-evaluation-d9545cd04a99)

#### Imbalanced Dataset
*Question: What problem may arise if we randomly split a dataset that has 99% negative class and 1% positive class?*
- 最好使用 stratify 的方式來將資料切割為 class 數量一樣的 train/test split
- 即使不是不平衡資料集也可以做，這樣可以確保 training 和 testing dataset 有類似的資料分布
```python=
from sklearn.model_selection import train_test_split
# 根據 y 的類別數來切割出每個類別都一樣的 train 和 test
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42, stratify=y)
```

#### Limited Data
*Question: In experiments with limited data, what could be a possible model evaluation issue if we perform a simple train-test split?*
- 可以利用 K-Fold CV 來解決資料較為不足的情況
![](https://i.imgur.com/s2Vozh8.png)

#### Feature Engineering Leakage
*Question: What’s wrong with standardizing on the whole data before doing a train-test split?*
- 假如我們在做資料分割之前就對整個資料做標準化，就會造成 data leakage，因為我們會透過 mean 以及 std 將 testing data 的特性洩漏給 training data
- 除此之外另一個常見的則是對整個資料做 Hot deck imputation

#### Group Leakage
*Question: If there are patient overlaps between the train, validation, and test sets, why might the model performance be greatly overestimated?*

- 要改用 object identifier 來切割資料，而不是 data points，這樣同一個物件所產生的資料只會存在於 training　data 或是 testing data

#### Time Leakage
*Question: Any issue with randomly splitting a time series data?*

- 按照時間順序進行 train/test 切割 (以較舊資料作為 train, 較新資料作為 test)，或是使用 walk-forward validation

## Tensorflow / Pytorch

[浅谈 PyTorch 中的 tensor 及使用](https://zhuanlan.zhihu.com/p/67184419)
[PyTorch 指南：17個技巧讓你的深度學習模型訓練變得飛快！
](https://bangqu.com/Ya9W74.html)


- torch.no_grad()
  - 在做 inference 的時候可以不追蹤模型的參數，因此可以將其包在no_grad()底下減少浪費的計算量
```python=
# requires_grad 是用來確認一個tensor是否需要求導
x = torch.randn(3, requires_grad = True)
print(x.requires_grad)
# True
print((x ** 2).requires_grad)
# True

with torch.no_grad():
    print((x ** 2).requires_grad)
    # False

print((x ** 2).requires_grad)
# True

```
- torch.ones
```python=
# 模擬8張圖片，每張圖片的大小是10*10
# 每個目標都初始化為1
images = torch.ones(8, 3, 10, 10)
targets = torch.ones(8, dtype=torch.long)
```

## Numpy / Pandas

[Python Numpy and Matrices Questions for Data Scientists](https://towardsdatascience.com/python-numpy-and-matrices-questions-for-data-scientists-167af1c9d3a4)

- pd.melt
  - 把複雜的資料拆解掉變成簡單的表格式資料 
  - `pandas.melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)`
	  - id_vars: 可使用 tuple、list、或 ndarray，用以設定不想要被轉換的欄位
	  - value_vars: 可使用 tuple、list、或 ndarray，用以設定想要被拆解的欄位。 如果省略則拆解全部欄位
	  - var_name : 轉換後 id 的名稱。如果省略則設定為原本 DataFrame 的欄位名稱或是 variable。
	  - value_name : 轉換後 value 欄位的名稱。如果省略則顯示原本 DataFrame 的欄位名稱或 value。
	  - col_level : 可使用 int、string。如果 columns 是 MultiIndex，則使用該參數來進行選擇。

## Sklearn

- [News in Scikit-learn 1.0](https://hackmd.io/7RlOjqQfQp2HlNpPc-i6ZQ)

- [Balanced_accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
	- 處理分類回歸任務的不平衡資料時需改成balanced_accuracy
    - balanced accuracy = (sensitivity + specificity) / 2
    	- sensitivity (Recall) = TP / (TP+FN) 實際為陽性的樣本中，判斷為陽性的比例zz
    	- specificity = TN / (FP+TN) 實際為陰性的樣本中，判斷為陰性的比例
	- accuracy = (TP+TN) / (TP+TN+FP+FN)

舉例來說

|          | Predicted Positive | Predicted Negative |
| -------- | -------- | -------- |
| Actual Positive     | 1        | 8                 |
| Actual Negative     | 2        | 989               |

此時 accuracy = 990 / 1000 = 99%
而 balanced accuracy = (((1/(1 + 8)) + ( 989/(2 + 989))) / 2 = 55.5%

## SQL

### CheatSheet

![](https://i.imgur.com/atEPqS0.jpg)

### JOIN

- INNER JOIN (交集): 僅顯示兩資料表對應欄位中值相同的欄位
- LEFT JOIN : 串聯兩個資料表中對應欄資料時，以資料表1的資料為主，若資料存在於資料表1，但資料表2沒有對應值時，仍顯示資料表1中的資料。
- RIGHT JOIN 串聯兩個資料表中對應欄資料時，以資料表2的資料為主，若資料存在於資料表2，但資料表1沒有對應值時，仍顯示資料表2中的資料。

![](https://i.imgur.com/a6YSlmW.jpg)


### 常見面試題
[5 Common Problems](https://towardsdatascience.com/5-common-sql-interview-problems-for-data-scientists-1bfa02d8bae6)
#### Second Highest Salary
Question : 找出Salary第二高的是多少
```
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
```
Ans :
- 解一 : 利用 MAX() 找出不等於 MAX 的MAX值
```sql=
SELECT MAX(salary) AS SecondHighestSalary
FROM Employee
WHERE salary != (SELECT MAX(salary) FROM Employee)
```
- 解二 : 利用  IFNULL 找出非NULL值以及用 OFFSET 找出第二高的值
```sql=
SELECT
    IFNULL(
        (SELECT DISTINCT Salary
        FROM Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1 --限制只取一筆且略過第一筆
        ), null) as SecondHighestSalary
FROM Employee
LIMIT 1
```

#### Duplicate Emails
Question : 找出所有在 table `person` 中有重複的 eamils
```
+----+---------+
| Id | Email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+
```
Ans :
- 解一 : 用 COUNT 
```sql=
SELECT Email
FROM (
  SELECT Email, count(Email) AS count
  FROM Person
  GROUP BY Email
) as email_count
WHERE count > 1
```
- 解二 : 用 HAVING
```sql=
SELECT Email
FROM Person
GROUP BY Email
HAVING count(Email) > 1
```
#### Rising Temperature
Question: 給定一個`Wether`Table, 找出所有前一天的溫度比今天高的 DATE_Id

```
+---------+------------------+------------------+
| Id(INT) | RecordDate(DATE) | Temperature(INT) |
+---------+------------------+------------------+
|       1 |       2015-01-01 |               10 |
|       2 |       2015-01-02 |               25 |
|       3 |       2015-01-03 |               20 |
|       4 |       2015-01-04 |               30 |
+---------+------------------+------------------+
```
Ans : 
- 利用 DATEDIFF(startdate, enddate) 來確定今天和昨天是否差一天
```sql=
SELECT DISTINCT a.Id
FROM Weather a, Weather b
WHERE a.Temperature > b.Temperature
AND DATEDIFF(a.Recorddate, b.Recorddate) = 1
```

#### Department Highest Salary
Question : 給定`Employee`Table 和 `Department`Table, 找出每一個 Department 中薪水最高的人是誰、住哪和其薪水

```
Employee
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Jim   | 90000  | 1            |
| 3  | Henry | 80000  | 2            |
| 4  | Sam   | 60000  | 2            |
| 5  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
```
```
Department
+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+
```
```
Expect Output
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| IT         | Jim      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+
```

Ans : 

```sql=
SELECT Employee.Name AS 'Employee',
       Department.Name AS 'Department',
       Salary
FROM Employee
INNER JOIN Department ON Employee.DepartmentId = Department.Id
WHERE (DepartmentId, Salary)
IN ( SELECT-- SELECT Highest Salary and Id in Department
         DepartmentId, MAX(Salary)
     FROM
         Employee
     GROUP BY DepartmentId
)
```
#### Exchange Seat
Question : 給定一個 `seat` Table, 其中的 id 是一個連續增加的值, 希望把座位改成原本相鄰的人不再相鄰
- 提示: 若座位數是奇數, 則不必改變最後一個人的座位
```
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Abbot   |
|    2    | Doris   |
|    3    | Emerson |
|    4    | Green   |
|    5    | Jeames  |
+---------+---------+
```
Ans : 
```sql=
SELECT 
    CASE 
       -- 判斷是否為奇數個 row, 若是, 則最後一個座位不動
       WHEN((SELECT MAX(id) FROM seat)%2 = 1) 
       AND id = (SELECT MAX(id) FROM seat) THEN id
       -- 對每一個奇數座位都+1 (1,3,5) -> (2,4,6)
       WHEN id%2 = 1 THEN id + 1
       -- 偶數座位-1 (2,4) -> (1,3)
       ELSE id - 1
    END AS id, student
FROM seat
ORDER BY id
```

## Basic ideas ( Statistic related )

### 常態性檢測 ( Normality Testing )
- 用來檢測資料是否為常態分布的各種方法

#### 常態分布的特性

- 平均值、中位數和眾數，三者是同一個值
  - 若為標準常態分布，其 $\mu =0$，$\sigma ^{2} = 1$
- 大約有 68% 的觀測值會落在中央左右二側的一個標準差 σ 之內，95% 的觀測值會落在二個標準差之內
- 常態曲線以平均值 μ 為中心，左右兩側正負一個標準差 σ 的地方，即曲線上所謂的反曲點 (infection point)
- 理論上這個曲線會向二個尾端無限延伸
- 常態分布的資訊熵在所有的已知均值及變異數的分布中最大

#### 有什麼是建立在常態分布的假設之上? 

只要整體資料夠多，適用於中央極限定理的話，整體資料分布可以不符合，只要sample mean符合常態分布就好
- 所有的有母數方法 (Parametric Statistical Methods)都建立在常態分布的假設之上
  - 包含 t-test、ANOVA、Pearson coefficient of correlation
  [Parametric and Nonparametric: Demystifying the Terms 
](https://www.mayo.edu/research/documents/parametric-and-nonparametric-demystifying-the-terms/doc-20408960)
![](https://i.imgur.com/32SHeob.jpg)
- 在用任何 parametric statistical tests 之前都應該先測試 normality，否則就只能使用  non-parametric  statistical tests
#### 常態性檢測可分為圖形檢驗和統計檢驗兩種

1. 圖形檢驗 : 用 Histogram、Boxplot、Q-Q plot 看出資料的分布情形，也可用來比對跟標準情況的差別
   - Hisogram : bell-shaped
   ![](https://i.imgur.com/oy6Q3M1.jpg)
   - [Boxplot](https://www.simplypsychology.org/boxplots.html) : 中位數在正中間
![](https://i.imgur.com/yIWYW48.jpg)
   - Q-Q Plot : 資料呈現會近似 y=x
![](https://i.imgur.com/iB7Tm6Q.jpg)


2. 統計檢驗 : 用 Shapiro-Wilk test、Kolmogorov-Smirnov test、Anderson-Darling test、D'Agostino's K-squared test等測驗去檢測
[machinelearning mastery 實作](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
   - **Shaporio-Wilk test** : 資料小(n<50)的時候表現最好，不過資料更多也可以使用，資料非常態的情況也很適合   
   - **Kolmogorov-Smirnov test (K-S test)** : 適用於資料大(n>50)、基於 cdf 因此能對中央的資料密度更敏感
   - **Anderson-Darling test (A-D test)**: K-S test 的修正版本，能對尾端的資料密度更敏感，通常比K-S test更推薦使用
   - **D'Agostino's K-squared test** : 利用資料的 Skewness 和 Kurtosis 來判斷是否為常態分布
      - [Skewness (偏度)](https://zh.wikipedia.org/zh-tw/%E5%81%8F%E5%BA%A6) 是在衡量資料分布的不對稱性 
        - 正偏態(右偏態)資料集中在左邊，負偏態(左偏態)資料集中在右邊
      ![](https://i.imgur.com/slWRo8K.png)
      - [Kurtosis (峰度)](https://brewcode.stringlab.org/what-is-kurtosis-and-its-significance/) 是在衡量資料的 tail 有多大，Excess Kurtosis (超峰度) 則是在衡量它的 tail 和常態分布的 tail 有甚麼差
          -  超峰度為正稱為 leptokurtic
          -  超峰度為負稱為 platykurtic
          -  超峰度為0就是常態分布，稱為mesokurtic
![](https://i.imgur.com/KCEr2OC.jpg)

#### 可以如何選擇要用哪個統計檢驗方法?
[Normality Tests for Statistical Analysis: A Guide for Non-Statisticians](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/)
[無母數統計 by AIA](http://www.hmwu.idv.tw/web/R_AI/v2/hmwu_StatR-05-1_NonParametric_basic.pdf)

視覺化方面
- 若只有一個變數就用 Q-Q plot
- 有多個變數就用 Boxplot
- 若需要呈現給非專業人員就可以畫出 histogram


統計檢驗方面
- 優先使用 Shapiro-Wilk test
- 若是在對常態分布以外的機率分布做測試的時候要改用 A-D test 或是 K-S test ，不能用　Shapiro-Wilk test
- 最好不要用 K-S test

#### 如果不是常態分布的處理方法
[HOW TO DEAL WITH NON-NORMALITY DATA](https://epicdatastudio.xyz/how-to-deal-with-non-normality-data/)

可能原因 : 
- 離群值或極端值的影響
- 次族群
- 資料鑑別力不足
- 資料收集得不夠
- 觀察值為正值且趨近於零或有自然極限
- 資料為其他分布

方法 : 
1. 改用無母數方法
2. 轉換資料 
   - 移除或取代離群值 : 修剪 (Trimming) 或是縮尾 (Winsorizing)
      - 縮尾是指設定某個不合理的範圍，遇到這個範圍的資料時就認定為離群值，而修剪是直接把這些離群值刪掉 
   - 轉換資料 : 開根號、取對數、冪次轉換 (power transformation )、遞延轉換 (lagged transformation )
      - 最常用的是 Box-Cox power transformations 

[Implementation of Box-Cox power transformations](https://www.geeksforgeeks.org/box-cox-transformation-using-python/)
```python=
import numpy as np
from scipy import stats
  
# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
  
# generate non-normal data (exponential)
original_data = np.random.exponential(size = 1000)
  
# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(original_data)
  
# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
  
# plotting the original data(non-normal) and 
# fitted data (normal)
sns.distplot(original_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, 
            label = "Non-Normal", color ="green", ax = ax[0])
  
sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, 
            label = "Normal", color ="green", ax = ax[1])
  
# adding legends to the subplots
plt.legend(loc = "upper right")
  
# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)
  
print(f"Lambda value used for Transformation: {fitted_lambda}")
```

![](https://i.imgur.com/inPwtzx.png)

### 統計試驗 ( Statistical tests )

[Statistical Tests with Python
](https://python.plainenglish.io/statistical-tests-with-python-880251e9b572)

![](https://i.imgur.com/428Uhcr.jpg)


#### 分為有母數跟無母數方法
- 有母數 (parametric) : t 檢定、ANOVA、Pearson coefficient of correlation
- 無母數 (nonparametric) : 卡方檢定、The Sign Test、The Median Test、The Wilcoxon Rank Sum Test、Mann-Whitney Test、Kruskal-Wallis Test、The Spearman’s Rank Test


#### 無母數方法優缺分析
優點：
- 母群體分布未知或不是常態分布，或是樣本數不夠大時皆可使用。是無母數分析方法的最大優點。
- 計算簡單且快速。
- 雖然在母群實際上為常態分配時，較有母數分析方法不易得到顯著結果；但在母群體不是常態分布時，無母數分析方法之檢力較有母數分析方法高。

缺點：
- 只使用資料的符號、排序等特性，浪費了數值之集中趨勢、分散性及分佈所提供的資訊。
- 針對常態分布資料如果仍進行無母數分析，將使檢力降低。
- 當欲檢定的資料不符合有母數分析法之假設前提時才建議使用無母數分析法，
- 為一種互補的統計方法，而非用於取代有母數分析法。


#### 如何選擇統計試驗
![](https://i.imgur.com/U9SKZLR.png)
- [Overview](https://philipppro.github.io/Statistical_tests_overview/)
![](https://i.imgur.com/SvGFR0k.png)
![](https://i.imgur.com/a5g94AY.png)


#### Hypothesis Testing

[Hypothesis Testing Explained](https://www.kdnuggets.com/2021/09/hypothesis-testing-explained.html)

Type I and Type II error : 
   - 若 $H_0$ 為 True 但卻 Reject $H_0$ 就叫 Type I Error，其發生的機率等於 $\alpha$，也就是 level of significance (顯著水準)
   - 若 $H_a$ 為 True 但卻 Accept $H_0$ 就叫 Type II Error，其發生的機率等於 $\beta$

![](https://i.imgur.com/G5L0Inc.jpg)

只控制 Type I Error 的假設應用稱為 Significance tests (顯著性測試)
- **P-Value** : 假設 Null hypothesis ($H_0$) 是正確的情況下，所觀察到的統計量與目前已經實際觀測過的樣本一樣或是更加極端的機率
  - p-value 又稱 observed level of significance
  - 在某個機率分佈(t分布、z分布或是常態分布之類的)下大於臨界值的機率密度值
  - p-value 越小表示 $H_0$ 越可能是錯誤的，也表示有越多測量證據存在支持 $H_a$ 為真，這是因為觀察到了在 $H_0$ 為真的假設下 ，其相對極端例子的出現而造成的

- **Critical Value** : 測試接受區域的邊界(臨界值)，這個測試通常是指 null hypothesis
  - 概念上就是 level of significance，顯著水準，只是其定義會根據所做的檢定而變，比如在 t test 中就是 t 值，在 z test 中就是 z 值
  - 若轉換為 p-value 的話，其實就是統計檢定犯錯(Type I error)的機率，比如 $\alpha$=0.05，就是只有 5% 的機率這個檢定會犯 Type I error ( 誤將$H0$ reject掉的機率 )，犯錯的原因是隨機誤差
  - critical value of z 是指將 z 分布切為中央區域和尾端區域的那個分割點 ( z 分布就是標準常態分布)

#### z 檢定 (z test) : $\sigma$ Known
特性 :
- 屬於有母數方法
- 適用於 n>30，$\mu$ 未知但 $\sigma$ 已知的情況
- z分布等於標準常態分布 ($\mu$=0，$σ^2$=1之常態分布)

![](https://i.imgur.com/Wh4MnHx.jpg)

種類 : 
- One sample z-test
- Two independent sample z-test
- One sample z-test for proportion
- Two independent sample z-test for proportion
- Paired (Correlated) z-test : Match sample pair

#### t 檢定 (t test) : $\sigma$ Unknown

特性 :
- 屬於有母數方法，又稱 student's t test
- 適用於 n<30, $\sigma$ 未知的情況
- 利用 t 值來推斷出差異發生的機率，用其檢測兩群資料的平均值差異是否顯著
- 以樣本標準差 $s$ 來取代未知的標準差 $\sigma$
- 自由度 = n-1
  - 當自由度越大，t 分布越接近常態分布
  - 通常自由度=30就會視為(近似於)常態分佈(z分布)

![](https://i.imgur.com/rCmeZ90.jpg)

種類 : 
- One sample t-test
- Two independent sample t-test
- Paired (Correlated) t-test : Match sample pair
- Pooled t-test (Equal Variance) : 每個 group 中的數量一樣或是兩個 dataset 的 variance 相近
![](https://i.imgur.com/SNQwtpA.png)
- [t 分布 和 z 分布異同](https://smallcollation.blogspot.com/2013/08/tzsimilarities-and-differences-between.html#gsc.tab=0)
![](https://i.imgur.com/oEbnnSb.jpg)


#### 卡方檢定 (Pearson's Chi-squared test)

特性 : 
- 屬於無母數方法，用來分析兩變數之間的關係
- 只能用在類別型變數，且樣本皆為獨立的情況下
  - 其他的假設為 : 
    - 至少有 80% 的 cell 中其樣本數大於5
    - 每一檢定細格(cell)內的數據應該設為頻率或計數數目，而不是百分比或是經過轉換之數據
- Null hypothesis : 一個樣本中已發生事件的次數分配會遵守某個特定的理論分配
- 在 Null hypothesis 成立時會近似服從卡方分布的檢定 

種類 : 
- 顯著性檢定(Test of significance of change): 檢定一群受試者對事件前後反應的差異
- 適合度檢定(Test of goodness of fit): 檢定資料是否符合某個比例關係或某個機率分佈
- 同質性檢定(Test of homogeneity): 又稱齊一性檢定，檢定幾個不同類別中的比例關係是否一致
- 獨立性檢定(Test of independence): 檢定兩個分類變數之間是否互相獨立

卡方分布(Chi-Squared Distribution)
  - **the chi-squared distribution is a particular case of the gamma distribution**
  - allows you to estimate confidence intervals for a population standard deviation
  - It is the distribution of sample variances when the underlying distribution is normal
  - You can test deviances of differences between expected and observed values
  - You can conduct a chi-squared test
![](https://i.imgur.com/6O57CQs.png)

### 其他機率分布 : Gamma, Geometric, Poisson, Exponential

- Gamma 分布 (伽瑪分布)
  - 為連續型分布 
  - 用來預測某個未來事件發生之前須要等待多久
  - 用來預測某些最小值本來就是0的事件很有用
  - 為卡方分布和指數分布的 generalized 版本分布
![](https://i.imgur.com/KG3Xiln.png)

- Geometric 分布 (幾何分布)
  - 為離散型分布 
  - The geometric distribution represents the probability of having x Bernoulli(p) failures until first success?
    - 也就是說幾何分布回答了 "直到成功之前要失敗多少次?" 這個問題
  - 也能用幾何分布來找到 Bernoulli(1-p) 得到一次成功所需要的試驗次數
  - 若一個事件符合幾何分布，可用來檢查一個事件是否是 i.i.d

- Poisson 分布 (卜瓦松分布)
  - 為離散型分布 
  - 用來描述單位時間內某個隨機事件發生的次數
 ![](https://i.imgur.com/KmIuQ4t.png)

- Exponential 分布 (指數分布)
  - 為連續型分布 
  - 用來表示獨立隨機事件發生的時間間隔 
  - 與卜瓦松分布密不可分
    - 如果卜瓦松分配適合表示某一個區間內事件發生次數的機率，指數分配就可以描述二次事件發生的時間間隔的機率 
    - 比如說，若乘客抵達人數是卜瓦松分布，則乘客之間抵達的時間間隔就會是指數分布 ![](https://i.imgur.com/UTu2S3Y.png)

### 離群值檢測 Outlier Check / Anomaly Detection
- 也稱異常值檢測 (Anomaly dectection)

[Outlier Check for Dataset](https://datatest.readthedocs.io/en/stable/how-to/outliers.html#example-usage) 
[Multi-variate outlier detection in Python](https://towardsdatascience.com/multi-variate-outlier-detection-in-python-e900a338da10)
[5 Anomaly Detection Algorithms every Data Scientist should know](https://towardsdatascience.com/5-anomaly-detection-algorithms-every-


用在
1. 資料前處理
2. 篩選 unlabeled data，找出其中的 outlier
3. 分類 labeded data 的時候，若遇到類別非常不平衡的情況，也可以利用異常值檢測的演算法來做

#### Tukey fence (interquartile Range, IQR, 四分位距)
- 最常用的方法，可順便畫出 Box plot
- 是基於每個值本身的特性去檢測，並沒有考慮到資料之間的 interaction
- $IQR = D3 - D1$
  - $QD = (Q3 - Q1) / 2$, 四分位差
- 能用在 normal 和  slightly skewed distribution
  - 其他偵測 outliner 的方法因為對極端值過於敏感都只能用在 normal

步驟: 
1. 用升序(ascending order)排列資料
   - values = sorted(values)  
2. 找到資料的中位數 
   - midpoint = int(round(len(values) / 2.0))
3. Q1 為第一個資料到中位數這個範圍的中位數 
   - Q1 = median(values[:midpoint])
4. Q3 為中位數到最後一個資料的中位數 
   - Q3 = median(values[midpoint:])
5. 算出 IQR = Q3 - Q1
6. $Lower limit = Q1−(IQR×multiplier)$
7. $Upper limit = Q3+(IQR×multiplier)$
   - multiplier 通常是用 1.5, 若要偵測更遠的則是用 3.0
   - 也有論文說用 2.2 較為通用
8. 若有低於lower limit 或是高於 Upper limit 的就是 Outlier
![](https://i.imgur.com/SB3z5Ax.png)

#### Isolation forest
[ioslation forest by sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)

- 基於資料之間的 interaction 去判斷 outlier
- 常用在連續型、適度的高維資料以及大量資料
  - 遇到過高維度的資料最好先用 Kurtosis 或是其他方法做降維，不然效能會很差
- 先隨機選一個 feature，再從這個 feature 隨機選一個介於最大和最小值之間的 split value，利用它來對資料做二分
  - 其實就是在建構一個隨機型的決策樹
  - 獨立出一個 sample 所需要的 split value 數會正好等於這個決策樹從 Root 走到 leaf node 所需步數 (path length)
  - 離 Root 越近的越可能是 outlier
- 通常會做很多次再抽樣以及會對資料做抽樣，以較少量的資料來訓練即可
  - 對資料抽樣可避免 swamping，也就是正常資料和異常資料過於接近而無法分離，通常在資料過多的時候發生
  - 資料抽樣也可避免 Msaking，也就是當異常值過多且又被分在同一個 cluster 中的時候會很難分開單一個異常值
- 有趣的是也可以做單樣本，也就是只有正常的樣本但沒有異常樣本的訓練情況
  - 另外一個可以做單樣本的是 **One-Class SVM**，One-Class SVM會更適合用在中小型的資料上

例子 : 比如對 a,b,c,d 做 iForest

![](https://i.imgur.com/nmvJ5ju.png)
- 其中的 d 最有可能是異常值

#### Local outlier factor (LOF)
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor)
- 跟iForest一樣適合用在適度高維的資料
- 特點在於它同時考慮了局部和全局的關係來找出 outlier
- 對每一個資料點做評分，這個分數叫做 local outlier factor，其代表這個資料點的異常度 (the degree of abnormality)
  - LOF 是根據資料點本身相對於其鄰居的局部密度(local density)
  - 當資料點的密度比鄰居低越多，則越有可能是 outlier，因為這表示它的鄰居較少，這時候這個資料點的 LOF 會 >> 1
- 實作上這個 Local density 是透過資料點和它鄰居之間的距離來衡量的
  - 其實就是先算出 k-nearest neighbors，然後依據其結果來算出每個點的 LOF
  - outlier 佔資料集的比例越高，k的值就會需要越大

判斷標準
- LOF ~ 1  =>  Similar data point
- LOF < 1  =>  Inlier ( similar data point which is - inside the density cluster)
- LOF > 1  =>  Outlier

![](https://i.imgur.com/GWEO7AJ.jpg)


### Feature Scaling 特徵縮放

[Data Transformation: Standardization vs Normalization](https://www.kdnuggets.com/2020/04/data-transformation-standardization-normalization.html)

- The goal of applying Feature Scaling is to **make sure features are on almost the same scale** so that each feature is equally important and make it easier to process by most ML algorithms
- 有 Standardization (標準化) 和 Min-Max Normalization (歸一化) 兩種方法

需要做 feature scaling 的演算法
![](https://i.imgur.com/SQ7xYz7.png)
- 對於不需要距離的演算法，feature scaling 不重要
  -  像是 Naive Bayes, Linear Discriminant Analysis, and Tree-Based models (gradient boosting, random forest 

#### Standardization (Z-score normalization)
- 中心標準化方法
- 把資料都 scale 成 mean=0, std=1
- 可以更好的解決 Outlier 問題，因此通常會使用這個方法
![](https://i.imgur.com/r3oAg9w.png)

#### Min-Max Normalization 
- 離差標準化方法
- 把資料都 scale 到一個最大和最小值的區間值 (Ex: Scale 到0~1)
- 不能解決 Outlier 問題
![](https://i.imgur.com/DWvzZyc.png)


#### MaxAbsScaler
- 與 MinMaxScaler 類似
- 所有數據都會除以該列絕對值後的最大值
- 數據會縮放到到[-1,1]之間
  - 分母的最大值絕對值的原始資料在轉換後一定會變成-1或是1
- 可以用在 CSR 或是 CSC 稀疏矩陣或是zero-center data
  - 能夠保留稀疏性 
$$x_i = \frac{x_i}{max(\lvert x_i\rvert)}$$

#### Robust scalar
- 捨棄掉中位數並除以 IQR 來縮放
- 可以有效的縮放帶有 outlier 的數據
  -  如果數據中含有異常值，在縮放過程中會被捨去
$$x_{robust} = \frac{x - median(x)}{IQR}$$
$$IQR = Q3 - Q1$$


### Metrics
[Metrics to judge the sucess of a model](http://gael-varoquaux.info/interpreting_ml_tuto/content/01_how_well/01_metrics.html#classification-settings)

[Anomaly Detection — How to Tell Good Performance from Bad](https://towardsdatascience.com/anomaly-detection-how-to-tell-good-performance-from-bad-b57116d71a10)

#### Classification Metrics

Confusion Matrix 混淆矩陣相關
- **Sensitivity, Recall, TPR, 召回率)** = TP / P = TP / (TP+FN)
   - 判斷有病者為陽性之比率
- **Specificity, selectivity, TNR** = TN / N = TN / (TN+FP)
   - 判斷無病者為陰性之比率
- **False Positive Rate (FPR)** = FP / N = FP / (FP+TN) = **1 - Specificity**
   - Type I Error, 判斷無病者為有病之比率
- **Miss rate, False Negative Rate (FNR)** = FN / P = FN / (FN + TP) = **1 - Sensitivity** 
   - Type II Error, 判斷有病者為無病之比率


```python=
# Confusion Matrix in Python with Pandas
import pandas as pd
y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print (df_confusion)

Output:
Predicted  0  1  2
Actual
0          3  0  0
1          0  1  2
2          2  1  3
```



![](https://i.imgur.com/sdrh7zk.png)

有些應用在意的是 Precision，有些應用在意的是 Recall
- 如果是門禁系統，我們希望Precision可以很高，Recall就相較比較不重要，我們比較在意的是預測正向（開門）的答對多少，比較不在意實際正向（是主人）的答對多少
   - Precision 是指分類正確的正樣本占分類器判定為正樣本的樣本個數的比例 
   - Precision 高而 Recall 低的模型是一個非常謹慎的模型
- 如果是廣告投放，則Recall很重要，Precision就顯得沒這麼重要了，因為此時我們比較在意的是實際正向（是潛在客戶）的答對多少，而相對比較不在意預測正向（廣告投出）答對多少。
   - Recall 是指分類正確的正樣本占真正的正樣本個數的比例
   - Recall 高而 Precison 低的模型是一個寬鬆的模型

四個基本指標以外的其他相關 metrics
- **Accuracy, 準確率** = (TP + TN) / (P + N) = (TP + TN) / (TP + TN + FP + FN)
  - 預測正確的比率
- **balanced accuracy** = (TPR + TNR) / 2
  - 用在不平衡資料集
- **Precision, Positive Predictive Value (PPV), 準確率** = TP / (TP+FP)
  - 有病者占陽性之比率
- **Negative Predictive Value (NPV)** = TN / (TN + FN)
  - 無病者占陰性之比率
- **F1-score** = 2 * Precision * Recall / (Precision + Recall)

#### Regression Metrics

MAE, MSE, RMSE, R-squared, MAPE

- MAE 較能處理 outlier，MSE 遇到 outlier 則會指數放大它
- RMSE 是為了避免遇到平方出來會很大的資料而做的處理
- ${\displaystyle {\mbox{MAPE}}={\frac {100}{n}}\sum _{t=1}^{n}\left|{\frac {A_{t}-F_{t}}{A_{t}}}\right|}$
  - $A_t$是實際值
  - $F_t$是預測值
  - 若有資料會等於0不可用
  - 主要是用來比較某兩者，考慮的是相對誤差
  - 比如兩間飲料店賣的飲料數，這樣才可以把真正的實際值也考慮進去，才不會把預測99杯、賣出98杯與預測99999杯、賣出99998杯的飲料店當成一樣
- R-Squared = 1 - SSR / SST
- Adj R-Squared 則是把自變數的數量也考慮進去

Tips

- 選 MAE: 有異常值的情況，如果不想要這些異常值影響模型
- 選 MSE: 有一點異常值的情況，如果想要包含這些異常值
- 如果使用 MAE 作為性能評估指標得到很大偏差，你可能需要使用 RMSE 來撫平其偏差
    - 但若離群值的個別偏差程度非常大，RMSE還是會表現得很差，此時得改用 MAPE
- 如果資料集包含很多異常值，導致預測結果產生偏移，你可能需要用 MAE
- 面對實際值較低的序列，可以將其聚合到一個更大的時間範圍。例如，如果以星期為周期的值很低，你可以試試按照月份來進行預測，甚至按季度預測。你也可以通過簡單的除法，把原始時間序列分解到較小的時間範圍上。這一方法可以幫助你更好地使用 MAE 作為評估指標，同時對峰值做平滑處理。

#### Ranking Metrics

$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$
  - where $R_n$ and $P_n$ are the precision and recall at the $n_{th}$ threshold,
  - MAP is the mean of AP over all the queries

#### Similarity Metrics
- $Cosine(x,y) = \frac{x \cdot y}{|x||y|}$
- $Jaccard(U,V) = \frac{|U \cap V|}{|U \cup V|}$
  - Similarity of two sets $U$ and $V$ .
- $PMI(x;y) = \log{\frac{p(x,y)}{p(x)p(y)}}$
  - Pointwise Mutual Information(PMI)
  - Relevance of two events $x$ and $y$

## Take-home Exercise

[How To Ace The ML Engineer Take-Home Interview Exercise](https://towardsdatascience.com/how-to-ace-the-ml-engineer-take-home-interview-exercise-daf5ba590de4)