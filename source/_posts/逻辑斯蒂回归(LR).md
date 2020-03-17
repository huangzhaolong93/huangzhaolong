---
title: 逻辑斯蒂回归-LR
date: 2018-11-07 16:05:05
tags: [机器学习, 分类]
mathjax: true
---
# 理论基础
&emsp;&emsp;逻辑斯蒂回归是机器学习里的一种分类模型。本节将介绍逻辑斯蒂回归相关的理论基础知识。

## 相关知识
&emsp;&emsp;最大似然的思想：假如有一个罐子，罐子里有黑白两种颜色的球，球的数目不知，球的颜色比例也不知。这时候我们从罐子中有放回的拿出10个球，即拿出1个球，记录颜色，再放回罐中摇匀，此操作重复10次。假设记录结果显示拿出了7个白球，3个黑球。请问罐中黑白球比例最可能是多少？大部分人都会毫不犹豫的给出答案，$黑:白=3:7$。这背后运用到理论支持其实就是最大似然的思想。  
&emsp;&emsp;似然，指的就是使这种情况发生。最大似然就是使这种情况最大概率发生。接下来，我们来看看最大似然思想是如何得到这样的结果。
&emsp;&emsp;假设抽到黑球的概率是$P$,则抽到白球的概率就是$1-P$。此时抽了10次，抽到了7个白球3个黑球，这种情况的概率就是$P(3黑7白)=P^3(1-P)^7$。为了是这种情况最大概率发生，我们可以得到$$arg\,\max_{P} P^3(1-P)^7$$令导数为0，得到公式$$3P^2(1-P)^7+(-7)P^3(1-P)^6=0$$解得P=0.3.故抽到黑球的比例是0.3，黑球和白球的比例就是$3:7$。

## 模型特点
* 逻辑斯蒂回归的模型能力：**分类**  
&emsp;给定样本x的特征$ X=\lbrace x_1,x_2,...,x_n\rbrace $,判断样本的标签$Y$。  
* 逻辑斯蒂回归的学习方法：**有监督学习**  
&emsp;需要有训练样本对模型进行训练。
* 逻辑斯蒂回归的模型类型：**判别式模型**  
&emsp;通过训练的特征$X$和类别$Y$，直接计算$P(Y|X)$作为模型，直接根据样本的特征$X$计算类别$Y$。

## 模型推导
&emsp;&emsp;最大似然的思想可以总结为"透过现象看本质"。根据上面黑白球的例子，现象就是抽了10个球，3黑7白，本质就是抽到黑球的概率是0.3。而类比在逻辑斯蒂回归中，现象就是训练样本的<font color=#FF0000>特征和标签</font>，本质就是逻辑斯蒂回归(LR)模型的<font color=#FF0000>模型参数</font>。我们的任务就是寻找一组LR模型的参数，使得训练样本的特征和标签出现的概率最大。
&emsp;&emsp;假设我们有如下m个样本，每个样本有n维特征：
$$\lbrace x_1^{(1)},x_2^{(1)},...,x_n^{(1)},y^{(1)}\rbrace,\lbrace x_1^{(2)},x_2^{(2)},...,x_n^{(2)},y^{(2)}\rbrace,...,\lbrace x_1^{(m)},x_2^{(m)},...,x_n^{(m)},y^{(m)}\rbrace$$&emsp;&emsp;标签$y\in \lbrace0,1\rbrace$。LR模型的模型参数为$\theta$,根据最大似然的思想，我们认为这m个样本是已经发生的现实情况，故将问题转化为$$arg\,\max_{\theta}\prod_{x_i\in X}{P(Y=0|x_i,\theta)}^{(1-y_i)}{P(Y=1|x_i,\theta)}^{y_i}$$&emsp;&emsp;其中，$P(Y|X,\theta)$就是LR模型。
&emsp;&emsp;我们可以看到，逻辑斯蒂回归模型其实是一个分类模型，并不是回归模型。那为什么会有这么一个名字呢？笔者认为逻辑斯蒂回归可以拆解为<font color=#FF0000>逻辑斯蒂(logistic)+线性回归</font>。线性回归模型是一个回归模型，模型可以表示为$$y=\theta_1x_1+\theta_2x_2+...+\theta_nx_n=\mathrm{\theta}^\mathrm{T}\mathrm{x}$$&emsp;&emsp;模型的取值为$-\infty\sim+\infty$。那如何转化为分类问题呢，就需要一个函数映射，将$-\infty\sim+\infty$映射成$0\sim1$(概率表示)且具有连续、光滑、严格单调和关于中点中心对称。而logistic函数(sigmoid)恰好满足所有的需求。logistic函数为$y=\frac{1}{1+e^{-x}}$。绘制函数图像的代码如下：

```
import matplotlib.pyplot as plt     
import numpy as np 
import math
'''
def sigmoid(x):
    return (1/(1+math.exp(-x)))
#如果用上面的sigmoid，会导致分母溢出而报错，推荐用下面的sigmoid自定义函数
print(sigmoid(-10000000))
'''
def sigmoid(x):
    return math.exp(-np.lnaddexp(0, -x))
    
x = np.arange(-10., 10., 0.2)
y = [ sigmoid(x_i) for x_i in x]
plt.plot(x, y)     
plt.show() 
```
![Alt text](http://pjnwg6wvx.bkt.clouddn.com/sigmoid.png)

&emsp;&emsp;综合logistic函数和线性回归模型函数，我们可以得到逻辑斯蒂回归模型的模型函数$$y=\frac{1}{1+e^{-\mathrm{\theta}^T\mathrm{x}}}$$&emsp;&emsp;我们令$h(\theta,x)$表示LR模型的模型函数，即$h(\theta,x)=\frac{1}{1+e^{-\theta^\mathrm{T}x}}$。如果$h(\theta,x)=0.8$，我们可以认为该样本有$0.8$的概率是正样本，$0.2$的概率是负样本，即$P(y=1|x,\theta) = h(\theta,x)$和$P(y=0|x,\theta) = 1-h(\theta,x)$。将LR的模型函数代回到最大似然的求解问题中，得到$$arg\,\max_{\theta}\prod_{x_i\in X}(1-h(\theta,x))^{(1-y_i)}h(\theta,x)^{y_i}$$&emsp;&emsp;由于是累乘，我们使用单调的算子$ln(\cdot)$将累乘变成累加，且不影响最终结果。得到$$arg\,\max_{\theta}\sum_{x_i\in X}(1-y_i)\ln (1-h(\theta,x))+y_i\ln h(\theta,x)$$&emsp;&emsp;我们令$$L(\theta)=\sum_{x_i\in X}(1-y_i)\ln (1-h(\theta,x))+y_i\ln h(\theta,x)$$&emsp;&emsp;若要求解上述的$argmax$，则需要$L^\prime(\theta)=0$且$L^{\prime\prime}(\theta)<0$。根据LR模型的函数，有
$$\begin{eqnarray}L(\theta)
		&=&\sum_{x_i\in X}(1-y_i)\ln (1-\frac{1}{1+e^{-\mathrm{\theta}^T\mathrm{x}}})+y_i\ln \frac{1}{1+e^{-\mathrm{\theta}^T\mathrm{x}}}\\
		&=&\sum_{x_i\in X}y_i(\mathrm{\theta}^T\mathrm{x})-\ln(1+e^{\mathrm{\theta}^T\mathrm{x}})
	\end{eqnarray}$$&emsp;&emsp;由于$L(\theta)$是多变量的，故采用梯度上升法进行求解其最大值。如果要求$L(\theta)$的最大值，也就是等价于求$-L(\theta)$的最小值，此时$-L(\theta)$可以理解成逻辑斯蒂回归模型的损失函数。最终问题等价于用梯度下降法求解$-L(\theta)$的最小值。梯度下降法的参数更新过程为：$$\theta_j := \theta_j - \alpha \frac{\partial L(\theta)}{\partial \theta_j} ,(j=0,...,n)$$&emsp;&emsp;其中$\alpha$为学习率。那么$L(\theta)$关于$\theta_j$的偏导数为$$\begin{eqnarray}\frac{\partial L(\theta)}{\partial \theta_j}
	&=&\alpha\left(\sum_{x_i\in X}{y_ix_{ij}-x_{ij}\frac{e^{\mathrm{\theta}^T}}{1+e^{\mathrm{\theta}^T\mathrm{x}}}}\right)\\
	&=&\alpha\sum_{x_i\in X}x_{ij}(y_i-h(\theta,x_i))\\
	&=&\alpha\sum_{x_i\in X}x_{ij}\Delta y_i\end{eqnarray}$$&emsp;&emsp;上述的公式表达着这样一个道理：第$j$维特征上的模型参数，它的更新方向为学习率(即学习步长)$\alpha$乘上每个样本$x_i$在第$j$维度特征上真实值$y_j$和模型预测值$h(\theta,x_i)$的差。<br /><font color=#0099ff size=1>_简单来说，就是参数更新的方向是通过模型误差乘上学习步长进行调整的。_</font>

## 进阶优化
特征离散化，将连续的特征离散化成一系列的0和1。
&emsp;&emsp;1、特征容易增加或减少，特征解释性强，例如年龄这维特征$X_{age}$，比如第一版模型中，特征是$X_{age<18}$，这维特征可以理解成是否成年。后续第二版是可以直接增加$X_{18<age<25}$的特征，可以理解为是否为青年人。
&emsp;&emsp;2、特征$X$变成0或1，$y=sigmoid(\theta^TX)$中$\theta^TX)$将变成稀疏向量，其内积乘法运算速度快，计算结果方便存储，容易扩展。
&emsp;&emsp;3、特征离散化后，具有极强的鲁棒性。例如年龄这维特征$X_{age}$，特征是$X_{age<18}$。如果年龄数据中出现录入错误的，比如录入了身高数据175，在$X_{age<18}$这个特征中也仅仅是1，对结果影响没有那么大。但是如果采用的是连续特征，那么$\theta_{age}X_{age}$会因为$X_{age}=175$变得很大。导致结果偏差很大。
&emsp;&emsp;4、LR模型本质上还是线性模型，本质上仍然是$\theta^TX$，如果不离散化，年龄特征$X_{age}$只会有一个系数$\theta_{age}$，如果离散化，年龄特征可能会变成$X_{age<18}$,$X_{age<30}$,$X_{30<age<40}$等等一系列离散特征，这样每个特征前的参数都不一样，相当于对于年龄这位特征是分段线性的，提升模型的表达能力。
&emsp;&emsp;5、特征离散化后，有利于进行特征交叉。如果不做离散化，可能就只有$X_{age}$和$X_{sex}$的交叉，离散化之后，就可以是$X_{age<18}X_{sex=F}$表示女性未成年，$X_{18<age<25}X_{sex=M}$表示男性青年。特征A离散化为$M$个值，特征B离散为$N$个值，那么交叉之后会有$M\times N$个变量，进一步引入非线性，提升表达能力；
&emsp;&emsp;6、特征离散化后，模型更加稳定。比如特征重要度最高的是年龄特征。离散化后为$X_{30<age<40}$，那么年龄在$30-40$岁值都是一样的，不会因为年龄大了一岁，就使得最后的结果发生较大变化。
&emsp;&emsp;7、特征离散化后，通过将特征分散，起到弱化特征的作用，避免某一维度特征过大，从而导致过拟合。比如年龄特征$X_{age}$的特征重要度$\theta_{age}$很大，那么模型会过分依赖这维特征。一旦$X_{age}$出现了微小的波动，则都会对模型结果产生巨大的影响。
# 工程实现
代码中LR_classify为LR分类器模型，具体代码如下。
## 代码
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:52:02 2018

@author: huangzhaolong
"""

from sklearn.datasets import load_iris
import numpy as np
import random
import math

def load_data():
    
    iris = load_iris()
    X = iris.data
    Y = iris.target
    #特征离散化,多分类转二分类
    X = X[Y<=1].round()
    Y = Y[Y<=1]
    data_num = X.shape[0]
    index_list = [random.random() >= 0.2 for _ in range(data_num)] 
    train_index_list = []
    test_index_list = []
    for i in range(len(index_list)):
        if index_list[i]:
            train_index_list.append(i)
        else:
            test_index_list.append(i)
            
    X_train = X[train_index_list]
    X_test = X[test_index_list]
    Y_train = Y[train_index_list]
    Y_test = Y[test_index_list]
    
    return X_train, X_test, Y_train, Y_test

class LR_classify(object):
    
    def sigmoid(self, num_list):
        if isinstance(num_list, (int, float)):
            return 1/(1+math.exp(-num_list))
        else:
            result_list = []
            for num in num_list:
                if isinstance(num, (int, float)):
                    result_list.append(1/(1+math.exp(-num)))
                else:
                    print ("error type!")
                    return -1
            return result_list
    
    def LR_train(self, X_train, Y_train, train_step = 1000):
        #训练步数
        train_step = 10
        #初始化权重
        weights=[random.random() for _ in range(X_train.shape[1]+1)] 
        #训练集扩展为广义矩阵
        X_train = np.hstack((X_train, np.ones(X_train.shape[0]).reshape(X_train.shape[0],1)))
        #定义梯度下降的step
        learning_rate = 0.001
        #对于每个样本
        sample_num = X_train.shape[0]
        
        for step in range(train_step):
            #预测
            pred = self.sigmoid(np.dot(X_train,weights))
            for i in range(sample_num):
                error = Y_train[i]- pred[i]
                weights = weights + learning_rate * error * X_train[i]
            if step % 1 == 0:
                acc = cal_accuracy([round(z) for z in pred], Y_train)
                print("训练过程第",step,"步的准确率为：",acc)

        self.weights = weights
        
    def LR_test(self, X_test):
        X_test = np.hstack((X_test, np.ones(X_test.shape[0]).reshape(X_test.shape[0],1)))
        weights = self.weights
        pred = self.sigmoid(np.dot(X_test,weights))

        return pred
        
def cal_accuracy(pred, label):
    
    acc = list(pred-label).count(0)/len(pred-label)
    
    return acc
        
if __name__=="__main__":
    acc = 0
    
    for i in range(1):
        X_train, X_test, Y_train, Y_test = load_data()
        
        lr = LR_classify()
        
        lr.LR_train(X_train, Y_train)
        
        pred = lr.LR_test(X_test)
        
        acc += cal_accuracy([round(z) for z in pred],Y_test)
        
    acc = acc/1
    
    print("测试集准确率:",acc)
```

## 结果
```
训练过程第 0 步的准确率为： 0.46153846153846156
训练过程第 1 步的准确率为： 0.46153846153846156
训练过程第 2 步的准确率为： 0.46153846153846156
训练过程第 3 步的准确率为： 0.6153846153846154
训练过程第 4 步的准确率为： 0.8717948717948718
训练过程第 5 步的准确率为： 0.8717948717948718
训练过程第 6 步的准确率为： 0.9871794871794872
训练过程第 7 步的准确率为： 0.9871794871794872
训练过程第 8 步的准确率为： 1.0
训练过程第 9 步的准确率为： 1.0
测试集准确率: 1.0
```
	