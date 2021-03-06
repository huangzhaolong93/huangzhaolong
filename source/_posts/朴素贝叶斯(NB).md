---
title: 朴素贝叶斯-NB
date: 2018-11-06 23:09:55
tags: [机器学习, 分类, 贝叶斯]
mathjax: true
---
# 理论基础
&emsp;&emsp;朴素贝叶斯是机器学习里的一种分类模型。本节将介绍朴素贝叶斯相关的理论基础知识。

## 相关知识
&emsp;&emsp;贝叶斯学派的思想：  
$$先验概率+数据分布=后验概率$$  &emsp;&emsp;在这里举个例子帮助大家有个简单的理解：投掷一枚硬币100次，20次正面朝上，求这枚硬币正面朝上的概率$P(正)$。
* 频率学派：$P(正)=\frac{20}{100}=0.2$
* 贝叶斯派：    
	* 先验概率：这枚硬币是从银行拿出来的，我认为它正面朝上的概率是0.5，并假设之前它已经被投掷1000次：500次正，500次负。
	* 数据分布：本次的投掷结果：20次正，80次负。
	* 后验概率：$P(正)=\frac{500+20}{1000+100}=0.473$  

&emsp;&emsp;贝叶斯派被频率学派所诟病的就是所谓的先验概率。一般来说先验概率就是我们对于数据所在领域的历史经验，但是这个经验常常难以量化或者模型化，于是贝叶斯派大胆的假设先验分布的模型，比如正态分布，beta分布等。这个假设一般没有特定的依据，因此一直被频率学派认为很荒谬。
## 模型特点
* 朴素贝叶斯的模型能力：**分类**  
&emsp;给定样本x的特征$ X=\lbrace x_1,x_2,...,x_n\rbrace $,判断样本的标签$Y$。  
* 朴素贝叶斯的学习方法：**有监督学习**  
&emsp;需要有训练样本对模型进行训练。
* 朴素贝叶斯的模型类型：**生成式模型**  
&emsp;通过训练样本计算特征$X$和标签$Y$的联合分布$P(X,Y)$。联合分布$P(X,Y)$可以看做是朴素贝叶斯的模型，里面的每个概率都是模型的参数。
* 朴素贝叶斯的特点：**朴素**  
&emsp;各维特征对标签的影响相互独立,$P(Y|X)=P(Y|x_1)P(Y|x_2)...P(Y|x_n)$

## 数学基础  
&emsp;&emsp;先看条件独立公式，如果$X$和$Y$相互独立，$$P(X,Y)=P(X)P(Y)$$&emsp;&emsp;根据条件概率公式：
$$P(X,Y)=P(X|Y)P(Y) = P(Y|X)P(X)$$&emsp;&emsp;可以推出贝叶斯公式：
$$P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$$&emsp;&emsp;其中，$P(Y)$表示类别标签的先验分布，$P(X|Y)$表示在已知类别标签$Y$的情况下，特征$X$取值的条件概率。$P(X)$表示特征的分布，所有类别下$P(X)$均是相同的。
&emsp;&emsp;对于二分类问题，通过比较$P(Y=0|X)$和$P(Y=1|X)$,就可以实现分类。

## 模型推导
&emsp;&emsp;假设我们有如下m个样本，每个样本有n维特征：
$$\lbrace x_1^{(1)},x_2^{(1)},...,x_n^{(1)},y^{(1)}\rbrace,\lbrace x_1^{(2)},x_2^{(2)},...,x_n^{(2)},y^{(2)}\rbrace,...,\lbrace x_1^{(m)},x_2^{(m)},...,x_n^{(m)},y^{(m)}\rbrace$$&emsp;&emsp;训练过程：根据训练样本建立联合分布$P(X,Y)$。
&emsp;&emsp;第一步：通过训练样本，可以计算训练样本中标签的分布$P(Y)$，通过最大似然估计，我们可以认为样本中类别标签的分布就是类别标签的先验分布$P(Y)$。<font color=#0099ff size=1>_最大似然估计：通俗来说就是已知样本分布的情况，得到能够导致这种情况(似然)最大概率发生的参数。_</font>假设所有标签为$Y=0$的训练样本个数共$m_0$个，所有标签为$Y=1$的训练样本个数共$m_1$个。则$P(Y=0)=\frac{m_0}{m_0+m_1}$。
&emsp;&emsp;第二步：针对每种标签下的样本，统计每一维特征$x$的分布，得到条件分布$P(x|Y)$。例如针对所有标签为$Y=0$的训练样本，统计第$i$维特征$x_i$的取值，假设有两种取值0和1，取值为0共$i_0$个,则$i_1=m_0-i_0$。则条件概率$P(x_i=0|Y=0)=\frac{i_0}{m_0}$。
&emsp;&emsp;第三步：根据标签的先验分布$P(Y)$和每维特征基于标签的条件分布$P(x|Y)$，建立联合分布$P(X,Y)$。例如$P(x_i=0,Y=0)=P(x_i=0|Y=0)P(Y=0)=\frac{i_0}{m_0}\times\frac{m_0}{m_0+m_1}=\frac{i_0}{m_0+m_1}$。<font color=#0099ff size=1>_注：这里联合分布$P(X,Y)$就可以看成是模型训练的参数。_</font>
&emsp;&emsp;预测过程：根据联合分布$P(X,Y)$和特征$X$,判断$P(Y=0|X)$与$P(Y=1|X)$的大小。
&emsp;&emsp;第一步：根据预测样本的每维特征$x_i$和联合分布$P(X,Y)$，都可以计算$P(Y=0|{x_i})$和$P(Y=1|{x_i})$。
&emsp;&emsp;第二步：根据特征对标签的影响相互独立，我们有：$$P(Y|X)=P(Y|{x_1}{x_2}...{x_n})=P(Y|{x_1})P(Y|{x_2})...P(Y|{x_n})$$&emsp;&emsp;第三步：比较$P(Y=1|X)$和$P(Y=0|X)$，确定标签。


# 实例讲解
&emsp;&emsp;接下来，我会有一个简单的例子，帮助大家充分理解上述讲的过程。  
&emsp;&emsp;假设我们拿到了10个男生的信息，并让一位女生根据这些信息进行判断是否考虑嫁给他们（下述随机举例，不代表真实情况）。

| 特征 | 高 | 富 | 帅 | 温柔 |女生嫁不嫁 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 男生1 | 高 | 富 | 帅 | 温柔| 嫁 |
| 男生2 | 高 | 穷 | 搓 | 温柔 | 不嫁 |
| 男生3 | 矮 | 穷 | 帅 | 不温柔 | 不嫁 |
| 男生4 | 矮 | 穷 | 搓 | 温柔 | 不嫁 |
| 男生5 | 矮 | 富 | 帅 | 不温柔 | 不嫁 |
| 男生6 | 高 | 穷 | 帅 | 温柔 | 嫁 |
| 男生7 | 矮 | 穷 | 帅 | 温柔 | 不嫁 |
| 男生8 | 矮 | 富 | 搓 | 温柔 | 嫁 |
| 男生9 | 高 | 穷 | 搓 | 不温柔 | 不嫁 |
| 男生10 | 高 | 富 | 帅 | 不温柔 | 嫁 |

现在，有一个男生x（样本），他高、富、搓、温柔（特征），请问这个女生是否会选择嫁给他（分类）。这就是个典型的分类问题。
根据贝叶斯公式：$$P（嫁|高、富、搓、温柔） = \frac{P(嫁)P(高、富、搓、温柔|嫁)}{P(高、富、搓、温柔)}$$

第一步：我们需要考虑女生嫁人的主观意愿，也就是标签的先验分布$P(嫁)$。  
&emsp;&emsp;就是无论对方男生条件怎么样，这个女生自己到底有多想嫁人。根据上述的10个样本，我们可以知道，这个女生想嫁人的概率$P(嫁)=\frac{4}{10}$，这个女生不想嫁人的概率$P(不嫁)=\frac{6}{10}$。  <br /><font color=#0099ff size=1>_这里有一点需要说明:先验分布我们是不知道的，也不是能算的(女生的心思你不懂)。但是现实的结果是10个男生里，女生想嫁4个人。所以我们可以假定在某种先验分布的情况下，让这种现实结果出现的概率最大。所以就得到了先验分布$P(嫁)=\frac{4}{10}$。_</font>  
&emsp;&emsp;我们想象一个极端的例子，如果10个男生里，女生的选择全是不嫁，我们可以推断出这个女生可能在当前阶段是个独身主义者，完全不考虑嫁给任何人（即使样本不能代表任何人,比如王校长），即$P(嫁)=0$。即使样本是王校长,朴素贝叶斯模型也会认为女生选择不嫁。

第二步：我们知道了女生嫁人的主观意愿，我们还需要知道，如果女生选择嫁的给男生x，他的高、富、搓对嫁这个结果分别有多大的影响，也就是条件概率$P(高、富、搓、温柔|嫁)$。
&emsp;&emsp;接下来就要用到朴素贝叶斯的“朴素”。朴素贝叶斯认为，男生的高、富、搓（特征）对嫁（标签）这个结果的影响是相互独立的，也就是$P(高、富、搓、温柔|嫁)=P(高|嫁)P(富|嫁)P(搓|嫁)P(温柔|嫁)$。  <br /><font color=#0099ff size=1>_这里有一点需要说明:为什么要假设特征对标签的影响是相互独立的呢？因为如果不独立，特征空间就是1个三维空间，特征总共2\*2\*2=8，如果独立呢，特征空间就是3个一维空间，2+2+2=6。一个累乘，一个累加。在现实生活中，往往有非常多的特征，每维特征的取值也是非常之多，而“朴素”正是以牺牲准确率为代价来大幅降低计算强度。_</font>


我们选择重新整理下样本，只看选择嫁的这4个男生

| 特征 | 高 | 富 | 帅 | 温柔 |女生嫁不嫁 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 男生1 | 高 | 富 | 帅 | 温柔| 嫁 |
| 男生6 | 高 | 穷 | 帅 | 温柔 | 嫁 |
| 男生8 | 矮 | 富 | 搓 | 温柔 | 嫁 |
| 男生10 | 高 | 富 | 帅 | 不温柔 | 嫁 |

$P(高|嫁)$的计算方式：在所有选择嫁的男生里，高的男生占几个。可以得到$P(高|嫁)=\frac{3}{4}$。同理可以得到$P(富|嫁)=\frac{3}{4}$，$P(搓|嫁)=\frac{1}{4}$，$P(温柔|嫁)=\frac{3}{4}$。

对于同一个样本而言，$P(X)$都是一样的，故只需要比较
贝叶斯公式的分子$P(Y=0)P(X|Y=0)$和$P(Y=1)P(X|Y=1)$即可。
我们可以计算得到$$P(嫁)P(高、富、搓、温柔|嫁)=\frac{4}{10}\times\frac{3}{4}\times\frac{3}{4}\times\frac{1}{4}\times\frac{3}{4}=\frac{9}{640}$$

$$P(不嫁)P(高、富、搓、温柔|不嫁)=\frac{6}{10}\times\frac{2}{6}\times\frac{1}{6}\times\frac{1}{6}\times\frac{3}{6}=\frac{1}{120}=\frac{9}{1080}$$


可以得到：$P(嫁|高、富、搓)>P(不嫁|高、富、搓)$  
朴素贝叶斯模型就可以告诉这个女生，对于男生x：<font color=#FF0000>嫁</font> ！

## 进阶优化
由于朴素贝叶斯模型里假设特征相互独立，当计算联合概率时，就可能用到各维特征空间中概率的累乘。累乘存在一个致命的缺点：<font color=#FF0000>一旦其中一项为0，整项就为0</font>。  
当特征很多时，非常容易出现稀疏情况，$P(x_i|Y)=0$，导致$P(X={x_1,x_2,...,x_i,...,x_n}|Y)=0$，直接导致最后的结果$P(Y|X)=0$。显然不够合理。为了避免这种情况发生，可以采取<font color=#FF0000>平滑策略</font>。  
最常见的平滑策略就是拉普拉斯平滑，即在统计计算概率时，默认某类样本中，在各维空间中各个取值至少出现了一次。如果训练样本集数量充分大时，并不会对结果产生影响，并且解决了上述概率为0的尴尬局面。

# 优缺点
## 优点
* 思路简单
* 空间复杂度低（联合概率$P(X,Y)$只需要二维存储）
* 时间复杂度低（$P(X),P(Y),P(X|Y)$可以从联合分布$P(X,Y)$中直接计算）

## 缺点
* 假设了各维对结果的影响相互独立，现实中不成立。（比如富就会对帅有影响，有钱更容易打扮变帅。）当特征之间影响较大，朴素贝叶斯的性能就不太好。

# 工程实现
代码中NB_classify为朴素贝叶斯模型，具体代码如下。
## 代码
```
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:52:02 2018

@author: huangzhaolong
"""
#利用sklearn自带的鸢尾花数据集
from sklearn.datasets import load_iris
import random

def load_data():
    #加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    Y = iris.target
    #特征离散化
    X = X.round()
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

class NB_classify(object):
    
    def NB_train(self, X_train, Y_train):
        '''
        在已知X和Y的情况下，构建联合分布P(X,Y),即参数计算，计算方式为统计
        根据贝叶斯公式，计算P(X,Y)=P(X|Y)*P(Y)
        '''
        p_xy = []
        #对于每个类别
        p_y = {}
        for k in set(Y_train):
            total_xi_y = list(Y_train).count(k)
            print("对于类别为",k,"的样本共有",total_xi_y,"个")
            #计算P(Y)
            p_y.setdefault(k,list(Y_train).count(k)/Y_train.shape[0])
            p_x_y_sub = []
            #对每个维度的特征，计算P(x|Y)
            for i in range(X_train.shape[1]):
                #对于每一维特征,统计当前类别下样本个数,得到P(x|Y)
                p_x_sub_y_sub = {}
                print("对于类别为",k,"的样本，第",i,"维的离散特征共有",len(set([(round(item)) for item in X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]])),"种")
                #对每个维度下特征的取值，计算p(x=xi|Y)
                for j in set(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]):
                    #统计每个类别下，每个特征占该类别样本个数的比例,并做贝叶斯平滑
                    print("对于类别为",k,"的样本，第",i,"维特征为",j,"的样本共有",list(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]).count(j), "个")
                    #计算p(y)*p(x|y)
                    p_x_sub_y_sub.setdefault(j,p_y[k]*(list(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]).count(j)+1)/(total_xi_y+len(set(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]))))
                p_x_y_sub.append(p_x_sub_y_sub)
            '''
            最后得到P(X|Y)的模型参数，即矩阵M(xy)，其中矩阵M为m行n列,m为类别个数，n为特征维数。每个元素是一个字典，字典的key是该维特征的取值，字典的value是概率
            例如矩阵M中的元素M_ij，M_ij是一个字典。
            M_ij的key是第j维特征的所有取值，例如k_j
            M_ij的value是第j维特征取值为k_j时，类别为i的概率P(xj=k_j|i)，也就是所有类别为i的样本下，第j维特征为k_j的概率v_ij。
            v_ij = 所有类别为i的样本下，第j维特征为k_j的样本数数/所有类别为i的样本数
            '''
            p_xy.append(p_x_y_sub) 
        print("P(Y)的先验概率：\n",p_y)
        print("P(X,Y)的联合概率：\n",p_xy)
        self.p_xy = p_xy
        
    def NB_test(self, X_test):
        p_xy = self.p_xy
        pred = []
        #对于每个样本
        for z in range(len(X_test)): 
            #对于每个类别
            pj = [1]*3
            for j in range(3):
                #对于每个特征,通过联合概率分布，计算每个特征下得到的类别P(yj|xi)
                for i in range(len(X_test[z])):
                    if X_test[z][i] in p_xy[j][i]:
                        p_xi_yj = p_xy[j][i][X_test[z][i]]
                    else:
                        #如果训练集没有出现过该特征，则说明构建的P(X,Y)联合分布并不准确，生成式模型需要训练样本足够大，足够全
                        p_xi_yj = 0.0000001
                    #计算全部特征到每个类别的概率P(yj|X)，这里是认为各特征对类别是独立的，可以连乘
                    pj[j] = pj[j]*p_xi_yj
            #根据概率最大计算最可能的类别，即 argmax P(yj|X)
            pred.append(pj.index(max(pj)))
        
        return pred
        
def cal_accuracy(pred, Y_test):
    acc = list(pred-Y_test).count(0)/len(pred-Y_test)
        
    return acc
    
if __name__=="__main__":
    acc = 0
    
    for i in range(1):
        X_train, X_test, Y_train, Y_test = load_data()
        
        nb = NB_classify()
        nb.NB_train(X_train, Y_train)
        pred = nb.NB_test(X_test)
        acc += cal_accuracy(pred,Y_test)
        
    acc = acc/1
    
    print("acc:",acc)
        
```

## 结果
```
对于类别为 0 的样本共有 42 个
对于类别为 0 的样本，第 0 维的离散特征共有 3 种
对于类别为 0 的样本，第 0 维特征为 4.0 的样本共有 4 个
对于类别为 0 的样本，第 0 维特征为 5.0 的样本共有 34 个
对于类别为 0 的样本，第 0 维特征为 6.0 的样本共有 4 个
对于类别为 0 的样本，第 1 维的离散特征共有 2 种
对于类别为 0 的样本，第 1 维特征为 3.0 的样本共有 25 个
对于类别为 0 的样本，第 1 维特征为 4.0 的样本共有 17 个
对于类别为 0 的样本，第 2 维的离散特征共有 2 种
对于类别为 0 的样本，第 2 维特征为 1.0 的样本共有 17 个
对于类别为 0 的样本，第 2 维特征为 2.0 的样本共有 25 个
对于类别为 0 的样本，第 3 维的离散特征共有 2 种
对于类别为 0 的样本，第 3 维特征为 0.0 的样本共有 41 个
对于类别为 0 的样本，第 3 维特征为 1.0 的样本共有 1 个
对于类别为 1 的样本共有 40 个
对于类别为 1 的样本，第 0 维的离散特征共有 3 种
对于类别为 1 的样本，第 0 维特征为 5.0 的样本共有 5 个
对于类别为 1 的样本，第 0 维特征为 6.0 的样本共有 27 个
对于类别为 1 的样本，第 0 维特征为 7.0 的样本共有 8 个
对于类别为 1 的样本，第 1 维的离散特征共有 2 种
对于类别为 1 的样本，第 1 维特征为 2.0 的样本共有 12 个
对于类别为 1 的样本，第 1 维特征为 3.0 的样本共有 28 个
对于类别为 1 的样本，第 2 维的离散特征共有 3 种
对于类别为 1 的样本，第 2 维特征为 3.0 的样本共有 3 个
对于类别为 1 的样本，第 2 维特征为 4.0 的样本共有 24 个
对于类别为 1 的样本，第 2 维特征为 5.0 的样本共有 13 个
对于类别为 1 的样本，第 3 维的离散特征共有 2 种
对于类别为 1 的样本，第 3 维特征为 1.0 的样本共有 29 个
对于类别为 1 的样本，第 3 维特征为 2.0 的样本共有 11 个
对于类别为 2 的样本共有 41 个
对于类别为 2 的样本，第 0 维的离散特征共有 4 种
对于类别为 2 的样本，第 0 维特征为 8.0 的样本共有 5 个
对于类别为 2 的样本，第 0 维特征为 5.0 的样本共有 1 个
对于类别为 2 的样本，第 0 维特征为 6.0 的样本共有 23 个
对于类别为 2 的样本，第 0 维特征为 7.0 的样本共有 12 个
对于类别为 2 的样本，第 1 维的离散特征共有 3 种
对于类别为 2 的样本，第 1 维特征为 2.0 的样本共有 4 个
对于类别为 2 的样本，第 1 维特征为 3.0 的样本共有 34 个
对于类别为 2 的样本，第 1 维特征为 4.0 的样本共有 3 个
对于类别为 2 的样本，第 2 维的离散特征共有 4 种
对于类别为 2 的样本，第 2 维特征为 4.0 的样本共有 1 个
对于类别为 2 的样本，第 2 维特征为 5.0 的样本共有 17 个
对于类别为 2 的样本，第 2 维特征为 6.0 的样本共有 20 个
对于类别为 2 的样本，第 2 维特征为 7.0 的样本共有 3 个
对于类别为 2 的样本，第 3 维的离散特征共有 2 种
对于类别为 2 的样本，第 3 维特征为 1.0 的样本共有 1 个
对于类别为 2 的样本，第 3 维特征为 2.0 的样本共有 40 个
P(Y)的先验概率：
 {0: 0.34146341463414637, 1: 0.3252032520325203, 2: 0.3333333333333333}
P(X,Y)的联合概率：
 [[{4.0: 0.03794037940379404, 5.0: 0.2655826558265583, 6.0: 0.03794037940379404}, {3.0: 0.2017738359201774, 4.0: 0.13968957871396895}, {1.0: 0.13968957871396895, 2.0: 0.2017738359201774}, {0.0: 0.3259423503325942, 1.0: 0.015521064301552107}], [{5.0: 0.04537719795802609, 6.0: 0.21176025713745508, 7.0: 0.06806579693703914}, {2.0: 0.10065814943863724, 3.0: 0.22454510259388305}, {3.0: 0.030251465305350726, 4.0: 0.189071658158442, 5.0: 0.10588012856872754}, {1.0: 0.23228803716608595, 2.0: 0.09291521486643438}], [{8.0: 0.044444444444444446, 5.0: 0.014814814814814814, 6.0: 0.17777777777777778, 7.0: 0.0962962962962963}, {2.0: 0.03787878787878787, 3.0: 0.26515151515151514, 4.0: 0.0303030303030303}, {4.0: 0.014814814814814814, 5.0: 0.13333333333333333, 6.0: 0.15555555555555556, 7.0: 0.029629629629629627}, {1.0: 0.015503875968992248, 2.0: 0.3178294573643411}]]
acc: 0.9629629629629629
```
