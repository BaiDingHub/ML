#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def norm(datasets,str):
    '''
    归一化数据
    Inputs:
    -datasets:数据集
    -str:要处理的数据特征
    '''
    sample_mean = datasets[str].mean()
    sample_std = datasets[str].std()
    datasets[str] = (datasets[str] - sample_mean)/sample_std
    


# In[ ]:


def getSeveralAtt(datasets,str,num):
    '''
    从数据中提取出ｎｕｍ个属性结点(等深提取)
    Inputs:
    -datasets:数据集
    -str:属性名称
    -num:要提取出来的个数
    
    Outputs:
    -att:从数据及中提取出的属性结点
    '''
    if num == None:
        att = list(set(datasets[str]))
        return att
    att = []
    dataList = list(datasets[str])
    dis = int(len(dataList)/num)
    dataList.sort()
    for i in range(num):
        att.append(dataList[i*dis - 1])
    att = list(set(att))
    return att


# In[ ]:


def transOnehot(dataSet,index):
    length = len(dataSet)
    max = dataSet[index].max().astype(int)
    result = []
    for number in dataSet[index]:
        zero = np.zeros(max + 1)
        zero[int(number)] = 1
        result.append(zero.tolist())
    return np.array(result)

