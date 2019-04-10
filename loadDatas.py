#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProcess import *


# In[22]:


def load_Treedatas(filename= 'diabetes.csv',nums = [None,10,None,None,10,30,50,None],trainFrac = 0.75):
    datasets = pd.read_csv(filename)
    datasets.sample(frac=1)   #打乱数据
    sum = len(datasets)
    train_num = int(sum * trainFrac)
    X_train = datasets.values[:train_num,:-1]
    Y_train = datasets.values[:train_num,-1]
    X_test = datasets.values[train_num:,:-1]
    Y_test = datasets.values[train_num:,-1]
    dataset = datasets[:train_num]
    
    length = len(dataset.columns.values)
    atts = []
    for (i,arg) in enumerate(zip(nums,dataset.columns.values)):
        num,str = arg
        att = getSeveralAtt(dataset,str,num)
        atts.append(att)
    return X_train,Y_train,X_test,Y_test,atts


# In[1]:


def load_Titanicdatas(dataSet,change_survived):
    """处理数据"""
    #将名字中的Mr,Mrs,Miss抽出来
    dataSet['Title'] = dataSet['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
    dataSet.loc[~(dataSet['Title'].isin(['Master.','Mrs.','Miss.','Master.'])),'Title_Val'] = 4
    dataSet.loc[(dataSet['Title'].isin(['Mr.'])),'Title_Val'] = 0
    dataSet.loc[(dataSet['Title'].isin(['Mrs.'])),'Title_Val'] = 1
    dataSet.loc[(dataSet['Title'].isin(['Miss.'])),'Title_Val'] = 2
    dataSet.loc[(dataSet['Title'].isin(['Master.'])),'Title_Val'] = 3
    #转化成onehot类型   
    title_val = transOnehot(dataSet,'Title_Val')
    dataSet['Title_Val_0']  = title_val[:,0]
    dataSet['Title_Val_1']  = title_val[:,1]
    dataSet['Title_Val_2']  = title_val[:,2]
    dataSet['Title_Val_3']  = title_val[:,3]
    dataSet['Title_Val_4']  = title_val[:,4]
    
    #将性别转化成离散值Sex_Val{'male':0,'female':1}
    sexes = dataSet['Sex'].unique()
    genders_mapping_one = dict(zip(sexes, range(0, len(sexes) + 1)))
    dataSet['Sex_Val'] = dataSet['Sex'].map(genders_mapping_one).astype(int)
    
    #将SibSp,Parch 转化成一个离散值Family
    dataSet['FamilySize'] = dataSet['SibSp'] + dataSet['Parch']
    #将Fare归一化
    dataSet.loc[(dataSet['Fare'].isnull()),'Fare'] = dataSet['Fare'].median()
    maxFare = dataSet['Fare'].max()
    minFare = dataSet['Fare'].min()
    dit = maxFare - minFare
    dataSet['FareNormal'] = dataSet['Fare']/dit
    
    #将Cabin的有无当成一个特征Cabin_Val{'unexist':0,'exist':1}
    dataSet.loc[ (dataSet.Cabin.notnull()), 'Cabin_Val' ] = 1
    dataSet.loc[ (dataSet.Cabin.isnull()), 'Cabin_Val' ] = 0
    
     #将Embarked转化成数字Embarked_Val={'S':0,'C':1,'Q':2}
    embarked_locs = dataSet['Embarked'].unique()
    embarked_locs_mapping = dict(zip(embarked_locs,range(0, len(embarked_locs) + 1)))
    dataSet['Embarked_Val'] = dataSet['Embarked'].map(embarked_locs_mapping).astype(int)
    #用众数替换目的地的缺失值
    if len(dataSet[dataSet['Embarked'].isnull()] > 0):
        dataSet.replace({'Embarked_Val' : 
                   { embarked_locs_mapping[np.nan] : embarked_locs_mapping['S']}
               },inplace=True)
    #转化成onehot类型    
    embarked_val = transOnehot(dataSet,'Embarked_Val')
    dataSet['Embarked_Val_0']  = embarked_val[:,0]
    dataSet['Embarked_Val_1']  = embarked_val[:,1]
    dataSet['Embarked_Val_2']  = embarked_val[:,2]
    
    
    #用众数(考虑性别和等级）替换缺失的Age值
    dataSet['AgeFill'] = dataSet['Age']
    dataSet['AgeFill'] = dataSet['AgeFill']                         .groupby([dataSet['Sex_Val'], dataSet['Pclass']])                         .apply(lambda x: x.fillna(x.median()))
    dataSet['AgeFill'] = dataSet['AgeFill'] - dataSet['AgeFill'].min()
    dit = dataSet['AgeFill'].max() - dataSet['AgeFill'].min()
    dataSet['AgeFill'] = dataSet['AgeFill']/dit
    
    #剔除用不到的特征
    dataSet = dataSet.drop(['Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Cabin','Title','PassengerId','Title_Val','Embarked_Val'], axis=1)
#     dataSet = dataSet.drop(['Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Cabin','PassengerId'], axis=1)
    if change_survived:
        dataSet['Survived_End'] = dataSet['Survived']
        dataSet = dataSet.drop(['Survived'], axis=1)
        
#     dataSet_modify = dataSet.values
    #对数据进行归一化
    dataSet_modify = dataSet.values
    return dataSet_modify


# In[ ]:


def load_Lineardatas(trainFrac = 0.75):
    '''
    加载Titanic的数据，用于线性模型
    '''
    df_train = pd.read_csv("train.csv")
    trainSet= load_Titanicdatas(df_train,change_survived=True)    #shape = (?,9)
    num = trainSet.shape[0]
    num1 = int(num * trainFrac)
    X_train = trainSet[:num1,:-1]
    Y_train = trainSet[:num1,-1]
    X_test = trainSet[num1:,:-1]
    Y_test = trainSet[num1:,-1]
    return X_train,Y_train,X_test,Y_test

