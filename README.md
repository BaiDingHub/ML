# ML
机器学习算法

## 数据集
------
一个关于糖尿病患者的数据集   diabets.csv
一个Kaggle中Titanic数据集　train.csv,test.csv

## 模型
### 1.决策树
        根据糖尿病的数据集获取训练集和测试集
        实现了预剪枝和后剪枝
    
### 2.线性模型
      SVM
      Logistic
      
### 3.随机森林
        使用决策树基学习器
        
### 4.KNN
        实现了k近邻算法
      
## 各文件夹描述
----

    dataProcess.py  
       用于处理数据集的函数
       包括：归一化，one-hot化，抽取特征边界
      
    loadDatas.py
      用于加载数据
      包含加载糖尿病数据集和Titanic数据的函数
      
    optim.py
      各种优化函数
      包括：各种loss函数
           各种激活函数
           各种梯度下降的优化方法
           normalization
           dropout
 
    Model.py
      包含各种模型
      有:决策树模型
         线性模型
   
    ModelTest.ipynb
       对模型的测试

