#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from loadDatas import *
import collections
from optim import * 


# In[ ]:


class TreeNode(object):
    '''
    决策树的节点信息
    '''
    def __init__(self,name = None,spiltValue = None,parentNode = None,feat = 'InNode'):
        '''
        -self.name:节点的名字
        -self.spiltValue:节点的边界数值
        -self.parentNode:节点的父节点
        -self.leftNode:节点的左子节点
        -self.rightNode:节点的右子节点
        -self.eva:当前节点的评价值
        -self.feat:当前节点的属性，'InNode':内部节点，'LeafNode':叶节点
        -self.index:当前节点的预测值
        '''
        self.name = name
        self.spiltValue = spiltValue
        self.parentNode = parentNode
        self.leftNode = None
        self.rightNode = None
        self.eva = 0
        self.feat = feat
        self.index = None
        self.values = []


# In[ ]:


class DecisionTree(object):
    '''
    决策树
    '''
    def __init__(self,category = 'ID3',preCut = False,numBorder = 5,EBorder = 0.05):
        '''
        -self.tree:决策树
        -self.category:决策树的模型类别,'ID3','C4.5','CART'
        -self.preCut:是否进行预剪枝
        -self.numBorder:预剪枝的数目边界，默认5
        -self.EBorder:预剪枝的信息熵边界，默认0.05
        '''
        self.tree = TreeNode()
        self.category = category
        self.preCut = preCut
        self.numBorder = numBorder
        self.EBorder = EBorder
    
    def bulidTree(self,x,y,atts,parentNode = None):
        '''
        模型训练,
        Inputs:
        -x:训练数据
        -y:训练标签
        -atts:每一个属性上的分类
        -parentNode:父节点
        
        Outputs:
        -treeNode:一个节点
        '''
        #创建子节点
        treeNode = TreeNode()

        #预剪枝
        if(self.preCut):
            #如果数据小于数据边界
            if(y.shape[0] <= self.numBorder):
                treeNode.spiltValue = None
                treeNode.parentNode = parentNode
                treeNode.feat = 'LeafNode'
                treeNode.index = max(collections.Counter(y),key = collections.Counter(y).get)
                return treeNode
        #如果该节点无训练数据，返回
        if(y.all() == None):
            return None
        #将第一个节点与模型的self.tree链接
        if parentNode == None:
            parentNode = self.tree
            self.tree.leftNode = treeNode
            self.tree.rightNode = treeNode
        #如果节点的标签都是一样的，即都为一个分类
        if (list(set(y)) == 1 ):
            treeNode.spiltValue = None
            treeNode.parentNode = parentNode
            treeNode.feat = 'LeafNode'
            treeNode.index = max(collections.Counter(y),key = collections.Counter(y).get)
            return treeNode
        #得到当前节点最好的分割属性和分割值
        bestName,bestValue,bestEva = self.chooseBestNode(x,y,atts)
        #预剪枝
        if(self.preCut):
            #如果评价值小于评价值边界
            if(bestEva < self.EBorder):
                treeNode.spiltValue = None
                treeNode.parentNode = parentNode
                treeNode.feat = 'LeafNode'
                treeNode.index = max(collections.Counter(y),key = collections.Counter(y).get)
                return treeNode
        treeNode.name = bestName
        treeNode.spiltValue = bestValue
        treeNode.eva = bestEva
        treeNode.parentNode = parentNode
        treeNode.index = max(collections.Counter(y),key = collections.Counter(y).get)
        #如果分割节点失败，将该节点设置为叶节点
        if(bestValue == None):
            treeNode.feat = 'LeafNode'
            return treeNode
        #分割数据
        X_left,Y_left,X_right,Y_right = self.spiltDatas(x,y,bestName,bestValue)
        #添加该节点的左右节点
        treeNode.leftNode = self.bulidTree(X_left,Y_left,atts,treeNode)
        treeNode.rightNode = self.bulidTree(X_right,Y_right,atts,treeNode)
        return treeNode
    
    def chooseBestNode(self,x,y,atts):
        '''
        选择当前属性的最好节点
        Inputs:
        -x:当前节点的训练数据
        -y:当前节点的训练标签
        -atts:当前节点可供选择的所有属性 list
        
        Outputs:
        -bestName:最好的节点列数
        -bestValue:最好的划分值
        -bestEva:得到的最好的评价
        '''
        bestName = None
        bestValue = None
        bestEva = -np.inf
        for col,att in enumerate(atts):
            for value in att:
                E = self.computeEva(x,y,col,value)
                if E> bestEva:
                    bestEva = E
                    bestName = col
                    bestValue = value
        return bestName,bestValue,bestEva
    
    def computeEva(self,x,y,col,value):
        '''
        计算评价值
        -x:该节点的数据
        -y:该节点的标签
        -col:属性的列数
        -value:要划分的值
        '''
        nums = x.shape[0]
        E = self.computEntropy(y)
        X_left,Y_left,X_right,Y_right = self.spiltDatas(x,y,col,value)
        num1 = len(X_left)
        p1 = num1/nums
        E1 = self.computEntropy(Y_left)
        num2 = len(X_right)
        p2 = num2/nums
        E2 = self.computEntropy(Y_right)
        
        if self.category == 'ID3':
            result = E - (p1 * E1 + p2 * E2)
        elif self.category == 'C4.5':
            result = E - (p1 * E1 + p2 * E2)
            result /= -(p1 * np.log(p1) + p2 * np.log(p2))
        elif self.category == 'CART':
            result = -(p1 * E1 + p2 * E2)
        return result
            
    def computEntropy(self,y):
        '''
        计算信息熵
        '''
        num = y.shape[0]
        num1 = np.sum(y)
        num2 = num - num1
        p1 = num1/num
        p2 = num2/num
        if self.category == ('ID3' or 'C4.5'):
            E = -(p1 * np.log(p1) + p2 * np.log(p2))
        elif self.category == 'CART':
            E = 1 - (p1 **2 + p2 **2)
        return E
    
    def spiltDatas(self,x,y,col,value):
        '''
        将数据分割
        Inputs:
        -x:待分割数据
        -y:带分割数据
        -col：要分割的属性位置
        -value:分割边界
        '''
        X_left =x[x[:,col] <= value]
        Y_left = y[x[:,col] <= value]
        X_right =x[x[:,col] > value]
        Y_right = y[x[:,col] > value]
        return X_left,Y_left,X_right,Y_right
    
    def predict(self,X,Y = None):
        pre = []
        for i,x in enumerate(X):
            if Y.all()!= None:
                y = Y[i]
            t = self.tree.leftNode
            while(t.feat != 'LeafNode'):
                t.values.append(y)
                name = t.name
                value = t.spiltValue
                if(value == None):
                    t = t.parentNode
                    break
                if(x[name] <= value):
                    t = t.leftNode
                else:
                    t = t.rightNode
            pre.append(t.index)
        pre = np.array(pre).reshape((len(pre),))
        if(y.all() == None):
            return pre
        score = np.sum(pre == Y)
        score /= Y.shape[0]
        return pre,score
    
    def afterCut(self,X_test,Y_test,tree = None):
        if(tree == None):
            tree = self.tree.leftNode
            self.predict(X_test,Y_test)
        if(tree.feat == 'LeafNode'):
            return
        self.afterCut(X_test,Y_test,tree.leftNode)
        self.afterCut(X_test,Y_test,tree.rightNode)
        #判断是否需要剪枝
        acc = np.sum(tree.values == tree.index)/len(tree.values)
        acc1 = (np.sum(tree.leftNode.values == tree.leftNode.index) + 
               np.sum(tree.rightNode.values == tree.rightNode.index))/len(tree.values)
        if(acc >= acc1):
            tree.leftNode = None
            tree.rightNode = None
            tree.feat = 'LeafNode'
        if(len(tree.values) <= self.numBorder):
            tree.leftNode = None
            tree.rightNode = None
            tree.feat = 'LeafNode'


# In[4]:


class Linear(object):
    '''
    线性分类器
    '''
    def __init__(self):
        '''
        -self.w : 初始化权重，(D,H)
        -self.b: 初始化bias,(H,)
        '''
        self.W = None
        self.b = None
    def train(self,X,y,out_dims,
              lr = 1e-5,reg = 1e-2,
              batch_size = 32,epoch = 5,weight_scale = 1e-5,printFreq = 20,
              grad_function = sgd,activation_function = 'relu'):
        '''
        Inputs:
        -X:训练数据 (N,D)
        -y:数据标签,(H,)
        -lr:学习率
        -reg:正则化参数
        -batch_size:每次迭代的数目
        -epoch:对全部数据迭代的次数
        -weight_scale:对Ｗ初始化的权重
        -printFreq:经过几个batch输出一次loss和accuracy
        
        Outputs:
        -loss_history:list,所有的loss
        '''
        N,D = X.shape
        H = out_dims
        self.W = weight_scale * np.random.randn(D,H)
        self.b = np.zeros((H,))
        loss_history  = []
        acc_history = []
        config = {'lr':lr}
        
        ## 设置激活函数
        if activation_function == 'relu':
            self.activation_forward = relu_forward
            self.activation_backward = relu_backward
        if activation_function == 'sigmoid':
            self.activation_forward = sigmoid_forward
            self.activation_backward = sigmoid_backward
        if activation_function == 'tanh':
            self.activation_forward = tanh_forward
            self.activation_backward = tanh_backward
        
        iter_nums = int(N/batch_size) * epoch
        for i in range(iter_nums):
            #随机获得batch_size个数据
            index = np.random.randint(0,N,batch_size)
            xx = X[index]
            yy = y[index]
            #正向传播
            z = xx.dot(self.W) + self.b
            #激活函数
            out,cache = self.activation_forward(z)
            #得到准确率
            acc = np.sum(out.argmax(axis = 1) == yy)/yy.shape[0]
            acc_history.append(acc)
            #得到loss
            loss,dout = self.loss(out,yy)
            loss_history.append(loss)
            #激活函数反向传播
            dx = self.activation_backward(dout,cache)
            #得到梯度
            dw = xx.T.dot(dx)
            db = dout.sum(axis = 0)
            #梯度下降
            self.W = grad_function(self.W,dw,config)
            self.b = grad_function(self.b,db,config)
            
            if (i+1) % printFreq == 0:
                print("epoch ",int(i/(iter_nums/epoch)),'|',epoch,'\t','acc = ',acc,'\tloss = ',loss)
        return loss_history,acc_history
            
    def loss(self,out,y):
        '''
        依靠具体的模型决定
        '''
        pass
    
    def predict(self,X,y = None):
        z = X.dot(self.W) + self.b
        out,_ = self.activation_forward(z)
        out = out.argmax(axis = 1)
        if y.all() == None:
            return out,_
        acc = np.sum(out == y)/y.shape[0]
        return out,acc


# In[ ]:


class SVM(Linear):
    def loss(self,out,y):
        loss,dx = svm_loss(out,y)
        return loss,dx


# In[ ]:


class Logistic(Linear):
    def loss(self,out,y):
        loss,dx = softmax_loss(out,y)
        return loss,dx

