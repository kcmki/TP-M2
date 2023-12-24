
import pandas as pd
import numpy as np
from math import sqrt
from lib.decisionTree import decisionTree
from lib.knn import Knn

class Classification:
    #marge is the percentage of the train data between 0 and 1
    def __init__(self, data,marge,n_tree=100,maxDepth=10,minSamplesSplit=2):
        self.train=data.sample(frac=marge)
        self.test=data.drop(self.train.index)
        self.trees = None
        self.knn = None
        self.randomTrees = None

    def testKnn(self,data=None,distance="euclidienne",k=2):
        
        results = []
        
        self.knn = Knn(np.array(self.train)) 
        if data is None:
            data = self.test

        for i in range(len(data)):
            elem = list(data.iloc[i])[:-1]
            classe = self.knn.getClass(elem,algo=distance,k=k)
            results.append(classe)

        return results
    
    def trainDecisionTree(self,maxDepth=10,minSamplesSplit=2):
        
        self.tree = decisionTree(maxDepth=maxDepth,minSamplesSplit=minSamplesSplit)
        trainX,trainY = np.array(self.train)[:,:-1],np.array(self.train)[:,-1].reshape(-1,1)
        self.tree.fit(trainX,trainY)
        
        return self.tree
    def testDecisionTree(self,data=None):
        if data is None:
            data = self.test
        if self.tree is None:
            print("You need to train the decision tree first")
            return None
        testX,testY = np.array(data)[:,:-1],np.array(data)[:,-1].reshape(-1,1)
        pred = self.tree.predict(testX)
        pred = np.array(pred).reshape(-1,1)

        return pred
    
    def trainRandomForest(self,n_tree=100,maxDepth=10,minSamplesSplit=2):
        self.randomTrees = []

        nb_cols = int(sqrt(len(self.train.columns)-1))

        for i in range(n_tree):
            random_subset = self.train.sample(n=len(self.train), replace=True)
            cols = []
            while len(cols) < nb_cols:
                col = np.random.randint(0,len(self.train.columns)-1)
                if col not in cols:
                    cols.append(col)
            cols.append(len(self.train.columns)-1)
            traincols = self.train.iloc[:,cols]
            traincols = traincols.loc[random_subset.index]
            trainX,trainY = np.array(traincols)[:,:-1],np.array(traincols)[:,-1].reshape(-1,1)
            tree = decisionTree(maxDepth=maxDepth,minSamplesSplit=minSamplesSplit)
            tree.fit(trainX,trainY)
            self.randomTrees.append((tree,cols))
        
        return self.randomTrees

    def testRandomForest(self,data=None):
        if data is None:
            data = self.test

        if self.randomTrees is None:
            print("You need to train the random forest first")
            return None
        
        results = []
        for i in range(len(data)):
            elem = list(data.iloc[i])[:-1]
            elementClasses = []
            for tree,cols in self.randomTrees:
                row = [elem[i] for i in cols[:-1]]
                prediction = tree.predict([row])[0]
                elementClasses.append(prediction)
            classe = max(set(elementClasses), key=elementClasses.count)
            results.append(classe)

        return results

    def confMatrix(self,results):
        self.matrix = np.zeros((len(self.test["Fertility"].unique()),len(self.test["Fertility"].unique())))
        for i in range(len(self.test)):
            self.matrix[int(self.test.iloc[i]["Fertility"])][int(results[i])]+=1
        return self.matrix

    def getMetrics(self,confMatrix):
        metrics = {}
        for i in range(len(confMatrix)):
            TP = confMatrix[i][i]
            FP = sum(confMatrix[:,i])-TP
            FN = sum(confMatrix[i,:])-TP
            TN = sum(sum(confMatrix))-TP-FP-FN
            metrics[i] = {}
            metrics[i]["precision"] = TP/(TP+FP)
            metrics[i]["recall"] = TP/(TP+FN)
            metrics[i]["accuracy"] = (TP+TN)/(TP+FP+FN+TN)
            metrics[i]["f1"] = 2*(metrics[i]["precision"]*metrics[i]["recall"])/(metrics[i]["precision"]+metrics[i]["recall"])
        return metrics
