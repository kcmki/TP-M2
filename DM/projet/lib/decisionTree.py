
import numpy as np

class Node:
    def __init__(self,threshold=None,featureIndex=None,left=None,right=None,infoGain=None,value=None):
        #attributs pour les noeuds
        self.threshold = threshold
        self.featureIndex = featureIndex
        self.left = left
        self.right = right
        self.infoGain = infoGain

        #class correspondante pour les feuilles
        self.value = value


class decisionTree:
    def __init__(self,maxDepth=10,minSamplesSplit=2):

        self.root = None

        #condition d'arret
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit

    def buildTree(self,data,curr_depth=0):
        X,y = data[:,:-1],data[:,-1]
        numSamples,numFeatures = X.shape

        #construction de l'arbre
        if numSamples >= self.minSamplesSplit and curr_depth <= self.maxDepth:

            #trouver le meilleur split
            bestSplit = self.BestSplit(data,numSamples,numFeatures)
            #verifier si l'information gain est positive
            try:
                if bestSplit["infoGain"] > 0:
                    #construire les sous arbres
                    leftSubTree = self.buildTree(bestSplit["data_left"],curr_depth+1)
                    rightSubTree = self.buildTree(bestSplit["data_right"],curr_depth+1)
                    return Node(threshold=bestSplit["threshold"],
                                featureIndex=bestSplit["featureIndex"],
                                left=leftSubTree,
                                right=rightSubTree,
                                infoGain=bestSplit["infoGain"])
            except:
                pass

        leafValue = self.calculateLeafValue(y)
        return Node(value=leafValue)

    def calculateLeafValue(self,y):
        #calculer la classe majoritaire
        y = list(y)
        return max(y,key=y.count)

    def BestSplit(self,data,numFeatures):
        #initialiser les variables
        bestSplit = {}
        maxInfoGain = -float("inf")
        #parcourir les features
        for feature in range(numFeatures):
            featureValues = data[:,feature]
            possibleThresholds = np.unique(featureValues)
            #parcourir les seuils
            for threshold in possibleThresholds:
                data_left,data_right = self.split(data,feature,threshold)
                if len(data_left) > 0 and len(data_right) > 0:
                    y,y_left,y_right = data[:,-1],data_left[:,-1],data_right[:,-1]
                    #calculer l'information gain
                    currInfoGain = self.infoGain(y,y_left,y_right,"gini")
                    #mettre a jour le meilleur split
                    if currInfoGain > maxInfoGain:
                        bestSplit["featureIndex"] = feature
                        bestSplit["threshold"] = threshold
                        bestSplit["data_left"] = data_left
                        bestSplit["data_right"] = data_right
                        bestSplit["infoGain"] = currInfoGain
                        maxInfoGain = currInfoGain
        return bestSplit
    def split(self,data,feature,threshold):

        #separer le dataset en deux
        data_left = np.array([row for row in data if row[feature] <= threshold])
        data_right = np.array([row for row in data if row[feature] > threshold])
        return data_left,data_right
    
    def infoGain(self,parent,left,right):
        #calculer l'information gain
        weight_p = len(parent)
        weight_l,weight_r = len(left),len(right)
        p = self.gini(parent)
        l = self.gini(left)
        r = self.gini(right)
        return p - (weight_l/weight_p*l + weight_r/weight_p*r)
    
    def gini(self,y):
        #calculer l'indice de gini
        classes = np.unique(y)
        gini = 0
        for i in classes:
            p_i = len(y[y==i])/len(y)
            gini += p_i**2
        return 1-gini
    
    def fit(self,X,y):
        data = np.concatenate((X,y),axis=1)
        self.root = self.buildTree(data)
    
    def predict(self,X):
        #predire les classes
        preditions = [self.traverseTree(self.root,x) for x in X]
        return preditions
    
    def traverseTree(self,node,x):
        if node.value != None:
            return node.value
        featureValue = x[node.featureIndex]
        if featureValue <= node.threshold:
            return self.traverseTree(node.left,x)
        else:
            return self.traverseTree(node.right,x)
        
    def printTree(self):
        #afficher l'arbre
        self.printNode(self.root)

    def printNode(self,node,indent=""):
        #afficher un noeud
        if node.value != None:
            print(indent+"Classe",str(node.value))
        else:
            print(indent+str(node.featureIndex))
            print(indent+"Seuil",str(node.threshold))
            self.printNode(node.left,indent+"  ")
            self.printNode(node.right,indent+"  ")
