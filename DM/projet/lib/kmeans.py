from lib.knn import Knn
import random
import pandas as pd

class kmeans:
    def __init__(self,pdata,K=2,distance="manhattan"):
        self.K = K
        self.pdata = pdata
        self.centroids = []
        self.datas = {}
        
        self.distance = distance
        for i in range(self.K):
            self.datas[i] = []

    def init_centroids(self,init="random"):
        if init == "random":
            for i in range(self.K):
                randm = random.randint(0,self.pdata.shape[0]-1)
                self.centroids.append(list(self.pdata.iloc[randm]))
        else:
            X = self.pdata.shape[0]//self.K
            for i in range(self.K):
                self.centroids.append(list(self.pdata.iloc[(i*X)+(X//2)]))

    def init_data(self):
        for i in range(self.pdata.shape[0]):
            row = list(self.pdata.iloc[i])
            self.datas[self.getCluster(row)].append(row)

    def centroid(self,data):
        return data.mean()
    
    def updateCentroid(self):
        self.centroids = []
        for _,data in self.datas.items():
            df = pd.DataFrame(data)
            center = list(self.centroid(df))
            center[-1] = round(center[-1])
            self.centroids.append(center)

    def addToDatas(self,item):
        self.datas[self.getCluster(item)].append(item)

    def getCluster(self,item):
        self.Knn = Knn(self.centroids)
        distances = self.Knn.getDistance(item,algo=self.distance)
        distances = sorted(distances.items(), key=lambda t: t[1])
        return distances[0][0] 
    
    def updateDatas(self):
        for key in self.datas.keys():
            for elem in self.datas[key]:
                newCluster = self.getCluster(elem)
                if newCluster != key:
                    self.datas[key].remove(elem)
                    self.datas[newCluster].append(elem)
    
    def intraClusterDistance(self):
        intra = 0
        for key in self.datas.keys():
            for elem in self.datas[key]:
                intra += self.getDistance(elem,self.centroids[key],algo=self.distance)**2
        return intra
    
    def interClusterDistance(self):
        inter = 0
        centroids = self.centroids
        for i in range(len(centroids)):
            for j in range(i,len(centroids)):
                inter += self.getDistance(self.centroids[i],self.centroids[j],algo=self.distance)**2
        return inter
    
    def run(self,iter=100,init="random"):
        self.init_centroids(init=init)
        self.init_data()

        for _ in range(iter):
            old = self.centroids.copy()
            
            self.updateCentroid()
            if self.centroids == old:
                print("breaked at ",_," iteration")
                return _
            self.updateDatas()
    

    def getDistance(self,a,b,algo="manhattan"):
        if algo == "euclidienne":
            return self.euclidienne(a,b)
        if algo == "manhattan":
            return self.manhattan(a,b)
        if algo == "minkowski":
            return self.minkowski(a,b)
        if algo == "cosine":
            return self.cosine(a,b)
        if algo == "hamming":
            return self.hamming(a,b)
    def euclidienne(self,a,b):
        return sum([(a[i]-b[i])**2 for i in range(len(a))])**0.5
    def manhattan(self,a,b):
        return sum([abs(a[i]-b[i]) for i in range(len(a))])
    def minkowski(self,a,b):
        return sum([abs(a[i]-b[i])**1 for i in range(len(a))])**(1/1)
    def cosine(self,a,b):
        ab = 0
        a2 = 0
        b2 = 0
        for i in range(len(a)):
            ab += a[i] * b[i]
            a2 += a[i]**2
            b2 += b[i]**2
        return 1 - (ab/(a2**0.5 * b2**0.5))
    def hamming(self,a,b):
        return sum([1 if a[i] != b[i] else 0 for i in range(len(a))])