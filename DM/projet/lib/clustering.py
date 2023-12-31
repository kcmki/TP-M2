import numpy as np
from lib.kmeans import kmeans
from lib.dbscan import dbscan
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st

class Clustering():
    def __init__(self,data=None,seuil=0.8) -> None:
        self.data = data
        pass
    def kmeansClustering(self,K=2,distance="manhattan",itera=100,init="random"):
        self.kmeans = kmeans(self.data,K=K,distance=distance)
        it = self.kmeans.run(iter=itera,init=init)
        return self.kmeans.datas,it
    def dbscanClustering(self,eps=30,minPts=15):
        self.scan = dbscan(np.array(self.data),eps=eps,minPts=minPts)
        self.clst = self.scan.run()
        return self.clst
    
    def resultData(self, clst):
        kmResul = []
        Xkm = []
        for index,row in self.data.iterrows():
            for classe,cluster in clst.items():
                if  any(np.array_equal(np.array(row), arr) for arr in cluster):
                    kmResul.append(classe)
                    Xkm.append(row)
        return kmResul, Xkm
    def percent(self,data):
        clst = {}
        for i,cluster in data.items():
            clst["cluster"+str(i)] = {}
            for item in cluster:
                if item[-1] in clst["cluster"+str(i)].keys():
                    clst["cluster"+str(i)][item[-1]] += 1
                else:
                    clst["cluster"+str(i)][item[-1]] = 1
        clst = pd.DataFrame(clst)
        clst.replace(np.nan,0,inplace=True)
        percent = []
        for col in clst.columns:
            percent.append(max(clst[col])/sum(clst[col]))
        return percent,clst
    
    def drawScatter(self,clst):
        result,X = self.resultData(clst)
        real = np.array(X)[:,-1]
        X = np.array(X)[:,:-1]
        print(len(result),len(real))

        for i in range(len(X[0])):
            for j in range(i+1,len(X[0])):
                # Create a figure with two subplots side by side
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                # Scatter plot 1
                axes[0].scatter(X[:, i], X[:, j],c=real, cmap="viridis", label=np.unique(real))
                axes[0].set_title('Real classification')
                axes[0].set_xlabel(self.data.columns[i])
                axes[0].set_ylabel(self.data.columns[j])

                # Scatter plot 2
                axes[1].scatter(X[:, i], X[:, j],c=result, cmap="viridis", label=np.unique(result))
                axes[1].set_title('Clustering classification')
                axes[1].set_xlabel(self.data.columns[i])
                axes[1].set_ylabel(self.data.columns[j])

                # Adjust layout for better spacing
                plt.tight_layout()

                # Show the plots
                plt.show()
                st.pyplot()
                plt.close()

    def silhouette(self,clst):
        result,X = self.resultData(clst)
        real = np.array(X)[:,-1]
        X = np.array(X)[:,:-1]
        from sklearn.metrics import silhouette_score
        return silhouette_score(X, result)
    def calinski(self,clst):
        result,X = self.resultData(clst)
        real = np.array(X)[:,-1]
        X = np.array(X)[:,:-1]
        from sklearn.metrics import calinski_harabasz_score
        return calinski_harabasz_score(X, result)
    def davies(self,clst):
        result,X = self.resultData(clst)
        real = np.array(X)[:,-1]
        X = np.array(X)[:,:-1]
        from sklearn.metrics import davies_bouldin_score
        return davies_bouldin_score(X, result)