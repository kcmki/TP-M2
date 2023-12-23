

from math import sqrt


class Knn:
    def __init__(self,data):
        self.data = data

    def manhattan(self,index=None,data=None):
        if data:
            element = data
        else:
            element = self.data[index]
        distances = {}
        for i in range(len(self.data)):
            somme=0
            for j in range(len(self.data[i]) - 1):
                somme += abs(element[j]-self.data[i][j])
            distances[i]=somme
        return distances
    
    def euclidienne(self,index=None,data=None):
        if data:
            element = data
        else:
            element = self.data[index]

        distances = {}
        for i in range(len(self.data)):
            somme=0
            for j in range(len(self.data[i]) - 1):
                somme += pow(element[j]-self.data[i][j],2)
            distances[i]=sqrt(somme)
        return distances
    
    def minkowski(self,index=None,p=1,data=None):
        if data:
            element = data
        else:
            element = self.data[index]

        distances = {}
        for i in range(len(self.data)):
            somme=0
            for j in range(len(self.data[i])-1):
                somme += pow(abs(element[j]-self.data[i][j]),p)
            distances[i]=pow(somme,1/p)
        return distances 
    
    def cosine(self,index=None,data=None):
        if data:
            element = data
        else:
            element = self.data[index]

        distances = {}
        for i in range(len(self.data)):
            somme=0
            ab = 0
            a2 = 0
            b2 = 0
            for j in range(len(self.data[i])-1):
                ab += self.data[i][j] * element[j]
                a2 += pow(self.data[i][j],2)
                b2 += pow(element[j],2)

            somme = ab/(sqrt(a2)*sqrt(b2))
            distances[i]= 1 - somme
        return distances
        pass
    
    def hamming(self,index=None,data=None):
        if data:
            element = data
        else:
            element = self.data[index]

        distances = {}
        for i in range(len(self.data)):
            somme=0
            for j in range(len(self.data[i]) - 1):
                if element[j] != self.data[i][j]:
                    somme+=1
            distances[i]=somme
    
        return distances

    def getK(self,distances,k):
        return sorted(distances.items(), key=lambda t: t[1])[:k]
    
    def getClass(self,elem,algo="euclidienne",k=3):
        classes = {}

        if algo == "euclidienne":
            distances = self.euclidienne(data=elem)
        if algo == "manhattan":
            distances = self.manhattan(data=elem)
        if algo == "minkowski":
            distances = self.minkowski(data=elem)
        if algo == "cosine":
            distances = self.cosine(data=elem)
        if algo == "hamming":
            distances = self.hamming(data=elem)
        
        distances = self.getK(distances,k)
        
        for (i,dist) in distances:

            if self.data[i][-1] in classes:
                classes[self.data[i][-1]] += 1
            else:
                classes[self.data[i][-1]] = 1
        classes = dict(sorted(classes.items(), key=lambda t: t[1],reverse=True))
        
        return list(classes.keys())[0]