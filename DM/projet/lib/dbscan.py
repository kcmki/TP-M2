
import numpy as np

class dbscan:
    def __init__(self,data,eps=5,minPts=5):
        self.data = data
        self.eps = eps
        self.minPts = minPts
    def getDistance(self,elem1,elem2):
        somme=0
        for j in range(len(elem1)-1):
            somme += abs(elem1[j]-elem2[j])
        return somme
    def getNeighbors(self,pointIndex):
        neighbors = []
        for i in range(len(self.data)):
            distance = self.getDistance(self.data[pointIndex],self.data[i])
            if distance <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def getCorePoints(self):
        corePoints = []
        for i in range(len(self.data)):
            neighbors = self.getNeighbors(i)
            if len(neighbors) >= self.minPts:
                corePoints.append(i)
        return corePoints
    
    def clusterCorePoints(self,corePoints):
        clusters = {}
        clusterIndex = 0
        clusters[clusterIndex] = []
        queue = []
        visited = []
        while len(corePoints) >0:

            if len(queue) ==0:
                pointIndex = np.random.choice(corePoints)
            else:
                pointIndex = np.random.choice(queue)
                queue.remove(pointIndex)

            visited.append(pointIndex)
            pointNeighbors = self.getNeighbors(pointIndex)
            for newNeighbor in pointNeighbors:
                if newNeighbor not in queue and newNeighbor not in visited and newNeighbor in corePoints:
                    queue.append(newNeighbor)

            corePoints.remove(pointIndex)
            clustered = False
            # check if point has neighbors in an existing cluster
            for clusterNumber in clusters.keys():
                for neighbor in pointNeighbors:
                    if neighbor in clusters[clusterNumber]:
                        clusters[clusterNumber].append(pointIndex)
                        clustered = True
                        break
                if clustered:
                    break
            # create a cluster for this point
            if not clustered and len(pointNeighbors) > self.minPts:
                clusters[clusterIndex] = [pointIndex]
                clusterIndex += 1

        return clusters

    def fixData(self,clusterCore):
        clusters = {}
        for index in clusterCore.keys():
            for item in clusterCore[index]:
                if index in clusters.keys():
                    clusters[index].append(self.data[item])
                else:
                    clusters[index] = [self.data[item]]
        return clusters
    def fillClusters(self,clusterCore,outerPoints):
        for item in outerPoints:
            clustered = False
            for index in clusterCore.keys():
                for core in clusterCore[index]:
                    distance = self.getDistance(self.data[item],self.data[core])
                    if distance <= self.eps:
                        clusterCore[index].append(item)
                        clustered = True
                        outerPoints.remove(item)
                        break
                if clustered:
                    break
        return clusterCore

    def run(self):
        #cherche les points centraux
        corePoints = self.getCorePoints()
        outerPoints = [i for i in range(len(self.data)) if i not in corePoints]
        #clusterise les points centraux
        clusterCore = self.clusterCorePoints(corePoints)
        #ajoute les outlayer dans les clusters
        filledCore = self.fillClusters(clusterCore,outerPoints)
        #transforme les index en data
        self.clusters = self.fixData(filledCore)
        return self.clusters
        