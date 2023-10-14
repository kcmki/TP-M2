import numpy as np
import pandas as pd
from scipy import stats


def tendance(array):
    return np.mean(array),np.median(array),stats.mode(array)[0]


def quartiles(array,number):
    n = len(array) - 1
    
    sortedArray = np.sort(array)

    qrt = []
    for i in range(0,number+1):
        print(int(n*i/number))
        qrt.append(sortedArray[int(n*i/number)])

    return qrt

def none(array):
    none = 0
    print(len(array))
    for x in array:
        print(x)
    
        if x == None:
            none += 1

    return none, none/len(array)


Data = pd.read_excel('Exo1.xlsx')
Data_np = np.array(Data)
Data_np = Data_np[:,1:]


print(none(Data_np[:,0]))

exit(1)

def sepLine(line):
    return line.split(";")[1:]
def getArray(file):
    arr = []
    for line in file:
        line = sepLine(line)
        Line = []
        for x in line:

            x = x.replace(",",".")

            try:
                Line.append(float(x))
            except:
                Line.append(x)
        arr.append(Line)
    return arr


f = "Exo1.csv"
file = open(f,"r")

array = getArray(file)
print(array)
