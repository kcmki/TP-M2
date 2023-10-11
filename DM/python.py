
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
