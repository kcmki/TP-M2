# Prétraitement 
#   Expression réguliere
#   stopwords
#   normalisation 
#   
files = ["D1.txt","D2.txt","D3.txt","D4.txt"]

for i in range(len(files)):

    file = open(files[i]).read()

    tf = {}

    for word in file.split(" "):
        word = word.replace("\n","")
        word = word.replace(",","")
        if word in tf.keys():
            tf[word] +=1
        else:
            tf[word] = 1

    newF = open("Tf-"+files[i],"w+")
    
    newF.write(str(tf))

    continue
    for key in tf.keys():
        newF.writelines(key+":"+str(tf[key]))

