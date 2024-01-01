
import pickle


files = ["tf-D{}.txt","tf-D{}.txt","tf-D{}.txt","tf-D{}.txt"]

Dict = {}
for i in range(4):
    
    with open(files[i].format(i+1),"rb") as file:
        loaded_dict = pickle.load(file)
        for key in loaded_dict.keys():
            Dict[(key,i)] = loaded_dict[key]


print(Dict)
newF = open("Tf-All","wb+")
pickle.dump(Dict, newF)


exit(1)


