# Prétraitement 
#   Expression réguliere
#   stopwords
#   normalisation 
#   
files = ["D1.txt","D2.txt","D3.txt","D4.txt"]
import pickle
import nltk


MotsVides = nltk.corpus.stopwords.words('english')



for i in range(len(files)):

    file = open(files[i]).read()
    ExpReg = nltk.RegexpTokenizer('(?:[A-Z]\.)+|\d+(?:\.\d+)?DA?|\w+|\.{3}') # \d : équivalent à [0-9] >>> 
    Termes = ExpReg.tokenize(file) 
    TermesSansMotsVides = [terme for terme in Termes if terme.lower() not in MotsVides]
    Lancaster = nltk.LancasterStemmer()
    TermesNormalisation = [Lancaster.stem(terme) for terme in TermesSansMotsVides]
    my_dict = {}
    for term in TermesNormalisation:
        if term in my_dict:
            my_dict[term] += 1
        else:
            my_dict[term] = 1

    newF = open("Tf-"+files[i],"wb+")
    pickle.dump(my_dict, newF)