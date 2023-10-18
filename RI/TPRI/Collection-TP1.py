
files = ["tf-D1.txt","tf-D2.txt","tf-D3.txt","tf-D4.txt"]


for f in files:
    f = open(f).read()
    f = f.replace("{","").replace("}","")

    keynumb = f.split(",")

    for item in keynumb:
        key = item.split(":")[0].replace("'","")
        num = int(item.split(":")[1])

        print("k :",key," Num :",num)