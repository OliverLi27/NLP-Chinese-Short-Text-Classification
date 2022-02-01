'''
Created on Jan 25, 2022
@author: Xingchen Li
'''
import os
import jieba
import re

DATADIR = "./News/data"
reg = "[^\u4e00-\u9fa5]"
def preprocessing(datadir, name):
    dataset = []
    category = ["news_agriculture","news_house",
                "news_edu","news_car","news_culture"]
    with open(os.path.join(datadir, "_" + name)) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(maxsplit = 3)
            dataset.append([jieba.lcut(re.sub(reg,"", line[-1])), category.index(line[2])]) 
    with open(os.path.join(DATADIR, name), mode="w+") as f:
        for entry in dataset:
            for i in range(len(entry[0])):
                f.write(str(entry[0][i]))
                if i != len(entry[0])-1:
                    f.write(" ")
            f.write("\t")
            f.write(str(entry[1]))
            f.write("\n")

if __name__ == "__main__":
    preprocessing(DATADIR, "train.txt")
    preprocessing(DATADIR, "test.txt")
    preprocessing(DATADIR, "dev.txt")
    