import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import time
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import re

def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def metadata_extraction(file_name):
    # key_array = []
    # index_array = []
    # type_array = []
    # value_array = []
    description = []

    # group_count = 0
    attr_count = 0
    # dataset_count = 0
    line_count = 0
    substring1 =  'description'
    substring2 = 'Description'
    substring3 = 'comment'
    substring4 = 'Comment'
    with open(file_name,'r',errors='ignore') as ms:
        flag = 0
        
        for line in ms.readlines():
            line_count += 1
            index_num = 0
            for c in line:
                if c == ' ':
                    index_num += 1
                else:
                    break 
            # group = re.findall(r'GROUP "(.+?)" {',line)
            # if len(group) > 0:
            #   group_count += 1
            #   key_array.append(group[0])
            #   index_array.append(index_num//3)
            #   value_array.append(0)
            #   type_array.append("GROUP")
            #   continue
            # dataset = re.findall(r'DATASET "(.+?)" {',line)
            # if len(dataset)>0:
            #   dataset_count +=1
            #   key_array.append(dataset[0])
            #   index_array.append(index_num//3)
            #   value_array.append(0)
            #   type_array.append("DATASET")
            #   continue
            attr = re.findall(r'ATTRIBUTE "(.+?)" {',line)
            if len(attr) > 0:
                if substring1 in attr[0] or substring2 in attr[0] or substring3 in attr[0] or substring4 in attr[0]:
                    attr_count += 1
                    # key_array.append(attr[0])
                    # index_array.append(index_num//3)
                    flag = 1
            # data_type = re.findall(r'DATATYPE (.+?)$',line)
            # if len(data_type) >0:
            #   if flag == 1:
            #       string_type = re.findall(r'H5T_(.+?) {',data_type[0])
            #       if len(string_type)>0:
            #           type_array.append('STRING')
            #       else:
            #           type_array.append('NUMBER')
                # diff_key = len(type_array) - len(key_array)
                # if diff_key > 1:
                #   print("error "+str(line_count))
                # print(data_type[0])
            # line_split = line.split(':')
            # if len(line_split) > 1:
            #   attr_value = re.findall(r'"(.+?)"$',line_split[1])
            #   print(attr_value)
            attr_value = re.findall(r'\(0\): (.+?)$',line)
            if len(attr_value) > 0 and flag ==1:
                string_replace = attr_value[0].replace("\"","")
                description_list = string_replace.split('.')
                if description_list[len(description_list)-1] is "":
                    description_list.pop(len(description_list)-1)


                for sentence in description_list:
                    description.append(sentence)
                # value_array.append(attr_value[0])
                flag = 0
                # value_array.append(attr_value[0])
                # print(attr_value[0])
    # print("group_count="+str(group_count))
    # print("dataset_count="+str(dataset_count))            
    # print("attr_count="+str(attr_count))
    # print("length of differnet arrays")
    # print('length of description:')
    # print(len(description))
    # print('number of attr')
    # print(attr_count)
    # for i in range(10):
    #     print(i)
    #     print("======================")
    #     print(description[i])
    return description

def summary(metadata,dimension):
    sentences = []
    for s in metadata:
        sentences.append(s)
    # print(len(sentences))
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    stop_words = stopwords.words('english')
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    word_embeddings = {}
    filename = 'glove.6B.'+str(dimension)+'d.txt'
    f = open(filename, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((dimension,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((dimension,))
        sentence_vectors.append(v)
    # print(len(sentence_vectors))
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,dimension), sentence_vectors[j].reshape(1,dimension))[0,0]
    # for s in sentence_vectors:
    #     print(s)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    sn = 2
    for i in range(sn):
        print(ranked_sentences[i][1])


if __name__ == '__main__':
    # starttime = time.clock()
    # metadata = metadata_extraction('Metadata/GLAH12_634_2121_002_0365_0_01_0001')
    # endtime = time.clock()
    # print(endtime - starttime)
    # metadata = ['My name is Chenxu','My word is cat','how are you','how old are you']
    print("=========================")
    starttime = time.clock()
    summary(metadata,50)
    endtime = time.clock()
    print(endtime - starttime)
    print("=========================")
    starttime = time.clock()
    summary(metadata,100)
    endtime = time.clock()
    print(endtime - starttime)
    print("=========================")
    starttime = time.clock()
    summary(metadata,200)
    endtime = time.clock()
    print(endtime - starttime)









