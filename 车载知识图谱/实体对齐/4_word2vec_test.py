# -*- coding: UTF-8 -*-
import math
import gensim.models as word2vec
import jieba
import pandas as pd
import numpy as np
import xlrd
import xlwt
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.metrics.pairwise import cosine_similarity

# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model

#对每个句子的所有词向量取加权均值，来生成一个句子的vector
def build_sentence_vector_weight(sentence,size,w2v_model,key_weight):
    key_words_list=list(key_weight)
    sen_vec=np.zeros(size).reshape((1,size))
    count=0
    for word in sentence:
        try:
            if word in key_words_list:
                sen_vec+=(np.dot(w2v_model[word],math.exp(key_weight[word]))).reshape((1,size))
                count+=1
            else:
                sen_vec+=w2v_model[word].reshape((1,size))
                count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec

def read_excel(size,dic):
    file_userDict = 'C3车载设备字典.txt'  # 自定义的词典
    jieba.load_userdict(file_userDict)
    data = xlrd.open_workbook(r'故障提取结果.xlsx')
    file = xlwt.Workbook()
    table = data.sheet_by_index(0)
    nrows = table.nrows
    phrases = dict()

    for i in range(nrows):
        sen_vec = np.zeros(size)
        text = str(table.row(i)[2].value)
        text_seg = jieba.cut(text)
        text_list = []
        count = 0
        for word in text_seg:
            sen_vec += dic[word]
            text_list.append(word)
            count+=1
        if count != 0:
            sen_vec /= count
        phrases[text] = sen_vec
    return phrases

def embedding_dic():
    file_userDict = 'C3车载设备字典.txt'  # 自定义的词典
    jieba.load_userdict(file_userDict)
    data = xlrd.open_workbook(r'故障提取结果.xlsx')
    file = xlwt.Workbook()
    table = data.sheet_by_index(0)
    nrows = table.nrows
    word_list = []
    key_list1 = []
    key_list = []
    dic = dict()
    model = load_word2vec_model("word2vec.model")

    for i in range(nrows):
        text = str(table.row(i)[2].value)
        text_seg = jieba.cut(text)
        text_list = []
        for word in text_seg:
            text_list.append(word)
        word_list.append(text_list)

    for word in word_list:
        for text in word:
            key_list1.append(text)
    for word in key_list1:
        if not word in key_list:
            key_list.append(word)
    for word in key_list:
        dic[word] = model.wv.__getitem__(word)
    return dic

def show(phrases_dic):
    # 基于2d PCA拟合数据
    X = list(phrases_dic.values())
    print(X)
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    Y = list(phrases_dic.keys())
    dim2_dic = dict(zip(Y, result))
    print('dim2_dic',dim2_dic)
    # 可视化展示
    pyplot.scatter(result[:, 0], result[:, 1])
    print('result', result)
    pyplot.rcParams['font.sans-serif'] = ['FangSong']
    pyplot.rcParams['font.size'] = '5'
    for i, word in enumerate(phrases_dic.keys()):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    return dim2_dic


def compute_cosine(s1, s2, dim2_dic):
    vec1 = dim2_dic[s1]
    vec2 = dim2_dic[s2]
    #print(vec1,vec2)
    sample = [vec1, vec2]
    sim = cosine_similarity(sample)
    print(sim)
    return sim

def cosine_similarity(s1, s2, dim2_dic, norm=False):
    x = dim2_dic[s1]
    y = dim2_dic[s2]
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if all(x) == zero_list or all(y) == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    sim = 0.5 * cos + 0.5 if norm else cos
    print(sim)
    return sim   # 归一化到[0, 1]区间内


if __name__ == '__main__':
    dic = embedding_dic()
    phrases_dic = read_excel(100,dic)
    dim2_dic = show(phrases_dic)
    cosine_similarity('未收到进路预告信息','收不到进路预告信息',dim2_dic)

