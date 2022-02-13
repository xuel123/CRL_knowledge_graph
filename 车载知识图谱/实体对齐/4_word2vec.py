# -*- coding: UTF-8 -*-
import jieba
import re
import logging
import sys
import gensim.models as word2vec
from gensim.models.word2vec import LineSentence, logger

filePath = '语料库.txt'
fileSegWordDonePath = 'corpusSegDone.txt'

# 将每一行文本依次存放到一个列表
fileTrainRead = []
with open(filePath, encoding='utf-8') as fileTrainRaw:
    for line in fileTrainRaw:
        fileTrainRead.append(line)

# 去除标点符号
fileTrainClean = []
remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】['']""《》？“”‘’！[\\]^_`{|}~]+'
for i in range(len(fileTrainRead)):
    string = re.sub(remove_chars, "", fileTrainRead[i])
    fileTrainClean.append(string)

# 用jieba进行分词
fileTrainSeg = []
file_userDict = 'C3车载设备字典.txt'  # 自定义的词典
jieba.load_userdict(file_userDict)
for i in range(len(fileTrainClean)):
    fileTrainSeg.append([' '.join(jieba.cut(fileTrainClean[i][7:-7], cut_all=False))])  # 7和-7作用是过滤掉<content>标签，可能要根据自己的做出调整
    if i % 100 == 0:  # 每处理100个就打印一次
        print(i)

with open(fileSegWordDonePath, 'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write('\n'.encode("utf-8"))

def train_word2vec(dataset_path, out_vector):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    # sentences = LineSentence(smart_open.open(dataset_path, encoding='utf-8'))  # 或者用smart_open打开
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频）
    model = word2vec.Word2Vec(sentences, size=100, sg=1, window=5, min_count=1, workers=4, iter=50)
    # (iter随机梯度下降法中迭代的最大次数，sg为1是Skip-Gram模型)
    # 保存word2vec模型
    model.save("word2vec.model_hub")
    model.wv.save_word2vec_format(out_vector, binary=False)


if __name__ == '__main__':
    dataset_path = "corpusSegDone.txt"
    out_vector = 'corpusSegDone.vector'
    train_word2vec(dataset_path, out_vector)  # 训练模型
    model = word2vec.Word2Vec.load("word2vec.model_hub")
    print(model.wv.__getitem__("没收到"))

