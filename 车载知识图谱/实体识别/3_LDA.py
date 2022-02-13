# coding=utf-8
import jieba
import xlrd
import xlwt
import jieba.posseg as psg
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from gensim import corpora, models
import functools
import math
import jieba.analyse
import re
import gensim.models as word2vec

# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = 'hit_stopwords.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,'r',encoding='UTF-8').readlines()]
    return stopword_list

# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    jieba.load_userdict('./C3车载设备字典.txt')
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list

# 去除干扰词
def word_filter(seg_list, pos):
    tag_filter = ['a', 'd', 'n', 'v', 'vn']
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        #if not flag.startswith('n'):
            #continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1 and flag in tag_filter:
            filter_list.append(word)
    return filter_list

# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos, corpus_path='./故障现象.txt'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    doc_list = []
    for line in open(corpus_path, 'r',encoding='UTF-8'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)
    return doc_list

#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、车载知识图谱），主题数量
    def __init__(self, doc_list, keyword_num, model, num_topics=24):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        print('corpus',corpus)
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        print('self.corpus_tfidf',self.corpus_tfidf)

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        lda_result = []
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            lda_result.append(k)
        print("lda:",lda_result)

        return lda_result

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


def topic_extract( word_list, model, pos, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    lda = topic_model.get_simword(word_list)
    return lda

def get_keyphrases(doc_list, lda_result,word_score):
    """
    获取关键短语
    :param keywords_num,param min_occur_num:
    获取keywords_num个关键词构造的可能出现的短语，要求这个短语在原文本中出现的次数至少为min_occur_num
    :return: 关键短语的列表
        """
    keywords_set = set([item for item in lda_result])
    keyphrases = set()
    one = []
    phrases = []
    score = 0
    i = 0
    dic = dict()

    for word in doc_list:
        if word in keywords_set and word not in one:
            one.append(word)
            i = i + 1
            score = score + int(word_score[word])
        else:
            if len(one) > 1:
                keyphrases.add(''.join(one))
                phrases = ''.join(one)
                dic[phrases] = score/i
                print("info",phrases,i,score,score/i)

            if len(one) == 0:
                continue
            else:
                one = []
                score = 0
                i = 0
                phrases = []
            # 兜底
        #score = score/i
        #result.append(score)
    if len(one) > 1:
        keyphrases.add(''.join(one))
        phrases = ''.join(one)
        dic[phrases] = score / i

    print('故障短语：',keyphrases)
    print("dic", dic)
    dic_order = dict(sorted(dic.items(), key=lambda x: (-x[1],x[0]), reverse=False))
    print("排序后", dic_order)
    dic_first = list(dic_order.keys())[0:1]
    print("第一个:", dic_first)

    return dic_first

# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model

def read_excel():
    data = xlrd.open_workbook(r'车载设备故障.xlsx')
    file = xlwt.Workbook()
    table_w = file.add_sheet('fc', cell_overwrite_ok=True)

    table = data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols

    with open("score_reason.txt", "r", encoding="utf-8") as f:
        dic = []
        for line in f.readlines():
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            dic.append(b)
        score = dict(dic)
        print("score",score)

    for i in range(nrows):
        word_score = {}
        lda_score_key = []
        lda_score_value = []
        lda_word = []
        lda_score = []
        pos = True
        text = str(table.row(i)[1].value)
        text_re = re.sub(u"\\(.*?\\)|\\（.*?）|\\的|\\内|\\及|\\“|\\”", "", text)
        text_seg = jieba.cut(text_re)
        text_list = []
        for word in text_seg:
            text_list.append(word)
        print("word",text_list)
        seg_list = seg_to_list(str(table.row(i)[1].value), pos)
        print("处理前:",text)
        filter_list = word_filter(seg_list, pos)
        print("预处理:",filter_list)
        lda = topic_extract(filter_list, '车载知识图谱', pos)
        #model_hub = load_word2vec_model("word2vec.model_hub")  # 加载模型
        #print("词向量：",model_hub.wv.__getitem__(lda))  # 词向量
        for word in lda:
            if word in score.keys():
                lda_word = word
                lda_score = score[word]
            else:
                lda_word = word
                lda_score = 1
            lda_score_key.append(lda_word)
            lda_score_value.append(lda_score)
        word_score = dict(zip(lda_score_key,lda_score_value))
        print("word_score",word_score)
        phrases = get_keyphrases(text_list,lda,word_score)
        print('\n')


        #if i > 0:
        #    table_w.write(i, 0, table.row_values(i)[0])
        #    table_w.write(i, 1, table.row_values(i)[2])
        #    table_w.write(i, 2, filter_list)
        #    table_w.write(i, 3, lda)
        #    #table_w.write(i, 4, phrases)
        #    table_w.write(i, 5, phrases)

        #else:
        #    table_w.write(0, 0, "文件名")
        #    table_w.write(0, 1, "原词")
        #    table_w.write(0, 2, "分词结果")
        #    table_w.write(0, 3, "lda")
        #    #table_w.write(0, 4,"故障短语")
        #    table_w.write(0, 5, "故障原因")

    #file.save(r'lda_result.xls')
    return lda

if __name__ == '__main__':
    read_excel()