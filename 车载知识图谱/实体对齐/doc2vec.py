# coding:utf-8
import jieba
import gensim
import xlrd
import xlwt
from gensim.models.doc2vec import Doc2Vec

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_datasest():
    with open("故障现象提取结果.txt", 'r',encoding="utf-8") as cf:  ##此处是获取你的训练集的地方，从一个文件中读取出来，里面的内容是一行一句话
        docs = cf.readlines()
    x_train = []
    for i, text in enumerate(docs):
        ##如果是已经分好词的，不用再进行分词，直接按空格切分即可
        file_userDict = '../实体识别/C3车载设备字典.txt'  # 自定义的词典
        jieba.load_userdict(file_userDict)
        word_list = ' '.join(jieba.cut(text.split('\n')[0])).split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


def train(x_train, size=100, epoch_num=1):  ##size 是你最终训练出的句子向量的维度，自己尝试着修改一下

    model_dm = Doc2Vec(x_train, min_count=1, window=8, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_hub/model_dm_wangyi')  ##模型保存的位置

    return model_dm

def calcuatevec():

    file_userDict = '../实体识别/C3车载设备字典.txt'  # 自定义的词典
    jieba.load_userdict(file_userDict)
    model_dm = Doc2Vec.load("model_hub/model_dm_wangyi")
    data = xlrd.open_workbook(r'故障提取结果.xlsx')
    file = xlwt.Workbook()
    table_w = file.add_sheet('fc', cell_overwrite_ok=True)

    table = data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols

    for i in range(nrows):
        text = str(table.row(i)[2].value)
        seg_text = ' '.join(jieba.cut(text)).split(' ')
        inferred_vector_dm = model_dm.infer_vector(seg_text)  ##得到文本的向量
        print(text,"向量:",inferred_vector_dm)

    return inferred_vector_dm


def stest():
    model_dm = Doc2Vec.load("model_hub/model_dm_wangyi")
    test_text = ["ATP故障停车"]
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print  (inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=5)
    return sims

if __name__ == '__main__':
    x_train = get_datasest()
    #model_dm = train(x_train)

    doc_2_vec = calcuatevec()
    print(type(doc_2_vec))
    print(doc_2_vec.shape)



#    sims = stest()
#    for count, sim in sims:
#        sentence = x_train[count]
#        words = ''
#        for word in sentence[0]:
#            words = words + word + ' '
#        print(words, sim, len(sentence[0]))
