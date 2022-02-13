# coding=utf-8
import math
from gensim import corpora, models
import matplotlib.pyplot as plt

def ldamodel(path,num_topics):
    #cop = open(r'copus.txt', 'r', encoding='UTF-8')copus_reason.txt
    cop = open(path, 'r', encoding='UTF-8')
    train = []
    for line in cop.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)  # list of list 格式

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in
              train]  # corpus里面的存储格式（0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, random_state=1,
                          num_topics=num_topics)  # random_state 等价于随机种子的random.seed()，使每次产生的主题一致

    topic_list = lda.print_topics(num_topics, 10)
    #print("主题的单词分布为：\n")
    #for topic in topic_list:
    #    print(topic)
    return lda, dictionary

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('the info of this ldamodel: \n')
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0
        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("模型困惑度的值为 : %s" % prep)
    return prep

def graph_draw(topic, perplexity):  # 做主题数与困惑度的折线图
    x = topic
    y = perplexity
    plt.plot(x, y, linewidth=1)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()


if __name__ == '__main__':
    a = range(1, 50, 1)  # 主题个数
    p1 = []
    p2 = []
    for num_topics in a:
        path1 = 'copus_reason.txt'
        lda1, dictionary1 = ldamodel(path1,num_topics)
        corpus = corpora.MmCorpus('corpus.mm')
        testset1 = []
        for c in range(int(corpus.num_docs / 100)):  # 如何抽取训练集
            testset1.append(corpus[c * 100])
        prep1 = perplexity(lda1, testset1, dictionary1, len(dictionary1.keys()), num_topics)
        p1.append(prep1)
    for num_topics in a:
        path2 = 'copus.txt'
        lda2, dictionary2 = ldamodel(path2, num_topics)
        corpus = corpora.MmCorpus('corpus.mm')
        testset2 = []
        for c in range(int(corpus.num_docs / 100)):  # 如何抽取训练集
            testset2.append(corpus[c * 100])
        prep2 = perplexity(lda2, testset2, dictionary2, len(dictionary2.keys()), num_topics)
        p2.append(prep2)

    figsize = 7, 5
    figure, ax = plt.subplots(figsize=figsize)
    #graph_draw(a, p1)
    #graph_draw(a, p2)
    x = a
    y1 = p1
    y2 = p2
    plt.rcParams['font.sans-serif'] = ['STSong']
    #plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(x, y1, color= 'red',linewidth=1, label='故障原因')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # x_s = 24
    # y_s = 136.8383640086361
    # plt.scatter([24],[136.8383640086361],s=15,c='r')
    # plt.plot([24,24],[-1,136.8383640086361],linestyle='--',c='r')
    plt.plot(x, y2, color='black', linewidth=1, label='故障现象')
    x_s = 25
    y_s = 136.63659037341674
    plt.legend()
    plt.xlabel("主题数/个")
    plt.ylabel("困惑度")
    plt.show()