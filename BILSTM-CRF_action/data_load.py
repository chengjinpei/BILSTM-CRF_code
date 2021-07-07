#!/usr/bin/python
# -*- coding: UTF-8 -*-

import codecs
import data_utils
def load_sentences(path):
    """
    加载数据集，每一行至少包含一个汉字和一个标记
    句子和句子之间是以空格进行分割
    最后返回句子集合
    :param path:
    :return:
    """
    # 存放数据集
    sentences = []
    # 临时存放每一个句子
    sentence = []
    for line in codecs.open(path, 'r', encoding='utf-8'):
        # 去掉两边空格
        line = line.strip()
        # 首先判断是不是空，如果是则表示句子和句子之间的分割点
        if not line: #句子是非空
            if len(sentence) > 0:
                sentences.append(sentence)
                # 清空sentence表示一句话完结
                sentence = []
        else:
            if line[0] ==  " ":
                continue
            else:
                word = line.split()  #对字符串进行切片，变成list
                assert len(word) >= 2
                sentence.append(word)
        #循环走完，要判断一下，防止最后一个句子没有进入到句子集合中
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences,tag_sheme):
    '''更新指定的编码'''
    for i,s in enumerate(sentences):
        tags = [w[-1] for w in s] #拿到所有的标签编码
        if not data_utils.check_bio(tags):
            s_str = "\n".join(" ".join(w) for w in s)
            raise Exception("输入的句子应为BIO编码，请检查输入句子%i\n %s"%(i,s_str))
        if tag_sheme == 'BIOES':
            new_tags = data_utils.bio_to_bieos(tags)
            #print("编码转换完成")
            for word,new_tags in zip(s,new_tags):
                word[-1] = new_tags
        else:
            raise Exception('非法目标编码')

def word_mapping(sentences):
    """
    构建字典
    :param sentences:
    :return:
    """
    word_list = [[x[0]for x in s]for s in sentences]
    dico = data_utils.create_dico(word_list) #创建字典（将其转换成标号）
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    word_to_id,id_to_word = data_utils.create_mapping(dico)
    return  dico,word_to_id,id_to_word

def tag_mapping(sentences):
    """
    构建标签的字典
    :param sentences:
    :return:
    """
    tag_list = [[x[1]for x in s]for s in sentences] #拿到所有的标签
    dico = data_utils.create_dico(tag_list)
    tag_to_id,id_to_tag = data_utils.create_mapping(dico)
    return dico,tag_to_id,id_to_tag

def prepare_dataset(sentences,word_to_id,tag_to_id,train=True):
    """
    数据预处理，返回list，其中list包含：
    word_list
    word_id_list
    word char indexs分词信息
    tag_id_list
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param train:
    :return:
    """
    none_index = tag_to_id['O']
    data = []
    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[w if w in word_to_id else '<UNK>'] for w in word_list]
        segs = data_utils.get_seg_feature("".join(word_list)) #获取相应的分词信息
        if train:
            tag_id_list = [tag_to_id[w[-1]]for w in s]
        else:
            tag_id_list = [none_index for w in s]
        data.append([word_list,word_id_list,segs,tag_id_list])
    return data
if __name__ == "__main__":
    path = "data/ner.dev"
    sentences = load_sentences(path)
    #print(sentences)
    print("load sentences ending")
    update_tag_scheme(sentences,"BIOES")
    print("over")
    _,word_to_id,id_to_word=word_mapping(sentences)
    _,tag_to_id,id_to_tag = tag_mapping(sentences)
    dev_data = prepare_dataset(sentences,word_to_id,tag_to_id)
    print("预处理OK")
    data_utils.BatchManage(dev_data,120)