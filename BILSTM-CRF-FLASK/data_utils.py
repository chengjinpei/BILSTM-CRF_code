#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Chengjinpei
import jieba
jieba.initialize()
import math
import random
import codecs
import numpy as np
import os

def get_seg_feature(words):
    """
    利用jieba分词，采用类似BIO编码
    0表示单字词，1表示词的开始，2表示词中，3表示字的结尾
    :param words:
    :return:seg_features
    """
    seg_features = []
    word_list = list(jieba.cut(words))
    for word in word_list:
        if len(word) == 1:
            seg_features.append(0)
        else:
            temp = [2] *len(word) #除了首尾的所有位置都是2
            temp[0] = 1  #首部是1
            temp[-1] = 3  #末尾是3
            seg_features.extend(temp)
    return seg_features
def input_from_line(line, word_to_id):
    """
    :param line:  输入句子
    :param word_to_id: word的id号
    :return:
    """
    inputs = list()
    inputs.append([line])
    if line is None:
        return "错误的输入"
    line.replace(" ","$")
    inputs.append(
        [
            [word_to_id[word] if word in word_to_id else word_to_id["<UNK>"] for word in line]
        ]
    )
    inputs.append([get_seg_feature(line)])
    inputs.append([[]])
    return inputs
