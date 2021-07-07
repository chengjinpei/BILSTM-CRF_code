#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Chengjinpei
import os
import json
from collections import OrderedDict
import logging
import tensorflow as tf
import codecs

def get_logger(log_file):
    """
    定义日志方法
    :param log_file:
    :return:
    """
    #（1）创建一个logging实例
    logger = logging.getLogger(log_file)
    #（2）设置logger的全局日志级别为DEBUG ,详细的日志等级可以参考https://www.cnblogs.com/yyds/p/6901864.html
    logger.setLevel(logging.DEBUG)
    #（3）创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)  #	用于将日志记录发送到指定的目的位置，放入磁盘
    fh.setLevel(logging.DEBUG)
    #（4）创建一个屏幕(控台)的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()  #日志输出到屏幕上
    ch.setLevel(logging.INFO)
    #（5）设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s") #输出时间、日志器名称 、日志级别、日志信息
    #（6）添加ch,fh的格式(设置格式其，决定了日志记录最终的输出格式)
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    #（7）添加ch,fh文件到logger
    logger.addHandler(ch)  #为该logger对象添加一个handler对象
    logger.addHandler(fh)
    return logger


def load_config(config_file):
    """
    加载配置文件
    :param config_file:
    :return:
    """
    with open(config_file, encoding='utf-8') as f:
        return json.load(f)


def create(sess, Model, ckpt_path, config):
    """
    :param sess:
    :param Model:
    :param ckpt_path:
    :param config:
    :return:
    """
    model = Model(config)

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception("没有已存在模型可以获取")

    return model

def result_to_json(strings,tags):
    """
    :param strings:
    :param tags:
    :return:
    """
    item ={"string":strings,"entities":[]}
    entity_name =""
    entity_start = 0
    idx = 0

    for word,tag in zip(strings,tags):
        if tag[0] == "S":
            item["entities"].append({"word":word,"start":idx+1,"type":tag[2:]})
        elif tag[0] == "B":
            entity_name = entity_name+ word
            entity_start = idx
        elif tag[0] == "I":
            entity_name = entity_name+word
        elif tag[0] == "E":
            entity_name = entity_name+word
            item["entities"].append({"word":entity_name,"start":entity_start,'end':idx+1,"type":tag[2:]})
        else:
            entity_name = ""
            entity_start = idx
        idx = idx+1
    return item
