#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Chengjinpei
from flask import Flask,render_template,request

#系统
import os
import tensorflow as tf
import pickle

#自定义
import data_utils
import model_utils
from model import Model

app = Flask(__name__)
flags = tf.app.flags

#配置相关
flags.DEFINE_string('ckpt_path', os.path.join('modelfile', 'ckpt'), '保存模型的位置')
flags.DEFINE_string('map_file', 'maps.pkl', '存放字典映射及标签映射')
flags.DEFINE_string('config_file', 'config_file', '配置文件')

FLAGS = tf.app.flags.FLAGS

config = model_utils.load_config(FLAGS.config_file)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

with open(FLAGS.map_file, "rb") as f:
    word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)


def evaluate_line(line):
    with tf.Session(config = tf_config) as sess:
        model = model_utils.create(sess, Model, FLAGS.ckpt_path, config)
        result = model.evaluate_line(sess, data_utils.input_from_line(line, word_to_id), id_to_tag)
        return result

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/getNER',methods=['POST'])
def getNER():
    source = request.form.get('source')
    result = evaluate_line(source)
    return result

@app.route('/getPer',methods=['POST'])
def getPer():
    source = request.form.get('per')
    result = evaluate_line(source)
    #print(type(result))
    if result is None:
        return "没有可识别的实体"
    entites = result['entities']
    per = []
    for entity in entites:
        if entity['type'] == "PER":
            per.append(entity['word'])
    return " ".join(per)

@app.route('/getOrg',methods=['POST'])
def getOrg():
    source = request.form.get('org')
    result = evaluate_line(source)
    #print(type(result))
    if result is None:
        return "没有可识别的实体"
    entites = result['entities']
    org = []
    for entity in entites:
        if entity['type'] == "ORG":
            org.append(entity['word'])
    return " ".join(org)
if __name__ == '__main__':
    app.run(debug=True, port='5000', host='127.0.0.1')