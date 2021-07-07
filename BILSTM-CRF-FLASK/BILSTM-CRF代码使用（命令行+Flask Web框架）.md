# BILSTM-CRF代码使用（命令行+Flask Web框架）

## 1.环境搭建

（1）硬件环境：
操作系统：windows 10或者 linux(Ubuntu 16~18)   (本人使用的windows 10)
硬件配置：主要是显卡要求：1660Ti 6G
（2）软件环境：
这里最好自己创建一个虚拟环境，然后在里面配置一下各种库的版本。

 - 创建虚拟环境：conda create –name ccf_ner python==3.6
 - 进入虚拟环境：conda activate ccf_ner
 - tensorflow-gpu==1.13.1
 - cudatoolkit==10.0.130
 - cudnn=7.6.5  

## 2.代码框架

![image-20210706173718628](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706173718628.png)

- modelfile 存放训练好的BILSTM-CRF模型（ckpt）
- static 存放前端文件
- templates 前端网页
- app.py  Flask--Python编写的Web 微框架（api接口）
- config_file 模型初始化参数 （和训练代码一致）
- main.py 主函数(命令行测试)（和训练代码不一致）
- data_utils.py 数据预处理工具（和训练代码不一致）
- model.py 模型构建文件（和训练代码基本一致，添加了evaluate_line（）方法）
- model_utils.py 模型设计时的一些工具(和训练代码不一致)

## 2.1 app.py代码解析

Flask 最小应用

```python
from flask import Flask  #导入相应文件库
app = Flask(__name__) #初始化部分
"""第一部分，初始化：所有的Flask都必须创建程序实例，
web服务器使用wsgi协议，把客户端所有的请求都转发给这个程序实例
程序实例是Flask的对象，一般情况下用如下方法实例化
Flask类只有一个必须指定的参数，即程序主模块或者包的名字，__name__是系统变量，该变量指的是本py文件的文件名"""

@app.route('/') #s申请路由，http://xxxx/即为 hello Flask
def hello_world():
    return 'Hello Flask!'


if __name__ == '__main__':
    app.run()
```



```python
app = Flask(__name__)
flags = tf.app.flags

#配置相关
flags.DEFINE_string('ckpt_path', os.path.join('modelfile', 'ckpt'), '保存模型的位置')
flags.DEFINE_string('map_file', 'maps.pkl', '存放字典映射及标签映射')
flags.DEFINE_string('config_file', 'config_file', '配置文件')

FLAGS = tf.app.flags.FLAGS
#加载模型初始化参数
config = model_utils.load_config(FLAGS.config_file)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

#读取map映射文件word_to_id,tag_to_id
with open(FLAGS.map_file, "rb") as f:
    word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)

#测试每行
def evaluate_line(line):
    with tf.Session(config = tf_config) as sess:
        model = model_utils.create(sess, Model, FLAGS.ckpt_path, config)
        result = model.evaluate_line(sess, data_utils.input_from_line(line, word_to_id), id_to_tag)
        return result

#路由调用网页html
@app.route('/')
def index():
    return render_template("index.html")
#获取所有的命名实体
@app.route('/getNER',methods=['POST'])
def getNER():
    source = request.form.get('source')
    result = evaluate_line(source)  #调用evaluate_line函数，提取命名实体
    return result
#获取姓名实体
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
#获取机构实体
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
```

## 2.2 main.py

代码和flask的基本一致，主要包含模型参数配置、对输入句子进行映射、调用模型、输出结果

```python
flags = tf.app.flags



#配置相关


flags.DEFINE_string('ckpt_path', os.path.join('modelfile', 'ckpt'), '保存模型的位置')
flags.DEFINE_string('map_file', 'maps.pkl', '存放字典映射及标签映射')
flags.DEFINE_string('config_file', 'config_file', '配置文件')


FLAGS = tf.app.flags.FLAGS

def evaluate_line():
    config = model_utils.load_config(FLAGS.config_file)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file,"rb") as f:  #读取映射文件
        word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config = tf_config) as sess:
        model = model_utils.create(sess, Model, FLAGS.ckpt_path, config)
        while True:
            line = input("请输入测试的句子:")
            result = model.evaluate_line(sess, data_utils.input_from_line(line, word_to_id), id_to_tag)
            print(result)


def main(_):
    evaluate_line()

if __name__ == "__main__":
    tf.app.run(main)

```

## 2.3 data_utils.py

- get_seg_feature(words)  获取段落特征

![image-20210706182353450](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706182353450.png)

- input_from_line(line, word_to_id)  从命令行获取输入

  ![image-20210706182952171](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706182952171.png)

  

![image-20210706182737748](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706182737748.png)

## 2.4 model.py

添加了evaluate_line（）方法

![image-20210706195652135](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706195652135.png)

![image-20210706195952331](file://C:/Users/Chengjinpei/AppData/Roaming/Typora/typora-user-images/image-20210706195952331.png?lastModify=1625572790)

## 2.5 model_utils.py

- create(sess,Model,ckpt_path,config) 调取模型

![image-20210706200136150](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706200136150.png)

- result_to_json(strings,tags)  将结果转成json格式文件

```python
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
```

![image-20210706200400375](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706200400375.png)

# 3.代码测试

## 3.1 postman软件测试

Postman 从最初设计上就是为接口测试而生的，所以在程序员中广泛使用，在开发调试网络程序时跟踪一些网络请求，能够高效的帮助后端开发人员独立进行接口测试。软件使用教程参考：https://www.cnblogs.com/dreamyu/p/11716972.html

![image-20210706200928155](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706200928155.png)

# 3.2 Web前端调用

将文件中的前端文件加入项目中即可。在谷歌浏览器输入网址http://127.0.0.1:5000/即可

![image-20210706201039134](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706201039134.png)

![image-20210706201458617](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210706201458617.png)
