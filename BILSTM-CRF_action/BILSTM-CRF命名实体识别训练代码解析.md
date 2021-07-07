# BILSTM-CRF命名实体识别

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

## 2.代码整体框架

代码结构：

- data/ 用于存放训练集、测试集、验证集、词向量
- conlleval.py 用于测试模型的预测的准确率、精准率、召回率
- data_loader.py 用于读取数据
- data_utils.py  数据读取的一些工具
- main 主程序，用于训练
- map.pkl 用于模型的四个列表映射
- model.py 模型设计代码
- model_utils.py 模型设计时的一些工具

![image-20210519165817219](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210519165817219.png)

###  2.1 main.py主程序解析

- 首先是一些参数设置，这里直接调用tf.app.flags类，用于接收从终端传入的命令行参数

  <img src="C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210523110107749.png" alt="image-20210523110107749" style="zoom:80%;" />

【解析】tf.app.flags.DEFINE_string() ：定义一个用于接收 string 类型数值的变量，其中DEFINE_string() 方法包含三个参数，分别是变量名称，默认值，用法描述

- evaluate ()用于评估验证集和测试集的效果
- train（）用于训练模型

```python
（1）加载数据集
data_loader.load_sentences（）
    """
    加载数据集，每一行至少包含一个汉字和一个标记
    句子和句子之间是以空格进行分割
    最后返回句子集合
    :param path:
    :return: sentences (一共包含了2218个句子)，sentences里面包含sentence([词，lable])
    """
（2）转换编码--BIO标注转BIEOS
update_tag_scheme（）
（3）创建单词映射及标签映射
FLAGS.map_file
data_loader.word_mapping（）
（4）数据预处理，返回列表（word_list、word_id_list、word char indexs、tag_id_list）
data_loader.prepare_dataset（）
（5）准备批量数据（batch）
data_utils.BatchManager()
（6）撰写日志文件
（7）GPU等参数设置，开始训练模型
```

## 2.2 data_loader.py程序解析

- load_sentence()  读取句子

  ```
  加载数据集，每一行至少包含一个汉字和一个标记
  句子和句子之间是以空格进行分割
  最后返回句子集合
  ```

  ![image-20210606200339228](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210606200339228.png)

- updata_tag_scheme() 更新为指定编码BIO -->BIEOS

  data_utils.bio_to_bioes()

- word_mapping() 构建字典

  ![image-20210606200918687](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210606200918687.png)

  ```python
  """
  构建字典
  :param sentences:
  :return:dico,word_to_id,id_to_word
  """
  ```

  data_utils.create_dico()  #创建词频字典dico

  ```python
  """
  对于item_list中的每一个items，统计items中item在item_list中的次数
  item:出现的次数
  :param item_list:
  :return: dico
  """
  ```

  data_utils.create_mapping()  #按照词频词典的频率，将word和id依次做映射

  ```python
  """
  创建item to id, id_to_item
  item的排序按词典中出现的次数
  :param dico:
  :return:item_to_id, id_to_item
  """
  ```

- tag_mapping() 构建标签字典(同word_mapping一致)

  data_utils.create_dico()

  data_utils.create_mapping()

- prepare_dataset()  返回字及标签的映射list

  ```python
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
  :return:data
  
  ```
  
  ![image-20210606202656051](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210606202656051.png)

【注意】data列表里包含word_list、word_id_list、word char indexs分词信息、tag_id_list

data_utils.get_seg_features()

```python
"""
利用jieba分词，采用类似BIO编码
0表示单字词，1表示词的开始，2表示词中，3表示字的结尾
:param words:
:return:seg_fetures
"""
```

![image-20210606203026186](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210606203026186.png)

## 2.3 data_utils.py解析

- check_bio() 检测输入的标签是否为bio编码

- bio_to_bieos() 把bio编码转换成bieos编码

- bieos_to_bio()  把bieos编码转成bio编码

- create_dico() 统计items中item在item_list中的次数（相当于词频词典）（参照上一节）

- create_mapping（）创建item to id, id_to_item  item的排序按词典中出现的次数（参照上一节）

- get_seg_features()  利用jieba分词，获取seg_feature（参照上一节）

- load_word2vec() 读取wiki词向量100维

- augment_with_pretrained（）加载预训练的词向量

- class BatchManager()  数据batch处理

  self.sort_and_pad(data, batch_size)：对每个批次的数据，按照句子的长度进行排序，而且选取最大的长度的句子作为pad的长度

  将data划分成20个batch_data

  ![image-20210606203604902](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210606203604902.png)

  pad_data(data): 对data进行padding

  按照每个batch_data最大长度的句子，进行pad其他所有的句子，保证每个batch的数据是整齐的

  iter_batch(shuffle)是否打乱batch_data

  

## 2.4 model.py解析

class Model（）模型结构：

- embedding_layer()

  ```python
  def embedding_layer(self,word_inputs,seg_inputs,config,name=None):
      """
      :param word_inputs: one_hot编码  [batch_size,num_steps,emb_dim]
      :param seg_inputs: 分词的特征
      :param config: 配置
      :param name:层的命名
      :return: [batch_size,num_steps,emb_dim+seg_dim]
      """
      embedding = []
      with tf.variable_scope("word_embedding" if not name else name),tf.device('/gpu:0'):
          self.word_lookup = tf.get_variable(
              name="word_embedding",
              shape = [self.num_words,self.word_dim],
              initializer= self.initializer
          )
          embedding.append(tf.nn.embedding_lookup(self.word_lookup,word_inputs))
  
          if config['seg_dim']:
              with tf.variable_scope("seg_embedding"),tf.device('/gpu:0'):
                  self.seg_lookup = tf.get_variable(
                      name= 'sef_embedding',
                      shape = [self.num_sges,self.seg_dim],
                      initializer = self.initializer
                  )
                  embedding.append(tf.nn.embedding_lookup(self.seg_lookup,seg_inputs))
          embed = tf.concat(embedding,axis=-1)#词向量100维，段落20维，最终120维
      print(embed.shape.as_list())
      return embed
  ```

- biLSTM_layer()

  ```python
  """
  :param lstm_inputs: [batch_size,num_steps,emb_size][?,?,120]
  :param lstm_dim:
  :param lengths:
  :param name:
  :return: [batch_size,num_steps,2*emb_size] [?,?,200]
  """
  with tf.variable_scope("word_biLSTM" if not name else name):
      lstm_cell = {}
      for direction in ["forward","backward"]:
          with tf.variable_scope(direction):
              lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                  lstm_dim,
                  use_peepholes= True,
                  initializer = self.initializer,
                  state_is_tuple= True
              )
      outputs,final_status = tf.nn.bidirectional_dynamic_rnn(
          lstm_cell['forward'],
          lstm_cell['backward'],
          lstm_inputs,
          dtype=tf.float32,
  
          sequence_length=lengths
      )
  lstm_out = tf.concat(outputs,axis=2)
  print(lstm_out.shape.as_list())
  return lstm_out
  ```

- project_layer()

  ```python
  """
  :param lstm_outputs: [batch_size,num_steps,emb_size*2] [?,?,200]
  :param name:
  :return: [batch_size,num_step,num_tags] [?,?,13]
  """
  with tf.variable_scope('project_layer' if not name else name):
      with tf.variable_scope('hidden_layer'):
          #W:[100,13],b:[13,]
          W= tf.get_variable(
              "W",
              shape = [self.lstm_dim*2,self.lstm_dim],  #lstm_dim为隐藏层的维度
              dtype = tf.float32,
              initializer = self.initializer
          )
          b = tf.get_variable(
              "b",
              shape= [self.lstm_dim],
              dtype= tf.float32,
              initializer= tf.zeros_initializer()
          )
          out_put = tf.reshape(lstm_outputs,shape=[-1,self.lstm_dim*2]) #[?,200]
          hidden = tf.tanh(tf.nn.xw_plus_b(out_put,W,b)) #[?,100]
  
          #计算每个标签的分数
          with tf.variable_scope('logits'):
              W = tf.get_variable(
                  "W",
                  shape=[self.lstm_dim, self.num_tags],
                  dtype=tf.float32,
                  initializer=self.initializer
              )
              b = tf.get_variable(
                  "b",
                  shape=[self.num_tags],
                  dtype=tf.float32,
                  initializer=tf.zeros_initializer()
              )
              pred = tf.nn.xw_plus_b(hidden,W,b) #[?,13]
              pred_output = tf.reshape(pred,[-1,self.num_steps,self.num_tags]) #[?,?,13]
      print(pred.shape.as_list())
      return pred_output
  ```

- crf_loss_layer()

  ```python
  """
  
  :param project_logits: [1,num_steps,num_tags]
  :param lengths:
  :param name:
  :return: scalar loss
  """
  with tf.variable_scope('crf_loss' if not name else name):
      small_value = -10000.0
      start_logits = tf.concat(
          [
              small_value *
              tf.ones(shape=[self.batch_size, 1, self.num_tags]),
              tf.zeros(shape=[self.batch_size, 1, 1])
          ],
          axis=-1
      )
  
      pad_logits = tf.cast(
          small_value *
          tf.ones(shape=[self.batch_size, self.num_steps, 1]),
          dtype=tf.float32
      )
      #logits:[?,?,14],start_logits:[?,1,14],pad_logits:[?,?,1],target:[?,?]
      logits = tf.concat(
          [project_logits, pad_logits],
          axis=-1
      )
      logits = tf.concat(
          [start_logits, logits],
          axis=1
      )
  
      targets = tf.concat(
          [tf.cast(
              self.num_tags * tf.ones([self.batch_size, 1]),
              tf.int32
          ),
              self.targets
          ]
          ,
          axis=-1
      )
  
      self.trans = tf.get_variable(
          "transitions",
          shape=[self.num_tags + 1, self.num_tags + 1],
          initializer=self.initializer
      )
  
      log_likehood, self.trans = crf_log_likelihood(
          inputs=logits,
          tag_indices=targets,
          transition_params=self.trans,
          sequence_lengths=lenghts + 1
      )
      crf_output = tf.reduce_mean(-log_likehood)
      print(crf_output.shape.as_list())
      return crf_output
  ```

  ```python
  start_logits:[?,1,14]
  pad_logits:  [?,?,1]
  logits:      [?,?,14]
  logits:      [?,?,14]
  target :[?,?]
  trans :[14,14]
  crf_output :[?,?,13]
  length:[batch_size]
  preject_logit:[?,1,14]
  ```

- decode()

  ```python
      def decode(self,logits,lengths,matrix):
          """
  
          :param logits:[batch_size,num_steps,num_tags]
          :param lengths:
          :param matrix:
          :return:
          """
          paths = []
          small = -1000.0
          start = np.asarray([[small]*self.num_tags+[0]])
          for score ,length in zip(logits,lengths):
              score = score[:length]   #[1,batch_size]
              pad = small * np.ones([length,1]) #[batch_size,1]
              logits = np.concatenate([score,pad],axis=1)
              logits = np.concatenate([start, logits], axis=0)
              path,_ = viterbi_decode(logits,matrix) #[batch_size,num_steps,num_tags]
              paths.append(path[1:])
          print(len(paths))
          return paths
  ```

- create_feed_dict()  创建喂入字典

- run_step()   构建图去训练

- evaluate()    构建待评估样本 result: [char,gold,pred]

  ![image-20210705093839042](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210705093839042.png)

## 2.5model_utils.py解析

- get_logger(log_file) 定义日志文件

  定义日志的方法一共分成6步：

  ```python
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
  ```

- config_model() 初始化模型参数

  基本上就是FLAGS定义的一些参数

- make_path() 创建文件夹（如果ckpt和log文件不存在，则创建相应文件夹）

  ```python
  if not os.path.isdir(params.ckpt_path):
      os.makedirs(params.ckpt_path)
  if not os.path.isdir('log'):
      os.makedirs('log')
  ```

- save_config() 保存配置文件  --dump往配置文件里写内容

  ```python
  with open(config_file, 'w', encoding='utf-8') as f:
      json.dump(config, f, ensure_ascii=False, indent=4)  #indent表示缩进
  ```

- load_config() 加载配置文件 --load从配置文件里加载内容

  ```python
  with open(config_file, encoding='utf-8') as f:
      return json.load(f)
  ```

- print_config() 打印模型参数  --Python ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串。

  ```python
  for k, v in config.items():
      logger.info("{}:\t{}".format(k.ljust(15), v))  #ljust左对齐
  ```

- create() 创建模型实例

  ```python
  """
  :param sess:
  :param Model:
  :param ckpt_path:
  :param load_word2vec:
  :param config:
  :param id_to_word:
  :param logger:
  :return:
  """
  model = Model(config)
  
  ckpt = tf.train.get_checkpoint_state(ckpt_path)
  # 如果保存模型的文件存在，则读取；否则得重新训练
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      logger("读取模型参数，从%s" % ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
  
  else:
      logger.info("重新训练模型")
      sess.run(tf.global_variables_initializer())
      if config['pre_emb']:
          emb_weights = sess.run(model.word_lookup.read_value())
          emb_weights = load_word2vec(config['emb_file'], id_to_word, config['word_dim'], emb_weights)
          sess.run(model.word_lookup.assign(emb_weights))
          logger.info("加载词向量成功")
  return model
  ```

- test_ner() 测试ner效果，会调用conlleavl文件

  ```python
  """使用评估脚本  待评估文件-results
  :param results:
  :param path:
  :return:
  """
  output_file = os.path.join(path, 'ner_predict.utf8')  # 输出文件path/ner_predict.utf.8
  with codecs.open(output_file, "w", encoding="utf-8") as f_write:
      to_write = []
      for line in results: #results是模型的预测列表list
          for iner_line in line:
              to_write.append(iner_line + "\n")
          to_write.append("\n")
      f_write.writelines(to_write)
  eval_lines = return_report(output_file) #把整个文件评估的结果存进eval_line
  return eval_lines
  ```

  result结果，最终保存写到output_file：

  ![image-20210613104317583](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210613104317583.png)

  eval_line结果，最终在日志文件中显示出来:

  ![image-20210613104536280](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210613104536280.png)

- save_model() 保存模型

  ```python
  """
  :param sess:
  :param model:
  :param path:
  :param logger:
  :return:
  """
  checkpoint_path = os.path.join("path","ner.ckpt")  #path/ner.ckpt
  model.saver.save(sess,checkpoint_path)
  logger.info("模型已经保存")
  ```

![image-20210613103948345](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210613103948345.png)

# 调用代码

![image-20210705161252311](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210705161252311.png)

![image-20210705161415034](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210705161415034.png)

![image-20210705161503729](C:\Users\Chengjinpei\AppData\Roaming\Typora\typora-user-images\image-20210705161503729.png)
