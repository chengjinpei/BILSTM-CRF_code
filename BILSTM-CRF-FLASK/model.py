#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Chengjinpei
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import numpy as np
import data_utils
import model_utils


class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config['lr']
        self.word_dim = config['word_dim']
        self.lstm_dim = config['lstm_dim']
        self.seg_dim = config['seg_dim']
        self.num_tags = config['num_tags']
        self.num_words = config['num_words']
        self.num_sges = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # 申请占位符
        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="wordInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")

        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.word_inputs))
        length = tf.reduce_sum(used, reduction_indices=1) #[batch_size,1]

        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.word_inputs)[0]
        self.num_steps = tf.shape(self.word_inputs)[-1]
        #word_inputs: [batch_size, num_steps]
        # embedding层单词和分词信息
        embedding = self.embedding_layer(self.word_inputs, self.seg_inputs, config)

        # lstm输入层
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)  #[?,?,120]

        # lstm输出层
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths) #[?,?,200]

        # 投影层
        self.logits = self.project_layer(lstm_outputs)  #[?,?,13]

        # 损失
        self.loss = self.crf_loss_layer(self.logits, self.lengths)  #scalar loss

        with tf.variable_scope('optimizer'):
            optimizer = self.config['optimizer']
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradDAOptimizer(self.lr)
            else:
                raise Exception("优化器错误")

            grad_vars = self.opt.compute_gradients(self.loss)
            capped_grad_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']), v] for g, v in
                                grad_vars]

            self.train_op = self.opt.apply_gradients(capped_grad_vars, self.global_step)

            # 保存模型
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) #tf.global_variables()查看管理变量的函数

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

    def biLSTM_layer(self,lstm_inputs,lstm_dim,lengths,name=None):
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

    def project_layer(self,lstm_outputs,name= None):
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

    def crf_loss_layer(self,project_logits,lenghts,name= None):
        """

        :param project_logits: [1,num_steps,num_tags]
        :param lengths: [batchsize,1]
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
            ) #start_logits:[?,1,14]

            pad_logits = tf.cast(
                small_value *
                tf.ones(shape=[self.batch_size, self.num_steps, 1]),
                dtype=tf.float32
            ) #pad_logits:[?,?,1]
            #logits:[?,?,14],start_logits:[?,1,14],pad_logits:[?,?,1],target:[?,?]
            logits = tf.concat(
                [project_logits, pad_logits],
                axis=-1
            ) #project_logits:[1,?,13],pad_logits:[?,?,1] ->logits:[?,?,14]
            logits = tf.concat(
                [start_logits, logits],
                axis=1
            )#logits:[?,?,14],start_logits:[?,1,14]  ->[?,?,14]

            targets = tf.concat(
                [tf.cast(
                    self.num_tags * tf.ones([self.batch_size, 1]),
                    tf.int32
                ),
                    self.targets
                ]
                ,
                axis=-1
            )  #target:[batch_size,1]

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer
            ) #[14,14]

            log_likehood, self.trans = crf_log_likelihood(
                inputs=logits,               #[?,14]
                tag_indices=targets,         #[]
                transition_params=self.trans,#[14,14]
                sequence_lengths=lenghts + 1 #[]
            )
            crf_output = tf.reduce_mean(-log_likehood)
            print(crf_output.shape.as_list())
            return crf_output                #[?,?,13]
        #https://blog.csdn.net/yangfengling1023/article/details/82909585

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


    def create_feed_dict(self,is_train,batch):
        """
        :param is_train:
        :param batch:
        :return:
        """
        _,words,segs,tags = batch
        feed_dict = {
            self.word_inputs:np.asarray(words),
            self.seg_inputs:np.asarray(segs),
            self.dropout:1.0
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config['dropout_keep']
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess:
        :param is_train:
        :param bath:
        :return:
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _= sess.run(
                [self.global_step, self.loss, self.train_op], feed_dict
            )#feed_dict参数的作用是替换图中的某个tensor的值或设置graph的输入值
            return global_step, loss
        else:
           lengths, logits =  sess.run([self.lengths, self.logits], feed_dict)
           return lengths, logits

    def evaluate(self,sess,data_manager,id_to_tag):
        """
        :param sess:
        :param data_manager:
        :param id_to_tag:
        :return:
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths , logits = self.run_step(sess, False, batch)
            batch_paths = self.decode(logits, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        """
        :param sess:
        :param inputs:
        :param id_to_tag:
        :return:
        """
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        bath_path = self.decode(scores, lengths, trans)
        tags = [
            id_to_tag[idx] for idx in bath_path[0]
        ]

        return model_utils.result_to_json(inputs[0][0], tags)
