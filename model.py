import tensorflow as tf
import numpy as np


class LSTMPredictor:
    def __init__(self,
                 batch_size,
                 learning_rate=0.001,
                 grad_clip=5,
                 attr_count=13,
                 window=12,
                 output_dim=13,
                 lstm_layer_num=2):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.input_dim = attr_count * window
        self.output_dim = output_dim
        self.lstm_layer_num = lstm_layer_num

        self.input_seqs = tf.placeholder(tf.float32, shape=[None, 1, self.input_dim], name="input_seqs")
        self.target_seqs = tf.placeholder(tf.float32, shape=[None, 1, attr_count], name="target_seqs")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # build graph
        self.lstm_output, self.final_state = self.lstm_layer(self.input_seqs)
        self.logits = self.output_layer(self.lstm_output)
        self.loss = self.get_loss(self.logits, self.target_seqs)
        self.optimizer = self.get_optimizer(self.loss)

        self.init_op = tf.global_variables_initializer()

    def lstm_layer(self, inputs):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.input_dim)
        lstm_drop = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_drop for _ in range(self.lstm_layer_num)])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state)
        return outputs, state

    def output_layer(self, lstm_output):
        """
        soft_max layer
            input_size: lstm size
        :param lstm_output:
        :return:
        """
        seq_output = tf.concat(lstm_output, 1)
        x = tf.reshape(seq_output, [-1, self.input_dim])
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable(shape=[self.input_dim, self.output_dim],
                                        name='w',
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(shape=[self.output_dim],
                                        name='b',
                                        initializer=tf.zeros_initializer(),
                                        dtype=tf.float32)

        logits = tf.matmul(x, softmax_w) + softmax_b
        # output = tf.nn.softmax(logits)

        return logits

    def get_loss(self, logits, target_seqs):
        logits_ = tf.reshape(logits, [-1])
        target_seqs_ = tf.reshape(target_seqs, [-1])

        # tv = tf.trainable_variables()
        # regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数
        loss = tf.reduce_sum(tf.square(logits_ - target_seqs_))  # + regularization_cost
        return loss

    def get_optimizer(self, loss):
        trainable_var = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_var), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, trainable_var))

        return optimizer
