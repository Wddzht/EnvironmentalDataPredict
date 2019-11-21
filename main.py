# coding=utf-8
import argparse
import os
import time
import tensorflow as tf
import math
import data
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import LSTMPredictor

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

# hyper parameters
parser = argparse.ArgumentParser(description='LSTM Predictor task')
parser.add_argument('--data_path', type=str,
                    default=r".\EnvironmentalDataPredict\FittingMap\data\ptl_micro_Del_SO2.pare.txt",
                    help='train data source')
parser.add_argument('--output_path', type=str, default=r'.\EnvironmentalDataPredict\model',
                    help='output path')

parser.add_argument('--mode', type=str, default='demo', help='train/demo')
parser.add_argument('--model_path', type=str,
                    default=r'.\EnvironmentalDataPredict\model\1573090396',
                    help='model path for test or demo no /checkpoints')
parser.add_argument('--load_model', type=str, default='False', help='train/test/demo')

parser.add_argument('--window', type=int, default=6, help='#epoch of training')
parser.add_argument('--attr_count', type=int, default=11, help='#epoch of training')
parser.add_argument('--batch_size', type=int, default=8, help='#sample of each mini batch')
parser.add_argument('--epoch', type=int, default=1000, help='#epoch of training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--keep_prob', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--lstm_layer_num', type=int, default=2, help='lstm layer num')

args = parser.parse_args()

if args.mode == 'train':
    timestamp = str(int(time.time()))
    output_path = os.path.join(args.output_path, timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ckpt_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    model_path = os.path.join(ckpt_path, "model")

    batches, _, _ = data.read_str_data(args.data_path, n_seqs=args.batch_size, window=args.window,
                                       attr_count=args.attr_count)

    model = LSTMPredictor(batch_size=args.batch_size,
                          learning_rate=args.lr,
                          grad_clip=args.clip,
                          lstm_layer_num=args.lstm_layer_num,
                          attr_count=args.attr_count,
                          window=args.window,
                          output_dim=args.attr_count)  # todo:

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        if args.load_model == 'True':
            ckpt_file = tf.train.latest_checkpoint(os.path.join(args.model_path, 'checkpoints'))
            saver.restore(sess, ckpt_file)
        else:
            sess.run(model.init_op)
        new_state = sess.run(model.initial_state)

        counter = 0
        for epoch in range(args.epoch):
            for input_seqs, target_seqs in batches:
                counter += 1
                start = time.time()
                feed_dict = {model.input_seqs: input_seqs,
                             model.target_seqs: target_seqs,
                             model.keep_prob: args.keep_prob,
                             model.initial_state: new_state}
                batch_loss, new_state, _, logits, target_seqs = sess.run(
                    [model.loss, model.final_state, model.optimizer, model.logits, model.target_seqs],
                    feed_dict=feed_dict)

                end = time.time()

                if counter % 2000 == 0:
                    print('epoch: {}/{} '.format(epoch + 1, args.epoch),
                          'step: {} '.format(counter),
                          'err rate: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if counter % 2000 == 0:
                    saver.save(sess, model_path, counter)

        saver.save(sess, model_path, counter)


elif args.mode == 'demo':
    output_path = os.path.join(args.model_path, 'prediction.txt')
    output_path_t = os.path.join(args.model_path, 'target.txt')
    output_file = open(output_path, 'w')
    output_file_t = open(output_path_t, 'w')
    batches, _min, _max = data.read_str_data(args.data_path, n_seqs=args.batch_size, window=args.window,
                                             attr_count=args.attr_count)


    def output_prediction(prediction, target, rmse, mae, r2):
        for item in prediction:
            for i in range(len(item)):
                output_file.write('{:.2f}\t'.format(item[i]))
            output_file.write('\n')

        output_file.write('\nRMSE:\n')
        for item in rmse:
            output_file.write('{:.3f}\t'.format(math.sqrt(item)))
        output_file.write('\nMAE:\n')
        for item in mae:
            output_file.write('{:.3f}\t'.format(item))
        output_file.write('\nR2:\n')
        for item in r2:
            output_file.write('{:.3f}\t'.format(item))

        for item in target:
            for i in range(len(item)):
                output_file_t.write('{:.2f}\t'.format(item[i]))
            output_file_t.write('\n')


    def anti_normal(prediction):
        for i in range(len(prediction)):
            for j in range(len(prediction[i])):
                prediction[i][j] = prediction[i][j] * (_max[j] - _min[j]) + _min[j]


    def calculate_rate(prediction, target):
        _prediction = np.array(prediction)
        _target = np.array(target)
        r2_list = []
        mae_list = []
        rmse_list = []
        for j in range(len(_prediction[0])):
            r2_list.append(r2_score(_target[:, j], _prediction[:, j]))
            mae_list.append(mean_absolute_error(_target[:, j], _prediction[:, j]))
            rmse_list.append(math.sqrt(mean_squared_error(_target[:, j], _prediction[:, j])))
        return rmse_list, mae_list, r2_list


    def sample(checkpoint):
        model = LSTMPredictor(batch_size=args.batch_size,
                              learning_rate=args.lr,
                              grad_clip=args.clip,
                              lstm_layer_num=args.lstm_layer_num,
                              attr_count=args.attr_count,
                              window=args.window,
                              output_dim=args.attr_count)  # todo:

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            new_state_ = sess.run(model.initial_state)

            _prediction = []
            _target = []
            for input_seqs, target in batches:
                feed_dict = {model.input_seqs: input_seqs, model.keep_prob: 1, model.initial_state: new_state_}
                prediction, _ = sess.run([model.logits, model.final_state], feed_dict=feed_dict)
                _prediction.extend(prediction)
                _target.extend(target)

            _target = np.array(_target).reshape([-1, len(_target[0][0])])  # 由三维转二维
            anti_normal(_prediction)
            anti_normal(_target)
            rmse, mae, r2 = calculate_rate(_prediction, _target)
            output_prediction(_prediction, _target, rmse, mae, r2)


    ckpt_file = tf.train.latest_checkpoint(os.path.join(args.model_path, 'checkpoints'))
    sample(ckpt_file)
