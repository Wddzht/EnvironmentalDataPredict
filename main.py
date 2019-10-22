# coding=utf-8
import argparse
import os
import time
import tensorflow as tf
import math
import data
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
                    default=r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.pare.txt",
                    help='train data source')
parser.add_argument('--output_path', type=str, default=r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\model',
                    help='output path')

parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--model_path', type=str, default=r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\model\1571662098',
                    help='model path for test or demo no /checkpoints')
parser.add_argument('--load_model', type=str, default='False', help='train/test/demo')

parser.add_argument('--window', type=int, default=6, help='#epoch of training')
parser.add_argument('--attr_count', type=int, default=12, help='#epoch of training')
parser.add_argument('--batch_size', type=int, default=8, help='#sample of each mini batch')
parser.add_argument('--epoch', type=int, default=550, help='#epoch of training')
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

                if counter % 500 == 0:
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


    def output_prediction(prediction, target, mse, mae, r2):

        for item in prediction:
            for i in range(len(item)):
                attr = item[i] * (_max[i] - _min[i]) + _min[i]
                output_file.write('{:.2f}\t'.format(attr))
            output_file.write('\n')

        output_file.write('\nMSE:\n')
        for item in mse:
            output_file.write('{:.3f}\t'.format(item))
        output_file.write('\nRMSE:\n')
        for item in mse:
            output_file.write('{:.3f}\t'.format(math.sqrt(item)))
        output_file.write('\nMAE:\n')
        for item in mae:
            output_file.write('{:.3f}\t'.format(item))
        output_file.write('\nR2:\n')
        for item in r2:
            output_file.write('{:.3f}\t'.format(item))

        for item in target:
            for i in range(len(item[0])):
                attr = item[0][i] * (_max[i] - _min[i]) + _min[i]
                output_file_t.write('{:.2f}\t'.format(attr))
            output_file_t.write('\n')


    def calculate_rate(prediction, target):
        attr_count = len(prediction[0])
        mse, mae, r2 = [0 for k in range(attr_count)], [0 for k in range(attr_count)], [0 for k in range(attr_count)]
        adv_target = [0 for k in range(attr_count)]
        vari_target = [0 for k in range(attr_count)]

        count = 0
        for list in target:
            data = list[0]
            for j in range(len(data)):
                attr = data[j] * (_max[j] - _min[j]) + _min[j]
                adv_target[j] += attr
            count += 1

        for i in range(len(adv_target)):
            adv_target[i] = adv_target[i] / count

        for i in range(len(prediction)):
            for j in range(attr_count):
                attr_prediction = prediction[i][j] * (_max[j] - _min[j]) + _min[j]
                attr_target = target[i][0][j] * (_max[j] - _min[j]) + _min[j]
                mse[j] += (attr_prediction - attr_target) ** 2
                mae[j] += abs(attr_prediction - attr_target)
                if (j == 2):
                    print(str(attr_prediction) + ' ' + str(attr_target))
                vari_target[j] += (adv_target[j] - attr_target) ** 2

        for j in range(len(adv_target)):
            r2[j] = 1 - (mse[j] / vari_target[j])

        for j in range(len(mse)):
            mse[j] = mse[j] / count

        for j in range(len(mae)):
            mae[j] = mae[j] / count

        return mse, mae, r2


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
                feed_dict = {model.input_seqs: input_seqs,
                             model.keep_prob: 1,
                             model.initial_state: new_state_}
                prediction, _ = sess.run([model.logits, model.final_state],
                                         feed_dict=feed_dict)
                _prediction.extend(prediction)
                _target.extend(target)

            sub_prediction = []
            sub_target = []
            for i in range(int(len(_prediction) - 206), len(_prediction)):
                sub_prediction.append(_prediction[i])
                sub_target.append(_target[i])

            mse, mae, r2 = calculate_rate(sub_prediction, sub_target)
            output_prediction(_prediction, _target, mse, mae, r2)


    ckpt_file = tf.train.latest_checkpoint(os.path.join(args.model_path, 'checkpoints'))
    sample(ckpt_file)
