# -*- coding: utf8 -*-
import os

import tensorflow as tf
from tensorflow.python.ops import array_ops, sparse_ops

from models import OcrModelDef
from models.tf import TFTrainModel, TFInferModel
from models.tf import TFModelDef
from models.tf.ocr.feeder import TFCtcDataFeeder
from util import Logger
from util.io_utils import FileUtils
from util.label_utils import LabelUtils


K = tf.keras.backend


class TFOcrModelDef(TFModelDef, OcrModelDef):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.init_ocr_model(**kwargs)
        # self.rnn_size = self.args.get('rnn_size', 200)
        self.is_rnn  = self.args.get('is_rnn', "False")

    def decode_tensor_to_labels(self, results):
        pass

    def build_model_outputs(self, input, **kwargs):
        raise NotImplemented()

    def build_model_old(self, input=None, **kwargs):  ###########################
        input = self.input
        x=self.dense_net_to_seq(input, is_test=kwargs.get('is_test', False))
        self.y_pred = tf.keras.layers.Dense(15, name='out', activation='softmax')(x)
        decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])
        base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=decoded)
        base_model.summary()
        if kwargs.get('pretraining', True):
            weights_path = 'E:\PycharmProjects\work2\ai-training\pretraining_model\caibao.h5'
            base_model.load_weights(weights_path)
            nclass = self.charset.charset_size
            self.y_pred = tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)
            pre_decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])
            base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=pre_decoded)
            base_model.summary()
        return base_model

    def build_model(self, input=None, **kwargs):  # 2022-1-10
        input = self.input
        x = self.dense_net_to_seq(input, is_test=kwargs.get('is_test', False))
        if eval(self.is_rnn):
            fc_1 = tf.keras.layers.Dense(self.rnn_size * 3)(x)
            fc_2 = tf.keras.layers.Dense(self.rnn_size * 3)(x)

            gru_forward = tf.keras.layers.GRU(self.rnn_size, activation='relu', recurrent_activation='sigmoid',
                                              return_sequences=True, )(fc_1) # (batch,width/8,rnn_size)
            gru_backward = tf.keras.layers.GRU(self.rnn_size, activation='relu', recurrent_activation='sigmoid',
                                               go_backwards=True, return_sequences=True)(fc_2)# (batch,width/8,rnn_size)


            x = tf.keras.layers.Add()([gru_forward, gru_backward]) # (batch,width/8,rnn_size)


        nclass = self.charset.charset_size
        self.y_pred = tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)
        pre_decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])
        base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=pre_decoded)
        base_model.summary()
        return base_model


        # input = input or self.input
        # self.outputs = self.build_model_outputs(input, **kwargs)
        # base_model = tf.keras.Model(inputs=input, outputs=self.outputs)
        # base_model.summery()
        # return base_model


class CtcDecodeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def call(self, inputs, **kwargs):
        x = inputs[0]
        input_lens = K.flatten(inputs[1])

        greedy =False
        beam_width = 100
        top_paths = 1

        from tensorflow.python.ops import math_ops
        from tensorflow.python.ops import ctc_ops as ctc
        from tensorflow.python.framework import dtypes as dtypes_module

        y_pred = math_ops.log(array_ops.transpose(x, perm=[1, 0, 2]) + + 1e-7)
        input_length = math_ops.cast(input_lens, dtypes_module.int32)


        if greedy:
            (decoded, log_prob) = ctc.ctc_greedy_decoder(
                inputs=y_pred, sequence_length=input_length)
        else:
            (decoded, log_prob) = ctc.ctc_beam_search_decoder(
                inputs=y_pred,
                sequence_length=input_length,
                beam_width=beam_width,
                top_paths=top_paths,merge_repeated=False)
        decoded_dense = [
            sparse_ops.sparse_to_dense(
                st.indices, st.dense_shape, st.values, default_value=-1)
            for st in decoded
        ]

        # (decoded, log_prob) = K.ctc_decode(y_pred, input_lens, False)
        return decoded_dense


class TFCtcOcrModelDef(TFOcrModelDef):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.y_pred = None
        self.input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')

    # def build_model(self, **kwargs):  # step 6    ----> 运行到此地后， 该去向 何方
    #     input = self.input
    #     x=self.dense_net_to_seq(input, is_test=kwargs.get('is_test', False))
    #     self.y_pred = tf.keras.layers.Dense(16, name='out', activation='softmax')(x)
    #     # self.y_pred = tf.keras.layers.Dense(7656, name='out', activation='softmax')(x)
    #     decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])
    #     base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=decoded)
    #     base_model.summary()
    #     if kwargs.get('pretraining', True):
    #         weights_path = r'E:\PycharmProjects\work2\ai-training\pretraining_model\caibao.h5'
    #         # weights_path = r'E:\PycharmProjects\work2\ai-training\pretraining_model\full.7.h5'
    #         # base_model.load_weights(weights_path)
    #         nclass = self.charset.charset_size
    #         self.y_pred = tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)
    #         pre_decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])
    #         base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=pre_decoded)
    #         base_model.summary()
    #     return base_model

    def build_model(self, **kwargs):  # step 6    ----> 运行到此地后， 该去向 何方
        input = self.input
        x = self.dense_net_to_seq(input, is_test=kwargs.get('is_test', False))
        if eval(self.is_rnn):
            fc_1 = tf.keras.layers.Dense(self.rnn_size * 3)(x)
            fc_2 = tf.keras.layers.Dense(self.rnn_size * 3)(x)

            gru_forward = tf.keras.layers.GRU(self.rnn_size, activation='relu', recurrent_activation='sigmoid',
                                              return_sequences=True, )(fc_1) # (batch,width/8,rnn_size)
            gru_backward = tf.keras.layers.GRU(self.rnn_size, activation='relu', recurrent_activation='sigmoid',
                                               go_backwards=True, return_sequences=True)(fc_2)# (batch,width/8,rnn_size)


            x = tf.keras.layers.Add()([gru_forward, gru_backward]) # (batch,width/8,rnn_size)


        nclass = self.charset.charset_size
        self.y_pred = tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)
        pre_decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])
        base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=pre_decoded)
        base_model.summary()
        return base_model




    def build_model_old(self, **kwargs): #  step 6    ----> 运行到此地后， 该去向 何方
        input = self.input
        self.y_pred = self.build_model_outputs(input, **kwargs)

        decoded = CtcDecodeLayer(name='out_idxes')([self.y_pred, self.input_length])


        base_model = tf.keras.Model(inputs=[input, self.input_length], outputs=decoded)
        print('hou --------begin print network paras......................')
        # 实验 把所有的层都给冻结 -------
        # for layer in base_model.layers:
        #     print('layer:',layer)
        #     layer.trainable = False

        base_model.summary()
        print('hou --------end print network paras......................')
        return base_model

    def build_train_model(self, **kwargs):
        return TFCtcTrainModel(self, **kwargs)

    def build_infer_model(self, **kwargs):
        return TFCtcInferModel(self, **kwargs)


class TFCtcTrainModel(TFTrainModel):
    def __init__(self, model_def, **kwargs):
        super().__init__(model_def, data_feeder_type=TFCtcDataFeeder, **kwargs)   # TFCtcDataFeeder为数据的类

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


    def init_model(self):
        self.base_model = self.model_def.build_model()     # 将直接运行 构建 densenet网络 + ctc 的编码

        input = self.model_def.input
        y_pred = self.model_def.y_pred

        input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')
        labels = tf.keras.Input(name='label', shape=[None], dtype='float32')
        label_length = tf.keras.Input(name='label_length', shape=[1], dtype='int64')

        loss_out = tf.keras.layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        self.train_model = tf.keras.Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
        # self.train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])  # 原先的 --- hou
        ########################   fix by hou
        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        #optimizer = tf.keras.optimizers.SGD()
        lr_metric = get_lr_metric(optimizer)
        self.train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=['accuracy',lr_metric])
        ########################


class TFCtcInferModel(TFInferModel):
    def __init__(self, model_def, **kwargs):
        super().__init__(model_def, data_feeder_type=TFCtcDataFeeder, **kwargs)
        self.total_count = 0
        self.error_count = 0
        self.only_out_error = self.args.get('only_out_error', False)
        self.from_train_model=True

    def get_feed_names(self):
        return ['input:0', 'input_length_1:0']

    def get_out_names(self):
        return ['out_idxes/SparseToDense:0']



    def build_infer_model(self):
        tf.keras.backend.set_learning_phase(1)
        input =  self.model_def.input
        input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')
        y_pred = self.model_def.build_model_outputs(input, is_test=True)
        decoded = CtcDecodeLayer(name='out_idxes')([y_pred, input_length])

        base_model = tf.keras.Model(inputs=[self.model_def.input, input_length], outputs=decoded)
        base_model.summary()

        h5_file = self.get_model_file_for_load(ext_name=".h5")
        assert os.path.exists(h5_file)
        base_model.load_weights(h5_file)
        pb_file = FileUtils.change_ext_name(h5_file, '.pb')
        self.model_def.convert_model_to_pb(base_model,pb_file)
        self.load_pb_model(pb_file)


    def on_test_begin(self):
        super().on_test_begin()
        self.total_count = 0
        self.error_count = 0

    def on_test_end(self, total_time_span):
        super().on_test_end(total_time_span)
        if self.total_count > 0:
            Logger.info('predict count:%d, error count:%d, acc:[%0.4f], avg time:[%0.4f]', self.total_count,
                        self.error_count,
                        (1 - self.error_count * 1.0 / self.total_count),
                        total_time_span / self.total_count)

    def compare_label(self, pred_label, y_label):
        remove_chars = self.args.get('remove_chars', '')
        if remove_chars != '':
            pred_label = LabelUtils.remove_label_chars(pred_label, remove_chars)
            y_label = LabelUtils.remove_label_chars(y_label, remove_chars)

        if pred_label != y_label:
            flag = 'X'
            self.error_count += 1
        else:
            flag = ''

        return pred_label, y_label, flag

    def do_test_batch(self, data):
        batch, ori_batch = data
        feed_dict = {}
        for name in batch:
            tensor = self.input_map[name]
            feed_dict[tensor] = batch[name]

        with self.sess.graph.as_default():
            y_pred_list = self.sess.run(self.outputs, feed_dict=feed_dict)

            pred_labels=[]
            for y in y_pred_list[:]:
                text = self.model_def.charset.decode(y)
                pred_labels.append(text)

            seq_no = self.total_count
            i = 0
            for pred_label in pred_labels:
                y_label = ori_batch[i]['label']
                pred_label, y_label, flag = self.compare_label(pred_label, y_label)

                seq_no += 1

                if self.only_out_error and flag == '':
                    pass
                else:
                    file_name = ori_batch[i].get('file_name', '%05d' % seq_no)
                    print('predict %s\tPRED:%s\t\t%s\t\tLABEL:%s' % (file_name, pred_label, flag, y_label))
                i += 1

            self.total_count += len(ori_batch)

