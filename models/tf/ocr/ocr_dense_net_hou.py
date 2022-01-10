# -*- coding: utf8 -*-

import tensorflow as tf

# from models.tf.ocr import TFCtcOcrModelDef

K = tf.keras.backend


class TFOcrDenseNetModelDef():
    def __init__(self):
        # super().__init__(name, **kwargs)
        self.dense_block_layers =[8, 8, 8, 8]
        self.dense_block_growth_rate =  8
        # self.first_conv_filters = self.args.get('first_conv_filters', 64)
        self.first_conv_filters = 64

        self.first_conv_size =  5
        self.first_conv_stride =  2

        self.first_pool_size = 0
        self.first_pool_stride =  2

        self.last_conv_size = 0
        self.last_conv_filters = 0
        self.last_pool_size = 2

        self.trans_block_filters = 128

    @classmethod
    def conv_block(cls, input, growth_rate, dropout_rate=None, weight_decay=1e-4, padding='same', is_test=False):
        x = input
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding=padding)(x)
        if (dropout_rate and not is_test):
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    @classmethod
    def dense_block(cls, x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4, is_test=False):
        for i in range(nb_layers):
            cb = cls.conv_block(x, growth_rate, droput_rate, weight_decay, is_test=is_test)
            x = tf.keras.layers.Concatenate()([x, cb])
            # x = K.concatenate([x, cb], axis=-1)
            nb_filter += growth_rate
        return x, nb_filter

    @classmethod
    def transition_block(cls, input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4, is_test=False):
        x = input
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

        if (dropout_rate and not is_test):
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if (pooltype == 2):
            x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
        elif (pooltype == 1):
            x = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(x)
            x = tf.keras.layers.AveragePooling2D((2, 2), strides=2)(x)
        elif (pooltype == 3):
            x = tf.keras.layers.AveragePooling2D((2, 2), strides=2)(x)
        return x, nb_filter

    def attention_normal(self, input):
        a = tf.keras.layers.Permute((2, 1, 3), name="permute_first")(input)

        attention_ratio = 64 if self.input_height > 64 else self.input_height
        a = tf.keras.layers.Dense(attention_ratio, activation='softmax')(a)
        attention_probs = tf.keras.layers.Permute((2, 1, 3), name='attention_vec')(a)
        s = tf.keras.layers.multiply([input, attention_probs], name='attention_mul')
        # ATTENTION PART FINISHES HERE
        return s

    def dense_net_to_seq(self, input, is_test=False):
        x = input  # shape=(4, 29, 228, 1)
        _dropout_rate = 0.2
        _weight_decay = 1e-4

        nb_filter = self.first_conv_filters

        # 首次卷积
        x = tf.keras.layers.Conv2D(nb_filter, self.first_conv_size, strides=self.first_conv_stride, padding='same', # shape=(4, 15, 114, 64)  宽高缩小2倍
                                   use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(_weight_decay))(x)
        if self.first_pool_size:  # 不经过该分支
            # 首次池化
            x = tf.keras.layers.AveragePooling2D(self.first_pool_size, strides=self.first_pool_stride)(x)

        # begin dense net block
        nb_layers = self.dense_block_layers          # [8,8,8]
        growth_rate = self.dense_block_growth_rate   # 8

        for n_layer in nb_layers[:-1]:      # 跳出for循环的时--- shape=(4, 3, 28, 128)
            x, nb_filter = self.dense_block(x, n_layer, nb_filter, growth_rate, None, _weight_decay, is_test=is_test)

            trans_block_filters = self.trans_block_filters or nb_filter // 2
            x, nb_filter = self.transition_block(x, trans_block_filters, _dropout_rate, 2, _weight_decay,  # 宽高各缩小2倍
                                                 is_test=is_test)
        x, nb_filter = self.dense_block(x, nb_layers[-1], nb_filter, growth_rate, None, _weight_decay, is_test=is_test) # shape=(4, 3, 28, 192)

        if self.last_conv_size:
            # 最后一次卷积池化
            conv_filters = self.last_conv_filters or nb_filter
            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
            x = tf.keras.layers.Conv2D(conv_filters, self.last_conv_size, kernel_initializer='he_normal',
                                       padding='same',
                                       use_bias=False,
                                       kernel_regularizer=tf.keras.regularizers.l2(_weight_decay))(x)

            x = tf.keras.layers.AveragePooling2D(self.last_pool_size, strides=2)(x)

        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = tf.keras.layers.Activation('relu')(x)

        #x = tf.keras.layers.Permute((2, 1, 3), name='permute')(x)                          # shape=(4, 28, 3, 192)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(), name='flatten')(x)  # shape=(4, 28, 576)

        return x

    def build_input_tensor(self):
        return tf.keras.Input([self.input_height, None, self.input_channels], name='input')               # 原版
        # return tf.keras.Input([self.input_height, self.input_width, self.input_channels],batch_size=4, name='input')   # fix by hou, for debug

    def build_model_outputs(self, input, **kwargs):
        x = self.dense_net_to_seq(input, is_test=kwargs.get('is_test', False))
        nclass = self.charset.charset_size
        x = tf.keras.layers.Dropout(0.5)(x)

        y_pred = tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)
        return y_pred


class TFOcrDenseNetLstmModelDef(TFOcrDenseNetModelDef):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.rnn_size = self.args.get('rnn_size', 200)

    def build_model_outputs(self, input, **kwargs):
        nclass = self.charset.charset_size

        x = self.dense_net_to_seq(input)

        fc_1 = tf.keras.layers.Dense(self.rnn_size * 3)(x)
        fc_2 = tf.keras.layers.Dense(self.rnn_size * 3)(x)

        gru_forward = tf.keras.layers.GRU(self.rnn_size, activation='relu', recurrent_activation='sigmoid',
                                          return_sequences=True, )(fc_1)
        gru_backward = tf.keras.layers.GRU(self.rnn_size, activation='relu', recurrent_activation='sigmoid',
                                           go_backwards=True, return_sequences=True)(fc_2)


        x = tf.keras.layers.Add()([gru_forward, gru_backward])
        y_pred = tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)

        # x1 = tf.keras.layers.Dense(nclass)(gru_forward)
        # x2 = tf.keras.layers.Dense(nclass)(gru_backward)
        # x = tf.keras.layers.Concatenate()([x1, x2])
        # y_pred=tf.keras.layers.Dense(nclass, name='out', activation='softmax')(x)

        #y_pred = tf.keras.activations.softmax(x)

        return y_pred


if __name__ == '__main__':
    base_model = TFOcrDenseNetModelDef()
    input = tf.keras.Input([36, 285, 3], batch_size=4, name='input')
    x = base_model.dense_net_to_seq(input, is_test=False)
    print(x)