import numpy as np
from tensorflow.contrib import crf
from tensorflow.python.keras.callbacks import LearningRateScheduler

from models.tf import TFModelDef, TFTrainModel
from tensorflow import keras
import tensorflow as tf

from models.tf.keras_self_attention import SeqSelfAttention
from models.tf.nlp.feeder import NlpClassifyDataFeeder
from util.label_utils import LabelUtils, Charset


class TextClassifyAttention(TFModelDef):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.charset = Charset(charset_file=kwargs.get('charset_file', None),
                               charset_text=kwargs.get('charset_text', None), undef_char='\001')

        self.label_dict = LabelUtils.load_label_dict(kwargs.get('label_file'))
        self.label_count = self.label_dict.label_count()

    def data_shape(self):
        return (None,)

    def build_model(self, input, **kwargs):
        # model.add(input)
        x = keras.layers.Embedding(input_length=self.input_width,
                                   input_dim=self.charset.charset_size,
                                   output_dim=16,
                                   mask_zero=False)(input)
        # x = SeqSelfAttention(64, attention_activation='sigmoid')(x)
        # x= keras.layers.SpatialDropout1D(0.2)(x)
        x = keras.layers.Conv1D(32, 3, use_bias=False, strides=1)(x)
        x = keras.layers.AvgPool1D(2)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv1D(64, 3, use_bias=False)(x)
        x = keras.layers.AvgPool1D(2)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)



        # x = keras.layers.Flatten()(x)
        #
        # x = keras.layers.Bidirectional(keras.layers.GRU(units=128,
        #                                                 return_sequences=True))(x)


        x = keras.layers.GlobalAvgPool1D()(x)
        output = keras.layers.Dense(self.label_count, activation='softmax')(x)

        return tf.keras.Model(inputs=input, outputs=output)

    def build_train_model(self, **kwargs):
        return KerasTrainModel(self, **kwargs)


class ModelDefCheckpoint(keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_train_model(epoch=epoch)
        self.model.save_infer_model(epoch=epoch)



class KerasTrainModel(TFTrainModel):
    def __init__(self, model_def, **kwargs):
        super().__init__(model_def, data_feeder_type=NlpClassifyDataFeeder, **kwargs)
        self.model = None
        self.val_feed = NlpClassifyDataFeeder(self, is_test=False, filter_train_name="val*", batch_size=500)

    def do_run(self):
        gen_train = self.data_feeder.gen_train_batch()
        gen_val = self.val_feed.gen_val_batch()
        self.model.fit_generator(gen_train,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_data=gen_val,
                                 validation_steps=self.steps_per_epoch,
                                 epochs=self.epochs, callbacks=self.get_train_callbacks())

    def get_train_callbacks(self):
        lr_schedule = lambda epoch: 0.001 * 0.4 ** epoch
        learning_rate = np.array([lr_schedule(i) for i in range(10)])
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))

        return [change_lr]

    def init_model(self):
        self.model = self.model_def.build_model(self.model_def.input)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()
