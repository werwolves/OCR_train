# -*- coding: utf8 -*-
import numpy as np
from keras_preprocessing.text import Tokenizer

from util.data_feeder import DataFeeder
import re

class NlpClassifyDataFeeder(DataFeeder):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.charset = self.model.model_def.charset
        self.label_dict = self.model.model_def.label_dict
        # self.tokenizer = Tokenizer(char_level=True)
        # self.tokenizer.fit_on_texts(''.join(self.charset.charset))

    def process_batch(self, batch, is_test):
        size = len(batch)
        w = self.model.model_def.input_width
        batch_x = np.zeros((size, w))
        batch_y = np.zeros((size, self.label_dict.label_count()))

        i = 0
        for x in batch:
            text = x['text']
            label = int(x['label'][0])

            text = re.sub('[\x00-\xff]+', '', text)
            # text = re.sub('[0-9]', '0', text)

            batch_x[i, :] =   self.charset.encode_to_seq(text, w) #/ self.charset.charset_size * 1.0
            batch_y[i, label] = 1.0

            i+=1

        if is_test:
            return {"input": batch_x}
        else:
            return (batch_x, batch_y)
