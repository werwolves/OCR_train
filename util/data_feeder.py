# -*- coding: utf8 -*-

import time

from util import ArgUtils
from util.data_reader import create_data_reader

# noinspection PyMethodMayBeStatic
from util.queue_utils import GeneratorEnqueuer


class DataFeeder:
    def __init__(self, model, **kwargs):
        self.model = model
        self.args = ArgUtils.extend_args(model.args.copy(), **kwargs)

        self.input_path = self.args.get('input_path')
        self.input_format = self.args.get('input_format', 'folder-b64')
        self.batch_size = self.args.get('batch_size', 32)
        self.channels = self.model.model_def.input_channels
        self.enqueuer = None

    def process_batch(self, batch, is_test):
        raise NotImplementedError()

    def infinite_reader(self, reader):
        while True:
            batch = reader.fetch_batch()
            if batch is None:
                time.sleep(1)
                continue
            yield self.process_batch(batch, False)

    def gen_train_batch(self):
        """
        训练批次数据迭代生成器
        :return:
        """
        reader = create_data_reader(self.input_format,
                                    input_dir=self.input_path,
                                    filter_name=self.args.get('filter_train_name', "train-*"),
                                    batch_size=self.batch_size,
                                    min_batch_size=self.batch_size,
                                    channels=self.channels,
                                    rand_fetch=True,
                                    scan_period=15)
        use_multi_process = self.args.get('multi_process', "False")
        if use_multi_process == "False":  # 不使用多进程
            print('not multi_process','not multi_process')
            while True:
                batch = reader.fetch_batch()
                if batch is None:
                    time.sleep(1)
                    continue
                yield self.process_batch(batch, False)

        else: # 使用多进程
            print('multi_process', 'multi_process')
            if self.enqueuer is None:
                self.enqueuer = GeneratorEnqueuer(
                    self.infinite_reader(reader), use_multiprocessing=True)
                # self.enqueuer.start(max_queue_size=10 * self.batch_size, workers=2) # raw
                self.enqueuer.start(max_queue_size=1 * self.batch_size, workers=1)

        # if not self.args.get('multi_process', False):
        #     print('not multi_process','not multi_process')
        #     while True:
        #         batch = reader.fetch_batch()
        #         if batch is None:
        #             time.sleep(1)
        #             continue
        #         yield self.process_batch(batch, False)
        #
        # else:
        #     print('not multi_process', 'not multi_process')
        #     if self.enqueuer is None:
        #         self.enqueuer = GeneratorEnqueuer(
        #             self.infinite_reader(reader), use_multiprocessing=True)
        #         # self.enqueuer.start(max_queue_size=10 * self.batch_size, workers=2) # raw
        #         self.enqueuer.start(max_queue_size=1 * self.batch_size, workers=1)

            generator_out = None
            while True:
                while self.enqueuer.is_running():
                    if not self.enqueuer.queue.empty():
                        generator_out = self.enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.01)
                yield generator_out

    def gen_test_batch(self):
        reader = create_data_reader(self.input_format,
                                    input_dir=self.input_path,
                                    filter_name=self.args.get('filter_test_name', "test-*"),
                                    file_filter=self.args.get('file_filter', "*.txt"),
                                    batch_size=self.batch_size,
                                    channels=self.channels,
                                    rand_fetch=False,
                                    scan_period=0)
        while True:
            batch = reader.fetch_batch()
            if batch is None:
                break

            yield self.process_batch(batch, True)

    def create_val_reader(self):
        return create_data_reader(self.input_format,
                                    input_dir=self.input_path,
                                    filter_name=self.args.get('filter_val_name', "val-*"),
                                    batch_size=self.batch_size,
                                    min_batch_size=self.batch_size,
                                    channels=self.channels,
                                    rand_fetch=True,
                                    scan_period=0)

    def has_val_batch_data(self):
        reader = self.create_val_reader()
        return reader.has_data()

    def gen_val_batch(self):
        reader = self.create_val_reader()
        while True:
            batch = reader.fetch_batch()
            if batch is None:
                time.sleep(1)
                continue

            yield self.process_batch(batch, False)

#
# class DataFeeder2:
#     def __init__(self, model, ds, **kwargs):
#         self.model = model
#         self.args = kwargs
#         self.ds = ds
#         self.feed_queue = kwargs.get('feed_queue', 10)
#         self.queue_train = Queue(maxsize=self.feed_queue)
#         self.queue_test = Queue(maxsize=self.feed_queue)
#
#         _thread.start_new_thread(self._thread_load_to_queue, ())
#
#     def _thread_load_to_queue(self):
#         while True:
#             try:
#                 ok = False
#                 if self.ds.exists_train_data():
#                     ok |= self._feed_to_queue(False)
#                 if self.ds.exists_test_data():
#                     ok |= self._feed_to_queue(True)
#
#                 if not ok:
#                     time.sleep(2)
#             except BaseException as e:
#                 Logger.warning(e, exc_info=True)
#
#     def _feed_to_queue(self, is_test=False):
#         queue = self.queue_test if is_test else self.queue_train
#         if queue.full():
#             return False
#         batch = self.feed_batch_from_ds(is_test)
#         if batch is None:
#             return False
#         queue.put(batch)
#         return True
#
#     def feed_batch(self, is_test=False):
#         return self.feed_for_test() if is_test else self.feed_for_train()
#
#     def feed_batch_from_ds(self, is_test=False):
#         return self.feed_from_ds_for_test() if is_test else self.feed_from_ds_for_train()
#
#     def feed_for_test(self):
#         try:
#             return self.queue_test.get(timeout=5)
#         except Exception as e:
#             return None
#
#     def feed_for_train(self):
#         if self.queue_train.empty():
#             return None
#         return self.queue_train.get()
#
#     def feed_from_ds_for_test(self):
#         batch = self.ds.feed_test_batch(False)
#         if batch is None:
#             return None
#         return self.process_batch(batch, True)
#
#     def feed_from_ds_for_train(self):
#         batch = self.ds.feed_train_batch()
#         if batch is None:
#             return None
#         return self.process_batch(batch, False)
#
#     def process_batch(self, batch, is_test):
#         return (batch, batch)
