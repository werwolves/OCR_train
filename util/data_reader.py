# -*- coding: utf8 -*-

import _thread
import base64
import glob
import json
import os
import random
import time

import numpy as np

from util import Logger
from util.image_utils import ImageTools
from util.io_utils import FileUtils
from util.label_utils import LabelUtils


class BatchDataReader:
    """
    批次数据读取抽象类
    """

    def __init__(self, input_dir, batch_size, **kwargs):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.args = kwargs
        self.closed = False

    def fetch_batch(self):
        """
        获得一个批次数据
        """
        raise NotImplementedError()

    def close(self):
        self.closed = True

    def has_data(self):
        return True

class FolderScanner():
    """
    文件夹扫描工具类
    """

    def __init__(self, input_dir, filter_name, scan_period=0):
        self.glob_path = os.path.join(input_dir, filter_name)
        self.scan_period = scan_period
        self.items = None
        self.scanning = False
        # self.scan()
        if self.scan_period > 0:
            self.start()

    def scan(self):
        self.items = glob.glob(self.glob_path)

    def get_items(self, force_scan=False):
        items = self.items
        if force_scan or items is None:
            self.scan()
        return self.items

    def _thread_scan(self):
        while (self.scanning):
            self.scan()
            time.sleep(self.scan_period)

    def start(self):
        if self.scanning:
            return
        self.scanning = True
        _thread.start_new_thread(self._thread_scan, ())

    def close(self):
        self.scanning = False


class FolderBatchReader(BatchDataReader):
    """
    从文件夹读取批次数据的基础类
    """

    def __init__(self, data_loader_type, input_dir, filter_name, batch_size, **kwargs):
        super().__init__(input_dir, batch_size, **kwargs)
        self.data_loader_type = data_loader_type

        # 是否将所满足条件的文件或目录合并为一个结果
        self.merge_items = kwargs.get('merge_items', False)

        scan_period = 0 if self.merge_items else kwargs.get('scan_period', 0)
        self.scanner = FolderScanner(input_dir, filter_name, scan_period)
        self.rand_fetch = kwargs.get('rand_fetch', False)

        self.curr_index = -1
        self.batch_size = batch_size
        self.min_batch_size = kwargs.get('min_batch_size', 1)
        self.data_loader = None

    def has_data(self):
        loader = self._get_data_loader()
        if loader is None:
            return False

        return loader.has_data()

    def _get_data_loader(self):
        if self.data_loader is None or self.data_loader.eof:
            self.data_loader = self._next_data_loader()

        return self.data_loader

    def _next_data_loader(self):
        items = self.scanner.get_items(force_scan=self.merge_items)
        if len(items) == 0:
            return None

        if self.rand_fetch:
            # 随机训练模式
            if self.merge_items:
                # 合并结果
                ########################################################
                # print('load item count: %d' % len(items)) # raw
                if len(items)>0:
                    dir_name = os.path.basename(os.path.dirname(items[0]))
                    if 'train' in dir_name:
                        print('load traindataset count: %d' % len(items))
                    elif 'val' in dir_name:
                        print('load valdataset count: %d' % len(items))
                    else:
                        print('find unknown datasets....%s'%(dir_name))

                ########################################################
                #items= np.array(items)
                np.random.shuffle(items)
                return self.data_loader_type(items, **self.args)
            else:
                # 随机循环读取
                item = random.choice(items)
                return self.data_loader_type(item, **self.args)

        # 顺序读取
        if self.merge_items:
            # 合并方式，仅读一次
            if self.data_loader is None:
                items = np.sort(items).tolist()
                return self.data_loader_type(items, **self.args)
            else:
                return None
        else:
            # 逐个读取
            index = self.curr_index + 1
            if index >= len(items):
                return None
            item = items[index]
            self.curr_index = index
            return self.data_loader_type(item, **self.args)

    def fetch_batch(self):
        result_batch = []
        fetch_size = self.batch_size
        while True:
            loader = self._get_data_loader()
            if loader is None:
                # 读取结束
                break

            rows = loader.fetch_next_rows(fetch_size)
            if rows is None:
                continue

            result_batch += rows
            if len(result_batch) >= self.min_batch_size:
                break

            # 不足1批，继续补齐
            fetch_size = self.min_batch_size - len(result_batch)

        if len(result_batch) == 0:
            return None
        else:
            return result_batch

    def close(self):
        super().close()
        self.scanner.close()


class DataRowLoader:
    """
    记录加载的抽象基类
    """

    def __init__(self, **kwargs):
        self.args = kwargs
        self.eof = False
        self.row_index = 0

    def fetch_next_rows(self, size):
        rows = self.do_load_rows(self.row_index, size)
        if rows is None:
            self.close()
            return None

        fetch_size = len(rows)
        self.row_index += fetch_size
        if fetch_size < size:
            self.close()

        return rows

    def do_load_rows(self, begin, size):
        raise NotImplementedError()

    def close(self):
        self.eof = True

    def try_fetch_from_rows(self, rows, begin, size):
        end = begin + size
        max_size = len(rows)
        if end > max_size:
            end = max_size
        if begin >= end:
            return []
        return rows[begin:end]

    def has_data(self):
        return True

class FileDataRowLoader(DataRowLoader):
    """
    文件记录加载的基础抽象类
    """

    def __init__(self, file, **kwargs):
        super().__init__(**kwargs)
        self.file = file
        self.file_rows = None
        self.rand_fetch = kwargs.get('rand_fetch', True)

    def reload_file(self):
        self.row_index = 0
        self.file_rows = self.do_load_file(self.file)
        if self.rand_fetch:
            np.random.shuffle(self.file_rows)

    def do_load_rows(self, begin, size):
        if self.file_rows is None:
            self.reload_file()

        return self.try_fetch_from_rows(self.file_rows, begin, size)

    def do_load_file(self, file):
        raise NotImplementedError()

    def has_data(self):
        if self.file_rows is None:
            self.reload_file()
        return len(self.file_rows) > 0


class TxtFileRowLoader(FileDataRowLoader):
    """
    基于多行文本文件数据加载的基础类
    """

    def __init__(self, file, **kwargs):
        self.split_char = kwargs.get('split_char', '\007')
        self.fixed_label_len = kwargs.get('fixed_label_len', 0)
        super().__init__(file, **kwargs)

    def do_process_line(self, line):
        raise NotImplementedError()

    def process_line(self, rows, line):
        try:
            row = self.do_process_line(line)
            if row is None:
                return False

            rows.append(row)
            return True
        except Exception as e:
            Logger.warning(e)
            return False

    def do_load_file(self, file):
        rows = []
        try:
            i = 0
            with open(file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()

            Logger.debug('load file:%s lines:%d' % (file, len(lines)))
            if self.rand_fetch:
                np.random.shuffle(lines)

            for line in lines:
                i += 1
                if not line:
                    continue
                if not self.process_line(rows, line):
                    Logger.warning('failed to process line:%d,file:%s' % (i, file))

        except Exception as e:
            Logger.warning(e)

        return rows


class B64ImageRowLoader(TxtFileRowLoader):
    """
    以Base64格式存储的文本文件加载实现
    """

    def __init__(self, folder, **kwargs):
        self.to_gray = kwargs.get('channels', 1) == 1
        super().__init__(folder, **kwargs)

    def do_process_line(self, line):
        if self.fixed_label_len > 0:
            code = line[0:self.fixed_label_len]
            b64 = line[self.fixed_label_len + len(self.split_char):]
        else:
            texts = line.split(self.split_char)
            if len(texts) < 2:
                return False
            code = texts[0]
            b64 = texts[1]

        if not code or not b64:
            return False

        im_bytes = base64.b64decode(b64)
        im = ImageTools.open_from_bytes(im_bytes, to_gray=self.to_gray)
        return {'im': im, 'label': code}


class B64ImageJsonRowLoader(TxtFileRowLoader):
    """
    以Base64格式存储的Json文本文件加载实现
    """

    def __init__(self, folder, **kwargs):
        self.to_gray = kwargs.get('channels', 1) == 1
        super().__init__(folder, **kwargs)

    def do_process_line(self, line):
        texts = line.split(self.split_char)
        if len(texts) < 2:
            return False
        text = texts[0]
        b64 = texts[1]

        if not text or not b64:
            return False

        data = json.loads(text)
        im_bytes = base64.b64decode(b64)
        im = ImageTools.open_from_bytes(im_bytes, to_gray=self.to_gray)
        return {"im": im, "data": data}


class FolderDataRowLoader(DataRowLoader):
    """
    基于文件夹的数据加载基础类（1个文件表示一条记录）
    """

    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
        self.file_filter = kwargs.get('file_filter', '*.txt')
        self.rand_fetch = kwargs.get('rand_fetch', True)
        self.files = folder if isinstance(folder, list) else None

    def reload_folder(self):
        self.row_index = 0
        files = glob.glob(os.path.join(self.folder, self.file_filter))
        if self.rand_fetch:
            np.random.shuffle(files)
        else:
            files = np.sort(files).tolist()

        Logger.info('load %d files from path %s ...' % (len(files), self.folder))
        self.files = files

    def read_file_to_rows(self, rows, file):
        try:
            row = self.process_file(file)
            if row is None:
                return False

            rows.append(row)
            return True
        except Exception as e:
            Logger.warning(e)
            return False

    def do_load_rows(self, begin, size):
        if self.files is None:
            self.reload_folder()

        files = self.try_fetch_from_rows(self.files, begin, size)
        if len(files) == 0:
            return None

        rows = []
        for file in files:
            self.read_file_to_rows(rows, file)

        return rows

    def process_file(self, file):
        raise NotImplementedError()

    def has_data(self):
        if self.files is None:
            self.reload_folder()
        return len(self.files) > 0


class ImageFolderDataLoader(FolderDataRowLoader):
    """
    影像文件夹数据加载实现（一个文件对应一个影像文件）
    """

    def __init__(self, folder, **kwargs):
        self.to_gray = kwargs.get('channels', 1) == 1
        super().__init__(folder, **kwargs)

    def process_file(self, file):
        img_file = LabelUtils.try_get_image_file(file)
        if img_file is None:
            return

        label = LabelUtils.get_label_from_txt_file(file)
        im = ImageTools.open_from_file(img_file, self.to_gray)

        item = {'im': im, 'label': label, 'file_name': os.path.basename(img_file)}
        return item


class LineTextDataLoader(TxtFileRowLoader):
    def __init__(self, folder, **kwargs):
        super().__init__(folder, **kwargs)

    def do_process_line(self, line):
        items=line.split('\t')
        return {'text': items[-1], 'label': items[0:-1]}


def create_data_reader(type, **kwargs):
    batch_size = kwargs.pop('batch_size', 10)
    if type.startswith("folder-"):
        input_dir = kwargs.pop('input_dir')
        filter_name = kwargs.pop('filter_name')

        if type == 'folder-image':
            kwargs.pop('scan_period')
            filter_name += "/" + kwargs.get('file_filter', '*.txt')
            return FolderBatchReader(ImageFolderDataLoader, input_dir, filter_name, batch_size, merge_items=True,
                                     **kwargs)

        if type == 'folder-b64':
            filter_name += "/*.txt"
            return FolderBatchReader(B64ImageRowLoader, input_dir, filter_name, batch_size, **kwargs)

        if type == 'folder-b64-json':
            filter_name += "/*.txt"
            return FolderBatchReader(B64ImageJsonRowLoader, input_dir, filter_name, batch_size, **kwargs)

        if type == 'folder-text-line':
            filter_name += "/" + kwargs.get('file_filter', '*.txt')
            return FolderBatchReader(LineTextDataLoader, input_dir, filter_name, batch_size, **kwargs)

    raise Exception("can not create dataset of type:" + type)
