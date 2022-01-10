# -*- coding: utf8 -*-
import os
import shutil
import glob


class FileUtils:
    @staticmethod
    def make_dir(dir_path, clear=False):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        elif clear:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

    @staticmethod
    def change_ext_name(file_path, ext_name):
        base_path, file_name = os.path.split(file_path)
        name, _ = os.path.splitext(file_name)
        if len(ext_name) > 0 and ext_name[0:1] != '.':
            ext_name = '.' + ext_name

        return os.path.join(base_path, '%s%s' % (name, ext_name))

    @staticmethod
    def make_parent_path(file_path):
        folder = os.path.dirname(file_path)
        FileUtils.make_dir(folder)

    @staticmethod
    def read_text_from_file(txt_file, encoding='utf-8', strip=True):
        with open(txt_file, encoding=encoding) as f:
            text = f.read()
            if strip:
                text = text.strip()
            return text

    @staticmethod
    def read_lines_from_file(txt_file, encoding='utf-8', strip=True):
        with open(txt_file, encoding=encoding) as f:
            lines = f.readlines()
            if strip:
                lines = [x.strip() for x in lines]
            return lines

    @staticmethod
    def get_base_name(file_path):
        if file_path:
            base_path, file_name = os.path.split(file_path)
            name = os.path.splitext(file_name)[0]
            return name
        return ''

    @staticmethod
    def get_ext_name(file_path):
        base_path, file_name = os.path.split(file_path)
        _, ext_name = os.path.splitext(file_name)
        return ext_name

    @staticmethod
    def get_file_name(model_path):
        if model_path:
            base_path, file_name = os.path.split(model_path)
            return file_name
        return ''

    @staticmethod
    def get_file_write_time(file):
        return os.path.getmtime(file)

    @staticmethod
    def get_file_by_max_write_time(dir_name, filter):
        files = glob.glob(os.path.join(dir_name,filter))
        result_file=None
        max_ts = 0
        for file in files:
            ts=FileUtils.get_file_write_time(file)
            if ts > max_ts:
                result_file = file
                max_ts = ts

        return result_file

    @staticmethod
    def read_bytes( file):
        with open(file,'rb') as f:
            return f.read()