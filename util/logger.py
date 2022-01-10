# -*- coding: utf8 -*-
import logging
import logging.config
import os

from util.io_utils import FileUtils


class Logger:
    @staticmethod
    def init_logger(conf_file='./conf/logging.ini'):
        FileUtils.make_dir(os.path.join("./log"))
        logging.config.fileConfig(conf_file)

    @staticmethod
    def debug(*args, **kwargs):
        logging.debug(*args, **kwargs)

    def info(*args, **kwargs):
        logging.info(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        logging.warning(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        logging.error(*args, **kwargs)
