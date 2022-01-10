# -*- coding: utf8 -*-
import argparse
import distutils.util
import random
import time
import six

from util.logger import Logger


class ArgUtils:
    @staticmethod
    def add_arguments(parser, argname, type, default, help='', **kwargs):
        """Add argparse's argument.
        Usage:
        .. code-block:: python
            parser = argparse.ArgumentParser()
            add_argument("name", str, "Jonh", "User name.", parser)
            args = parser.parse_args()
        """
        type = distutils.util.strtobool if type == bool else type
        parser.add_argument(
            "--" + argname,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)

    @staticmethod
    def print_arguments(args):
        """Print argparse's arguments.
        Usage:
        .. code-block:: python
            parser = argparse.ArgumentParser()
            parser.add_argument("name", default="Jonh", type=str, help="User name.")
            args = parser.parse_args()
            print_arguments(args)
        :param args: Input argparse.Namespace for printing.
        :type args: argparse.Namespace
        """
        print("-----------  Configuration Arguments -----------")
        for arg, value in sorted(six.iteritems(vars(args))):
            print("%s: %s" % (arg, value))
        print("------------------------------------------------")


    @staticmethod
    def extend_args(dst, **kwargs):
        for k,v in kwargs.items():
            if v is None:
                continue
            dst[k]=v
        return dst


class BaseApp:
    _parser = argparse.ArgumentParser(description=__doc__)

    def __init__(self):
        random.seed(time.time())
        Logger.init_logger()
        self.args = {}

    def _parse_args(self):
        self.args = BaseApp._parser.parse_args()
        ArgUtils.print_arguments(self.args)

    @staticmethod
    def define_str_arg(name, default, help='', **kwargs):
        ArgUtils.add_arguments(BaseApp._parser, name, str, default, help, **kwargs)

    @staticmethod
    def define_int_arg(name, default, help='', **kwargs):
        ArgUtils.add_arguments(BaseApp._parser, name, int, default, help, **kwargs)

    @staticmethod
    def define_bool_arg(name, default, help='', **kwargs):
        ArgUtils.add_arguments(BaseApp._parser, name, bool, default, help, **kwargs)

    def on_load(self):
        pass

    def on_unload(self):
        pass

    def do_run(self):
        pass

    def run(self):
        self._parse_args()      # 接收参数，并且把参数打印出来
        self.on_load()          # noting to do
        self.do_run()           # important  section   -----> 又回到 train.py 文件夹下 do_run() 中
        self.on_unload()
