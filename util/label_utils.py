# -*- coding: utf8 -*-
import os
import re
import glob
import numpy as np
import os
from util import Logger
from util.io_utils import FileUtils


class Charset:
    # UNDEF_CHAR = u' '
    UNDEF_CHAR = u'卍' # 尝试采用该字符作为特殊字符分割图片，把该字从 full_7655 从字符集中去掉， 同时增加一个空格作为新的字符集

    def __init__(self, charset_file=None, encoding='utf-8', replace_dict_file='', charset_text=None,
                 undef_char=UNDEF_CHAR):
        self.name = FileUtils.get_base_name(charset_file)
        self.undef_char = undef_char
        self.charset_file = charset_file
        self.replace_dict_file = replace_dict_file
        self.encoding = encoding
        self.charset = []
        self.exclude_charset = set()
        self.replace_dict = LabelUtils.read_replace_dict(self.replace_dict_file)
        self.load_charset(charset_file, charset_text)
        self.charset_size = len(self.ch_dict)

        Logger.info("load charset from %s, count=%d" % (charset_file, self.charset_size))

    def load_charset(self, charset_file=None, charset_text=None):
        if not charset_file is None:
            self.charset = CharsetTools.read_text_from_file(charset_file, self.encoding) + self.undef_char
        elif not charset_text is None:
            self.charset = charset_text
        else:
            raise Exception("invalid charset")

        self._convert_to_dict()

    def _convert_to_dict(self, check_dup=True):
        self.ch_dict = {}
        i = 0
        new_charset = ""
        for ch in self.charset:
            if check_dup and ch in self.ch_dict:
                Logger.debug("dup char:%s" % (ch))
                print('find rechar:',ch)
                continue
            new_charset += ch
            self.ch_dict[ch] = i
            i += 1
        self.charset = new_charset

    def try_replace_from_dict(self, text):
        if not self.replace_dict:
            return text

        return LabelUtils.try_replace_from_dict(text, self.replace_dict)

    def clean_text_label(self, text):
        '''
        清理文本标签
        :param text:
        :return:
        '''

        text = self.try_replace_from_dict(text)
        label = ''
        for ch in text:
            if ch in self.exclude_charset:
                label += self.undef_char
            elif ch in self.ch_dict:
                label += ch
            # elif ch == ' ':
            elif ch == '卍':
                label += self.undef_char
            else:
                label += self.undef_char

        return label

    def encode_to_index(self, text):
        '''
        文本转向量
        :param text:
        :param max_len:
        :param def_fill_val:
        :return:
        '''
        label = self.clean_text_label(text)
        return [self.ch_dict[x] for x in label]

    def encode_to_seq(self, text, max_len):
        '''
        文本转向量
        :param text:
        :param max_len:
        :param def_fill_val:
        :return:
        '''
        code = np.zeros((max_len),dtype='int32')

        label = self.clean_text_label(text)
        n = 0
        for ch in label:
            code[n] = self.ch_dict[ch]
            n += 1
            if n>=max_len:
                # print('max len=%d, text=%s' %(max_len, text))
                break

        return code

    def encode(self, text, max_len, def_fill_val=100000.0):
        '''
        文本转向量
        :param text:
        :param max_len:
        :param def_fill_val:
        :return:
        '''
        code = np.ones((max_len),dtype=np.float32) * def_fill_val

        label = self.clean_text_label(text)
        n = 0
        for ch in label:
            code[n] = self.ch_dict[ch]
            n += 1

        return code

    def decode(self, vec):
        '''
        向量转文本
        :param vec:
        :return:
        '''
        text = ""
        for y in vec:
            i = int(y)
            if i >= 0 and i < self.charset_size:
                text += self.charset[i]

        return text.strip()


class CharsetTools:
    @staticmethod
    def read_text_from_file(txt_file, encoding='utf-8'):
        return FileUtils.read_text_from_file(txt_file, encoding, strip=True)

    @staticmethod
    def read_lines_from_file(txt_file, encoding='utf-8'):
        with open(txt_file, encoding=encoding) as f:
            return f.readlines()

    @staticmethod
    def read_charset_from_file(txt_file, encoding='utf-8'):
        text = CharsetTools.read_text_from_file(txt_file, encoding=encoding)
        c = set(text)
        return CharsetTools._clean_charset(c)

    @staticmethod
    def _clean_charset(charset):
        for c in '\t\r\n':
            if c in charset:
                charset.remove(c)
        return charset

    @staticmethod
    def read_charset_from_folder(folder, filter="*.txt", recursive=True, encoding='utf-8'):
        pattern = os.path.join(folder, filter)
        files = glob.glob(pattern, recursive=recursive)
        s = set()
        for file in files:
            text = CharsetTools.read_text_from_file(file, encoding=encoding)
            c = set(text)
            s = s | c

        return CharsetTools._clean_charset(s)

    @staticmethod
    def write_charset_to_file(txt_file, charset, encoding='utf-8'):
        s = list(charset) if isinstance(charset, set) else charset
        s.sort()
        with open(txt_file, 'w', encoding=encoding) as f:
            f.write(''.join(s))
            f.flush()
        return len(s)


class LabelDict:
    def __init__(self, dict_file_or_lines):
        if type(dict_file_or_lines) is list:
            self.label_list = dict_file_or_lines
        else:
            self.label_list = LabelUtils.read_label_list(dict_file_or_lines)

        self.unknown_label = "UNKNOWN"
        self.label_list += [self.unknown_label]
        self.unknown_label_index = len(self.label_list)-1
        self.label_map = LabelUtils.map_list_to_index(self.label_list)

    def label_count(self):
        return len(self.label_list)

    def encode_to_index(self, label):
        return self.label_map.get(label, self.unknown_label_index)

    def decode_label(self, index):
        if 0 < index < len(self.label_list):
            return self.label_list[index]
        return self.unknown_label


class LabelUtils:
    @staticmethod
    def map_list_to_index(label_list):
        m = {}
        i = 1
        for label in label_list:
            m[label] = i
            i += 1
        return m

    @staticmethod
    def read_label_list(txt_file):
        lines = FileUtils.read_lines_from_file(txt_file)
        label_list = []
        label_set = set()
        for line in lines:
            if line == '' or line in label_set:
                continue
            label_list.append(line)
        return label_list

    @staticmethod
    def load_label_dict(dict_file_or_lines):
        return LabelDict(dict_file_or_lines)

    @staticmethod
    def get_label_from_file_name(img_file, label_split_char='_', input_kind_pos=0):
        file_name = os.path.basename(img_file)
        x = 0
        for split_ch in label_split_char:
            a = file_name.rfind(split_ch)
            if a > input_kind_pos:
                x = a
                break

        if x <= 0:
            x = file_name.rfind('.')
        if x > 0:
            text = file_name[0:x]
        else:
            text = file_name

        return text

    @staticmethod
    def get_kind_from_file_name(img_file, input_kind_pos):
        file_name = os.path.basename(img_file)
        if input_kind_pos <= 0:
            return 0

        s_kind = re.sub('\D+', '', file_name[0:input_kind_pos])
        n_kind = int(s_kind) if s_kind else 0
        return n_kind

    @staticmethod
    def get_label_from_file_path(img_file, try_label_on_file_name=False, label_split_char='_', warnning=False,
                                 input_kind_pos=0):
        '''
        根据影像文件路径获得文本标签内容
        :param label_split_char:
        :param img_file:
        :param label_on_file_name: 是否从文件名上读取
        :return:
        '''

        txt_file = FileUtils.change_ext_name(img_file, '.txt')
        if not os.path.isfile(txt_file):
            if try_label_on_file_name:
                return FileUtils.get_label_from_file_name(img_file, label_split_char, input_kind_pos=input_kind_pos)
            elif warnning:
                Logger.warning('not find the txt file:%s' % txt_file)
            return ''

        return FileUtils.get_label_from_txt_file(txt_file)

    @staticmethod
    def get_label_from_txt_file(txt_file):
        text = ''
        try:
            if '.txt'.startswith(FileUtils.get_ext_name(txt_file)):
                text = FileUtils.read_text_from_file(txt_file, encoding='utf-8', strip=True)
        except:
            text = FileUtils.read_text_from_file(txt_file, encoding='gbk', strip=True)
        if len(text) > 0 and ord(text[0]) == 0xfeff:
            text = text[1:]

        return text

    @staticmethod
    def try_get_image_file(txt_file, ext_names=('.jpg', '.png', '.jpeg')):
        for ext_name in ext_names:
            img_file = FileUtils.change_ext_name(txt_file, ext_name)
            if (os.path.isfile(img_file)):
                return img_file
        return None

    @staticmethod
    def read_replace_dict(file):
        if not file or not os.path.isfile(file):
            return None

        lines = FileUtils.read_lines_from_file(file)
        dict = {}
        for line in lines:
            n = line.find('=')
            if n > 0:
                k = line[0:n]
                v = line[n + 1:].strip()
                dict[k] = v
        return dict

    @staticmethod
    def try_replace_from_dict(text, replace_dict):
        if not replace_dict:
            return text

        for k in replace_dict:
            text = text.replace(k, replace_dict[k])
        return text

    @staticmethod
    def remove_label_chars(label, chars):
        result = ''
        for c in label:
            if c in chars:
                continue
            result += c
        return result
