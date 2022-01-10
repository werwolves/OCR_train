import timeit

from aip import AipOcr
import os
import glob

from util.io_utils import FileUtils

""" 你的 APPID AK SK """
APP_ID = '11504307'
API_KEY = 'vgl1SIXOF5VmY5tHiEZf1kvv'
SECRET_KEY = 'odU1WnWo0dhfLnKCPoyBkxoKa6DPVZdk'

class BaiduOcrUtils:
    def __init__(self):
        self.client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    def predict_folder(self, folder, filters=("*.jpg","*.png"), make_txt_file=True):
        for name in filters:
            files = glob.glob(folder + "/" + name)
            self.predict_files(files, make_txt_file)

    def predict_files(self, files, make_txt_file=True):
        for file in files:
            txt_file = FileUtils.change_ext_name(file, '.txt')
            if os.path.exists(txt_file):
                continue
            try:
                self.predict_file(file, make_txt_file)
            except BaseException as e:
                print(e)

    def predict_file(self, file, make_txt_file=True, remove_sapce=True):
        text = self.predict_image_file(file,remove_sapce)

        if make_txt_file:
            txt_file = FileUtils.change_ext_name(file, '.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
        return text

    def predict_image_file(self, file ,remove_sapce=True):
        start = timeit.default_timer()
        """ 如果有可选参数 """
        options = {}
        options["language_type"] = "ENG"
        options["detect_direction"] = "false"
        options["detect_language"] = "false"
        options["probability"] = "false"

        """ 带参数调用通用文字识别, 图片参数为本地图片 """
        image_data = FileUtils.read_bytes(file)
        result = self.client.basicAccurate(image_data, options)

        text = ''
        words_result = result.get('words_result', [])
        for word in words_result:
            text += word.get('words')

        if remove_sapce:
            text = text.replace(' ','')

        elapsed = (timeit.default_timer() - start)
        print('Baidu Ocr Predict image:{:s} label {:s} - {:f} second'.format(FileUtils.get_file_name(file), text, elapsed))
        return text


if __name__ == '__main__':
    ocr = BaiduOcrUtils()
    ocr.predict_folder('/Users/hex/git/ai-platform/ai-training/data/fr-num/frnumfh')