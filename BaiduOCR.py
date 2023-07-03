#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from os import chdir, system
from os.path import dirname, abspath
from pathlib import Path
from re import search
from time import sleep, time
from typing import Dict, Callable

from aip import AipOcr
from cv2 import imread, resize, INTER_AREA, imencode, IMWRITE_JPEG_QUALITY, IMREAD_IGNORE_ORIENTATION, \
    IMREAD_COLOR, IMREAD_ANYCOLOR
from numpy import ndarray
from requests import get


# 二分查找
def binary_search(left: int, right: int, condition: Callable[[int], bool]) -> int:
    while left <= right:
        mid = (left + right) // 2  # 中间值取整
        if condition(mid):
            left = mid + 1
        else:
            right = mid - 1
    return left - 1


# 百度OCR类
class BaiduOCR:
    # 初始化
    def __init__(self, app_id: str, api_key: str, secret_key: str):
        self.client = AipOcr(app_id, api_key, secret_key)  # 初始化客户端
        self.paths = None  # 初始化图像路径
        self.options = {}  # 初始化参数字典
        self.images = {}  # 初始化图像字典
        self.size = 460800  # 初始化图像大小限制,450KB

    # 设置参数
    def set_options(self, **kwargs):
        self.options.update(kwargs)  # 修改参数

    # 设置图像路径
    def set_paths(self, paths: str):
        self.paths = paths  # 修改图像路径

    # 获取图像
    def get_images(self) -> Dict[str, Path]:
        path = Path(self.paths)  # 转换为Path对象
        if path.is_dir():
            self.images = {file.name: file for file in path.iterdir() if
                           file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']}  # 获取目录下所有图像
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.images = {path.name: path}  # 设置图像字典
            self.paths = path.parent  # 设置图像路径
        return self.images

    # 检查图像大小
    def check_image_size(self, image: Path) -> bool:
        size = image.stat().st_size  # 获取图像大小
        return size <= self.size

    # 压缩图像
    @staticmethod
    def resize_image(image: ndarray, best_quality: int, best_scale: int = 100) -> (ndarray, int):
        height, width = image.shape[:2]  # 获取图像高度和宽度
        new_width = int(width * best_scale / 100)  # 计算新的宽度
        new_height = int(height * best_scale / 100)  # 计算新的高度
        resized_image = resize(image, (new_width, new_height), interpolation=INTER_AREA)  # 重置图像大小
        encode_param = [int(IMWRITE_JPEG_QUALITY), best_quality]  # 设置图像质量
        result, mem_file = imencode('.jpg', resized_image, encode_param)  # 编码图像
        return mem_file, len(mem_file)  # 返回图像和图像大小

    def compress_image(self, file: Path) -> None:
        image = imread(file.as_posix(), IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR | IMREAD_ANYCOLOR)  # 读取图像,忽略方向
        height, width = image.shape[:2]  # 获取图像高度和宽度
        max_px = 4096  # 最大像素
        min_px = 15  # 最小像素
        best_quality = binary_search(1, 100,
                                     lambda quality: self.resize_image(image, quality)[-1] <= self.size)  # 二分查找最佳质量
        best_scale = binary_search(1, 100, lambda scale: self.resize_image(image, best_quality, scale)[
                                                             -1] <= self.size)  # 二分查找最佳缩放比例
        mem_file = self.resize_image(image, best_quality, best_scale)[0]  # 压缩图像
        if max(height, width) > max_px or min(height, width) < min_px:
            def condition(scale):
                width_px = int(width * scale / 100)
                height_px = int(height * scale / 100)
                return max(width_px, height_px) <= max_px and min(width_px, height_px) >= min_px

            best_scale = binary_search(1, 100, condition)  # 二分查找最佳缩放比例
            mem_file = self.resize_image(image, best_quality, best_scale)[0]  # 压缩图像

        # 检查是否存在同名文件
        new_file = file.with_suffix('.jpg')
        if new_file.exists():
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # 获取时间戳
            new_file = new_file.with_stem(new_file.stem + timestamp)  # 修改文件名

        file.rename(file.with_suffix(file.suffix + '.bak'))  # 重命名原文件
        new_file.write_bytes(mem_file.tobytes())  # 保存新文件

    def preprocess_images(self, max_threads: int = 8) -> None:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for name, image in self.get_images().items():
                if image.suffix.lower() != '.jpg' or not self.check_image_size(image):
                    print(f'图片尺寸超限,开始压缩{name}...')
                    executor.submit(self.compress_image, image)
                    print(f'{name}压缩完成!')
                else:
                    print(f'{name}不需要压缩!')

    # 获取文件后缀名
    @staticmethod
    def get_file_extension(url: str) -> None or str:
        match = search(r'\.(\w+)(?:\?|$)', url)
        if match:
            return match.group(1)
        else:
            return None

    # 保存excel表格
    @staticmethod
    def save_excel(file_path: str, data: bytes):
        Path(file_path).write_bytes(data)  # 保存excel表格

    # 处理图像
    def process_image(self, name: str, image: Path):
        file_name = Path(name).stem  # 获取不带后缀名的文件名
        print(f'开始处理{file_name}...')
        request_result = self.client.tableRecognitionAsync(image.read_bytes(), self.options)  # 调用表格文字识别异步接口
        request_id = request_result['result'][0]['request_id']  # 获取请求ID
        url_result = self.client.getTableRecognitionResult(request_id)  # 调用表格文字识别结果接口
        while url_result['result']['ret_code'] != 3:  # 循环等待结果完成
            sleep(2)
            url_result = self.client.getTableRecognitionResult(request_id)
        url = url_result['result']['result_data']  # 获取excel表格数据地址
        file_extension = self.get_file_extension(url)  # 获取文件后缀名
        file_name = f"{file_name}.{file_extension}"  # 拼接为完整的文件名
        data = get(url).content  # 下载数据到内存
        result_dir = f"{self.paths}/表格成果"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        self.save_excel(f"{result_dir}/{file_name}", data)  # 保存excel表格
        return f'{file_name}已经保存完毕!'

    # 单线程运行
    def run(self):
        for name, image in self.get_images().items():
            print(self.process_image(name, image))

    # 多线程运行
    def run_threaded(self, max_threads=4):
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for name, image in self.get_images().items():
                futures.append(executor.submit(self.process_image, name, image))
            for future in as_completed(futures):
                print(future.result())


# Test Code
if __name__ == '__main__':
    path = r'C:\Users\laozh\Desktop\Test'
    app_id = '35559753'  # 填入自己的app_id
    api_key = 'sCjEuoG8zCTmaTql2IIPL3Mp'  # 填入自己的api_key
    secret_key = '31nCisPb8NaUKmZsfFMgZWmzS2ASpzeV'  # 填入自己的secret_key
    '''
    ocr = BaiduOCR(ai, ak, sk)
    ocr.paths = img
    ocr.set_options(resultType='excel', languageType='CHN_ENG')
    # 免费用户不允许多线程
    # 付费用户最多10次/秒
    # ocr.run_threaded()
    # 预处理
    # 计时
    start = time()
    ocr.preprocess_images()
    ocr.run()
    end = time()
    print(f'总耗时:{end - start:.2f}秒')
    '''
    try:
        # 设置requirements.txt在当前目录下
        chdir(dirname(abspath(__file__)))
        system('pip install -r requirements.txt')

        # 开始计时
        start = time()
        # 实例化
        ocr = BaiduOCR(app_id, api_key, secret_key)
        ocr.paths = path  # 设置图片路径
        ocr.set_options(resultType='excel', languageType='CHN_ENG')  # 设置参数
        ocr.preprocess_images()  # 预处理
        ocr.run()  # 单线程运行
        # 多线程运行
        # ocr.run_threaded()

        # 结束计时
        end = time()
        print(f'总耗时:{end - start:.2f}秒')
    except Exception as e:
        print(e)
        input('按任意键退出...')
