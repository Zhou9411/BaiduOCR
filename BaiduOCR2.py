#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python version: 3.11.0
# System: Windows 11

from concurrent.futures.thread import ThreadPoolExecutor
from inspect import currentframe
from logging import Formatter, makeLogRecord, getLogger, ERROR, FileHandler
from os import walk, rename, remove
from os.path import join, splitext, isdir, dirname, basename, exists
from pathlib import Path
from re import search
from sys import getsizeof
from time import sleep, time
from urllib.request import urlopen, Request

from aip import AipOcr
from cv2 import resize, INTER_AREA, error, adaptiveThreshold, ADAPTIVE_THRESH_GAUSSIAN_C, \
    THRESH_BINARY, imencode, IMWRITE_JPEG_QUALITY, imread, imwrite, IMREAD_GRAYSCALE, bilateralFilter, \
    getStructuringElement, MORPH_ELLIPSE, dilate, MORPH_OPEN, morphologyEx


# 百度OCR类
class BaiduOCR:
    # 初始化
    def __init__(self, paths):
        self.client = None  # 初始化百度OCR客户端
        self.paths = paths  # 初始化图像路径
        self.options = {}  # 初始化参数字典
        self.images = {}  # 初始化图像字典
        self.max_size = 2621440  # 1024 * 1024 * 4 / 1.6 = 2621440 bytes = 2.5MB
        self.max_px = 4096  # Baidu限制最大图像像素:4096
        self.mix_px = 15  # Baidu限制最小图像像素:15
        self.ocr_max_threads = 1  # BaiduOCR付费用户好像限制最大线程数:10;免费用户好像最大2线程,避免不必要的麻烦,默认为1
        self.image_max_threads = 16  # 设置图片预处理线程,自行根据设备性能设置,默认16
        self.is_blueprint = False  # 输入是否为蓝图

    # 记录错误日志
    def log(self, message):
        # 获取调用函数的名称和行号
        func = currentframe().f_back.f_code.co_name
        row = currentframe().f_back.f_lineno

        # 获取当前时间戳
        timestamp = Formatter('%(asctime)s').formatTime(makeLogRecord({}))

        # 格式化日志消息
        log_message = f'{timestamp} - {func} - Line {row}: {message}'

        # 获取 logger 对象并设置日志级别为 ERROR
        logger = getLogger()
        logger.setLevel(ERROR)

        # 创建文件处理器并设置日志级别为 ERROR
        file = join(self.paths, 'error.log')

        file_handler = FileHandler(file, mode='a')
        file_handler.setLevel(ERROR)

        # 设置文件处理器的格式化器
        formatter = Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        # 为 logger 添加文件处理器
        logger.addHandler(file_handler)

        # 记录错误日志
        logger.error(log_message)

        # 关闭文件处理器并从 logger 中移除
        file_handler.close()
        logger.removeHandler(file_handler)

    # 设置百度OCR客户端(app_id, api_key, secret_key为配置文件中的参数)
    def set_client(self, app_id, api_key, secret_key):
        self.client = AipOcr(app_id, api_key, secret_key)

    # 设置路径
    def set_path(self, path):
        self.paths = path

    # 设置参数
    def set_options(self, **kwargs):
        self.options.update(kwargs)  # 修改参数

    # 设置文件大小限制
    def set_size_limit(self, max_size=2621440):
        self.max_size = max_size

    # 设置像素限制
    def set_px_limit(self, max_px=4096, mix_px=15):
        self.max_px = max_px
        self.mix_px = mix_px

    # 设置多线程进程数量
    def set_threads(self, ocr_max_threads=1, image_max_threads=16):
        self.image_max_threads = image_max_threads
        self.ocr_max_threads = ocr_max_threads

    # 设置是否识别蓝图
    def set_blueprint(self, is_blueprint=False):
        self.is_blueprint = is_blueprint

    # 获取图像文件
    def get_files(self):
        # 检查 self.paths 是否为一个目录
        if isdir(self.paths):
            # 遍历指定目录及其子目录
            for root, dirs, files in walk(self.paths):
                # 筛选出扩展名为 jpg/jpeg/png/bmp 的文件
                images = [join(root, file) for file in files if
                          file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                # 将文件绝对路径按照目录名称分类，存储在字典中
                if images:
                    self.images[root] = images
        else:
            # 获取文件扩展名
            ext = splitext(self.paths)[1]
            # 检查文件扩展名是否受支持
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 获取文件所在目录
                path = dirname(self.paths)
                # 将文件绝对路径按照目录名称分类，存储在字典中
                self.images[path] = [self.paths]
                # 修改 self.paths 的值为文件所在的目录
                self.set_path(path)
            else:
                # 抛出异常并记录错误日志
                try:
                    raise Exception(f'不受支持的文件类型: {ext}')
                except Exception as e:
                    self.log(e)
                    return False

    # 获取图像px长宽尺寸
    def get_pixel_size(self, image):
        try:
            return image.shape[:2]
        except AttributeError as e:
            self.log(f"获取图片尺寸失败: {e}")
            return False

    # 获取图像文件大小
    @staticmethod
    def get_file_size(image):
        return getsizeof(image)

    # 处理图像文件px限制
    def process_pixel(self, image, size):
        # 获取尺寸
        height, width = size
        # 计算缩放比例
        if max(height, width) > self.max_px:
            scale = self.max_px / max(height, width)
        elif min(height, width) < self.mix_px:
            scale = self.mix_px / min(height, width)
        else:
            return image  # 满足要求则不需要进行缩放处理

        # 计算缩放后的尺寸
        width = round(width * scale)
        height = round(height * scale)

        # 确保调整后的图像大小满足最大和最小像素限制
        width = max(self.mix_px, min(self.max_px, width))
        height = max(self.mix_px, min(self.max_px, height))

        try:
            image = resize(image, (width, height), interpolation=INTER_AREA)
        except error as e:
            self.log(f"缩放图片尺寸失败: {e}")
            return False

        return image

    # 滤波降噪
    def process_filter(self, image):
        # 应用双边滤波 bilateralFilter()
        try:
            # 创建核处理器
            kernel = getStructuringElement(MORPH_ELLIPSE, (5, 5))
            if self.is_blueprint:
                # 应用双边滤波
                image = bilateralFilter(image, 9, 55, 55)
                # 膨胀图像
                image = dilate(image, kernel, iterations=1)
            else:
                # 应用双边滤波
                image = bilateralFilter(image, 9, 55, 55)
                # 开运算
                image = morphologyEx(image, MORPH_OPEN, kernel)
        except error as e:
            print(f"滤波降噪失败: {e}")
            return False
        return image

    # 处理图像文件颜色
    def process_color(self, image):
        try:
            # 转换为二值图像
            if self.is_blueprint:
                image = adaptiveThreshold(image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 21)
            else:
                image = adaptiveThreshold(image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 5)
        except error as e:
            print(f"转换二值图像失败: {e}")
            return False
        return image

    # 处理图像文件大小限制
    def find_quality(self, image):
        min_quality, max_quality = (0, 100)
        while min_quality <= max_quality:
            mid_quality = (min_quality + max_quality) // 2
            try:
                _, encoded_image = imencode('.jpg', image, [IMWRITE_JPEG_QUALITY, mid_quality])
            except error as e:
                self.log(f"图像编码失败: {e}")
                return False
            size = len(encoded_image)
            if size < self.max_size:
                min_quality = mid_quality + 1
            else:
                max_quality = mid_quality - 1
        # 返回最好的图像质量
        return max_quality

    # 处理图像文件
    def process_image(self, file):
        try:
            image = imread(file, IMREAD_GRAYSCALE)
            size = self.get_pixel_size(image)
            if isinstance(size, bool):
                raise Exception(f"图像尺寸获取失败: {file}")
            image = self.process_filter(image)
            if isinstance(image, bool):
                raise Exception(f"图像滤波降噪失败: {file}")
            image = self.process_pixel(image, size)
            if isinstance(image, bool):
                raise Exception(f"图像尺寸处理失败: {file}")
            image = self.process_color(image)
            if isinstance(image, bool):
                raise Exception(f"图像颜色处理失败: {file}")
            quality = self.find_quality(image)
            if isinstance(quality, bool):
                raise Exception(f"图像大小处理失败: {file}")
            # 将源文件的扩展名修改为.bak
            rename(file, f"{file + '.bak'}")
            imwrite(join(f"{splitext(file)[0] + '.jpg'}"), image, [IMWRITE_JPEG_QUALITY, quality])
        except Exception as e:
            self.log(f"处理图像文件失败: {e}")
            return False

    # 获取文件扩展名
    @staticmethod
    def get_ext_name(url: str) -> None or str:
        match = search(r'\.(\w+)(?:\?|$)', url)
        if match:
            return match.group(1)
        else:
            return False

    # 下载文件
    @staticmethod
    def download_file(url: str) -> None or bytes:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Linux; Android 11; Find X6 Build/RKQ1.201217.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/96.0.4664.45 Mobile Safari/537.36',
                'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br'
            }

            request = Request(url, headers=headers)
            return urlopen(request).read()
        except Exception as e:
            print(f"下载文件失败: {e}")
            return False

    # 保存文件
    def save_file(self, file, data):
        try:
            # 存在文件则删除
            if exists(file):
                remove(file)
            with open(file, 'wb') as f:
                f.write(data)
                print(f"保存文件成功: {file}")
        except Exception as e:
            self.log(f"保存文件失败: {e}")
            return False

    # 执行百度OCR处理
    def process_ocr(self, image):
        try:
            # 调用百度OCR接口文字表格识别异步接口
            request_result = self.client.tableRecognitionAsync(Path(image).read_bytes(), self.options)
            request_id = request_result['result'][0]['request_id']
            # 检查请求是否成功
            if request_id is None:
                print(f"百度OCR接口调用失败: {request_result['error_code']}:{request_result['error_msg']}")
            else:
                print(f"百度OCR接口调用成功: {request_id}")
            # 调用百度OCR接口文字表格识别结果接口
            request_download = self.client.getTableRecognitionResult(request_id)
            # 检查请求是否成功
            # code:3 表示处理完成
            while request_download['result']['ret_code'] != 3:
                sleep(2)  # 延时2秒, 避免过于频繁调用接口
                request_download = self.client.getTableRecognitionResult(request_id)
            # 下载地址
            download_url = request_download['result']['result_data']
            # 获取文件拓展名
            ext_name = self.get_ext_name(download_url)
            file_dir = f"{dirname(image)}/{'表格输出成果'}"
            # 不存在则创建目录使用makedirs()
            makedirs(join(file_dir), exist_ok=True)
            file_name = f"{splitext(basename(image))[0]}.{ext_name}"
            # 下载文件
            data = self.download_file(download_url)
            # 保存文件
            self.save_file(join(file_dir, file_name), data)
        except Exception as e:
            self.log(f"百度OCR处理失败: {e}")
            return False

    # 多线程处理图像文件
    def process_images(self, images):
        with ThreadPoolExecutor(max_workers=self.image_max_threads) as executor:
            # 计数器
            count = 0
            for image in images:
                print(image)
                count += 1
                print(f"正在处理第 {count} 张图片")
                executor.submit(self.process_image, image)
                print(f"第 {count} 张图片处理完成")
            print(f"已处理 {count} 张图片")
        return count

    # 多线程处理ocr
    def process_ocrs(self, files):
        with ThreadPoolExecutor(max_workers=self.ocr_max_threads) as executor:
            # 计数器
            count = 0
            print(f"正在创建表格转换任务...")
            for file in files:
                count += 1
                print(f"正在将第 {count} 张表格添加进转换队列...")
                executor.submit(self.process_ocr, file)
            print(f"表格添加完成,共计 {count} 张表格,请耐心等待转换完成...")
        return count

    # 构建多线程对image文件进程ocr处理
    def process_thread(self):
        # 获取文件路径
        print(f"开始处理文件...")
        self.get_files()
        # 多线程处理图像文件
        for paths, images in self.images.items():
            print(f"开始处理 {paths} 下的图像文件")
            print(f"共有 {len(images)} 张图片需要处理")
            number = self.process_images(images)
            print(f"当前目录共计处理成功 {number} 张图片")
            if len(images) != number:
                print(f"当前目录共计处理失败 {len(images) - number} 张图片")

        # 更新处理后的文件路径
        print(f"开始更新文件路径...")
        self.get_files()
        sleep(1)  # 延时1秒,防止文件路径未更新完成
        print(f"更新文件路径完成...")
        # 多线程处理ocr
        for paths, files in self.images.items():
            print(f"开始转换 {paths} 下的表格")
            print(f"共有 {len(files)} 张表格需要转换")
            number = self.process_ocrs(files)
            print(f"当前目录共计转换成功 {number} 张表格")
            if len(files) != number:
                print(f"当前目录共计处理失败 {len(files) - number} 张表格")


# Test Code
if __name__ == '__main__':
    try:
        from tkinter.filedialog import askdirectory
        from os import system, chdir, makedirs
        from os.path import abspath

        try:
            # 设置requirements.txt在当前脚本目录下
            chdir(dirname(abspath(__file__)))
            system('pip install -r requirements.txt')
        except Exception as e:
            print(f"安装依赖失败: {e}")
            exit(0)
        # 选择文件夹
        # 调用tk组件弹窗选择文件夹
        paths = askdirectory()
        print(f"选择的文件夹为: {paths}")

        # 用户信息 app_id, api_key, secret_key
        user_info = {'app_id': '替换为你的app_id', 'api_key': '替换为你的api_key', 'secret_key': '替换为你的secret_key'}

        # 开始计时
        start_time = time()
        # 实例化
        baidu_ocr = BaiduOCR(paths)
        # 初始化客户端
        baidu_ocr.set_client(user_info['app_id'], user_info['api_key'], user_info['secret_key'])
        # 设置参数
        baidu_ocr.set_options(resultType='excel', languageType='CHN_ENG')  # 设置参数,导出为excel,识别中英文
        # 设置图片大小限值 bytes
        # Baidu 限制文件大小为4M,为base64编码后 + URL编码后的大小;base64编码后大小为原图大小的4/3,URL编码后大小不确定,编码方式不同体积可能增加,也可能不变
        # 故取系数1.33,0.27为保险系数,总系数为1.6:1024^2*4/1.6=2621440bytes=2.5M
        # 网络较差可设为较小值,避免过小,图像质量会很差,导致识别失败
        # baidu_ocr.set_size_limit(655360)
        # 设置像素限值 pixels
        # baidu_ocr.set_px_limit(4096, 15)  # Baidu 默认限制为Max_px = 4096, Min_px = 15
        # 设置多线程
        # baidu_ocr.set_threads(image_max_threads=16, ocr_max_threads=1)  # 默认图片处理为16, ocr处理为1
        # 设置图片处理模式
        # baidu_ocr.set_blueprint(True)  # 默认关闭蓝图处理参数(或其它任意底色)表格文件处理参数
        # 开始处理
        baidu_ocr.process_thread()
        # 结束计时
        end_time = time()
        print(f"处理完成, 共计耗时: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"处理失败: {e}")
        exit(0)
