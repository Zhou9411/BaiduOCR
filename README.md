# BaiduOCR
新手学习项目,调用Baidu SDK,没什么技术含量,欢迎大家多多指导!

	目前实现功能:
		1.根据用户指定限值,自行处理图片文件符合需求。
		2.自动查询Baidu OCR处理结果,将表格文件保存到本地。
		3.实现了多线程处理,图片预处理比较费时间,默认启用16线程,OCR因为采用Baidu免费账户,默认设置为1。
		  请根据自身情况设置。
		4.自动安装依赖库,请务必保证 requirements.txt与Python脚本在同一目录下,目前外部依赖如下:
		  a.baidu_aip==4.16.11
		  b.opencv_python==4.8.0.74
