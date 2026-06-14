# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter


# class DataScrapyPipeline:
#     def process_item(self, item, spider):
#         return item


import os
from urllib.parse import unquote, urlparse
from scrapy.pipelines.files import FilesPipeline
from scrapy import Request

class CustomPdfPipeline(FilesPipeline):
    
    def get_media_requests(self, item, info):
        """针对 item 中的每个 PDF 链接发起异步下载请求"""
        for file_url in item.get('file_urls', []):
            yield Request(file_url)

    def file_path(self, request, response=None, info=None, *, item=None):
        """自定义下载保存的文件名 (避免默认的乱码哈希名)"""
        # 从 URL 中解析出原本的文件名
        # unquote 用于 URL 解码，将 '%20' 等转回空格或中文字符
        original_filename = unquote(os.path.basename(urlparse(request.url).path))
        
        # 将文件统一下载到 'pdfs' 子目录下
        return f'pdfs/{original_filename}'