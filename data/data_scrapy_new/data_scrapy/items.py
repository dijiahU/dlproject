# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


# class DataScrapyItem(scrapy.Item):
#     # define the fields for your item here like:
#     # name = scrapy.Field()
#     pass


class SistSpiderItem(scrapy.Item):
    # 网页文本数据
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    
    # 文件下载专用字段
    file_urls = scrapy.Field() # 告诉 Scrapy 去哪里下载文件
    files = scrapy.Field()     # Scrapy 下载完成后，会把结果信息填在这里