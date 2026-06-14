# import scrapy
# from scrapy.linkextractors import LinkExtractor
# from scrapy.spiders import CrawlSpider, Rule


# class SistSpiderSpider(CrawlSpider):
#     name = "sist_spider"
#     allowed_domains = ["sist.shanghaitech.edu.cn"]
#     start_urls = ["https://sist.shanghaitech.edu.cn"]

#     rules = (Rule(LinkExtractor(allow=r"Items/"), callback="parse_item", follow=True),)

#     def parse_item(self, response):
#         item = {}
#         #item["domain_id"] = response.xpath('//input[@id="sid"]/@value').get()
#         #item["name"] = response.xpath('//div[@id="name"]').get()
#         #item["description"] = response.xpath('//div[@id="description"]').get()
#         return item


import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import re
import json

class SistSpider(CrawlSpider):
    name = "sist_spider"
    allowed_domains = ["shanghaitech.edu.cn"] #, "github.com"
    # 爬虫的起始种子页面
    start_urls = ["https://sist.shanghaitech.edu.cn/"]

    # 定义链接追踪规则 (这是 CrawlSpider 的核心)
    rules = (
        Rule(
            LinkExtractor(
                allow=r'.*',
                deny=(
                    # r'calendar',
                    # r'date=',
                    # 屏蔽 IT 中心这种长串字母数字混合的归档文章页面
                    r'it\.shanghaitech\.edu\.cn.*c\d{4,}', 
                    # 或者干脆屏蔽掉所有纯静态的 page.htm 模板
                    # r'page\.htm', 
                )
            ), 
            callback="parse_item", 
            follow=True
        ),
    )

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.visited_urls = set()
        # 强行读取上次的心血
        try:
            with open('/data/qingyuyang/dlproject/data/data_scrapy_new/sist_corpus.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if 'url' in data:
                        self.visited_urls.add(data['url'])
            print(f"已成功加载 {len(self.visited_urls)} 个历史 URL 记忆！")
        except FileNotFoundError:
            pass

    def parse_item(self, response):
        """
        这个函数定义了当爬虫打开一个新页面时，要抓取什么内容
        """
        # 如果当前网址在历史记忆里，直接抛弃不处理
        if response.url in self.visited_urls:
            return

        # 过滤掉非文本文件 (如果意外爬到了 PDF 或图片链接，直接跳过)
        if not response.headers.get('Content-Type', b'').startswith(b'text/html'):
            return

        # 提取网页标题
        title = response.css('title::text').get(default='').strip()
        
        # 提取网页正文所有的纯文本 (剥离 HTML 标签)
        # 使用 xpath 过滤掉 script 和 style 标签中的不可见代码
        raw_text_list = response.xpath('//body//text()[not(ancestor::script|ancestor::style)]').getall()
        
        # 简单清洗：去除多余的空白符和换行
        clean_text = ' '.join([text.strip() for text in raw_text_list if text.strip()])

        # --- 新增：提取该网页中所有的 PDF 链接 ---
        # 使用 XPath 寻找 href 属性中包含 '.pdf' 的所有链接
        pdf_relative_urls = response.xpath('//a[contains(@href, ".pdf")]/@href').getall()
        
        # 将相对路径 (如 /docs/syllabus.pdf) 转换为绝对路径 (https://...)
        pdf_absolute_urls = [response.urljoin(url) for url in pdf_relative_urls]

        # 将提取到的结构化数据 yield 出来 (Scrapy 会自动帮我们保存)
        yield {
            'url': response.url,
            'title': title,
            'content': clean_text,
            'file_urls': pdf_absolute_urls  # 把列表传给 Pipeline，Scrapy 会自动并发下载
        }