import scrapy
import pandas as pd
from scrapy.http import Request
from ..items import NewsItem

class NewsSpider(scrapy.Spider):
    
    newsPath = '../../MINDlarge_train/news.tsv'
    
    name = "news"
    start_urls = []
    allowed_domains = ["assets.msn.com"] 
 

    start_urls = [url for url in pd.read_csv(newsPath,sep='\t',header=None)[5].unique().tolist() if url is not None and len(url)>5 and url.startswith('http')]

    
    def parse(self, response):
        dStr = response.xpath("//*[@class='date']/time/text()").get()
        if dStr is not None:
            item = NewsItem()
            item['url'] = response.url
            item['publishDate'] = dStr.strip()
        
            yield item
        else:
            pass