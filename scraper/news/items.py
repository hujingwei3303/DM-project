import scrapy


class NewsItem(scrapy.Item):
    
    url = scrapy.Field()
    
    publishDate = scrapy.Field()
    
    pass
