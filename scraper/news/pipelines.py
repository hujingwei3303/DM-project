from itemadapter import ItemAdapter
import logging
import csv
import re
from datetime import datetime

class NewsPipeline:
    
    
    def __init__(self):
        self.fp = None  
        self.csvWriter = None
        self.saved_path = '../generate/newstimes.csv'
        self.headers = ['newsHash','publishDate']
        
    def open_spider(self, spider):
        self.fp = open(self.saved_path, 'w',encoding='utf-8')
        self.csvWriter = csv.writer(self.fp)
        self.csvWriter.writerow(self.headers)
        
    def process_item(self, item, spider):
        url = item['url']
        newsHash = ''
        hashes = re.findall(r'.*/labs/mind/(.*)\.html',url)
        if len(hashes)>0:
            newsHash = hashes[0]
            
        publishDate = item['publishDate']
        publishDate = int(datetime.strptime(publishDate, "%m/%d/%Y").timestamp())
        
        self.csvWriter.writerow([newsHash,publishDate])
        return item 

    def close_spider(self, spider):
        self.fp.close()
