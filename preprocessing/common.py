from datetime import datetime
import re
import numpy as np

def time2stamp(utc):
    return int(datetime.strptime(utc, "%m/%d/%Y %I:%M:%S %p").timestamp())

def url2Hash(url):
    if url is None:
        return np.nan
    
    f = re.findall(r'.*/labs/mind/(.*)\.html',url)
    return f[0] if len(f)>0 else np.nan
    