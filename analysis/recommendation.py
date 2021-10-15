import hnswlib
import numpy as np
import pandas as pd
import pickle
import os.path
from analysis.common import get_vector_columns

def initSearchIndex(force_create=False):
    fname = 'generate/searchIndex.pkl'
    if not force_create and os.path.isfile(fname):
        
        with open(fname,"rb") as f:
            p = pickle.load(f)
   
    else:
        
        dim = 100
        num_elements = 89222 #number of news we have
        data = pd.read_csv("generate/news_embedding.csv")
        vector_columns = get_vector_columns(data)
        ids = data.NID.apply(lambda nid:int(nid[1:]))
        data = data[vector_columns]

        # Declaring index
        p = hnswlib.Index(space = 'cosine', dim = dim) 

        p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
        p.add_items(data, ids)
        
        with open(fname,'wb') as f:
            pickle.dump(p,f)
    return p


def searchKNearestNeighbors(user_representation,k=5):
    p = initSearchIndex()
    
    vector_columns = get_vector_columns(user_representation)
    
    records = []
    for UID,g in user_representation.groupby('UID'):
        
        query = g[vector_columns]
        
        #get k nearest neighbor
        labels, _ = p.knn_query(query, k = k)
        
        
        for i in labels.flatten():
            NID = 'N'+str(i)
            records.append((UID,NID))
    
    return pd.DataFrame.from_records(records,columns=['UID','NID'])
    
        