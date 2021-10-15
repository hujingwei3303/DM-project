from scipy.spatial.distance import cosine,cdist,squareform
import pandas as pd
import numpy as np
from analysis.common import get_vector_columns

def baselineTest(history,default_radius=0.3):
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_news_meta = pd.read_csv('generate/news_cleaned.csv')

    df_history = pd.read_csv(history)
    df_history = df_history.merge(df_news_embedding,on='NID')
    df_history = df_history.merge(df_news_meta,on='NID')
    
    #Sort by publishDate ascendingly
    df_history.sort_values('publishDate',inplace=True)

    vector_columns = get_vector_columns(df_history)

    records_random = []
    records_latest = []
   
    for UID,g in df_history.groupby('UID'):
        if len(g)<4:
            continue
            
        #randomly draw 3 samples from group
        randomMedoid = g.sample(n=3)[vector_columns].values
        
        for m in randomMedoid:
            record = []
            record.append(UID)
            record += m.tolist()
            records_random.append(record)

        #draw the latest sample from group
        #print(g.index[-3:])
        latestMedoid = g.loc[g.index[-3:]][vector_columns].values
        
        for m in latestMedoid:
            record = []
            record.append(UID)
            record += m.tolist()
            records_latest.append(record)

    df_random = pd.DataFrame.from_records(records_random,columns=['UID']+vector_columns)
    df_random['radius'] = default_radius
    
    df_latest = pd.DataFrame.from_records(records_latest,columns=['UID']+vector_columns)
    df_latest['radius'] = default_radius
    
    return df_random,df_latest


def measurement(impression,df_user_representation):
    df_impression = pd.read_csv(impression)
    df_impression = df_impression[df_impression.attitude==1]
    
    #df_impression = df_impression[df_impression.UID=='U1']
    
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_impression = df_impression.merge(df_news_embedding,on='NID')
    
    vector_columns = get_vector_columns(df_impression)
        
    measure = []
    for UID,g in df_impression.groupby('UID'):
        
        user = df_user_representation[df_user_representation.UID==UID]
        
        if len(user)==0:
            continue
        
        user_radius = user.radius.values
        user = user[vector_columns].values
        
        positive = g[vector_columns]
        
        d = cdist(positive,user, metric='cosine')
       
        #at least one distance in radius
        hits = (d<user_radius).sum(axis=1)>0
        recall = hits.mean()
        
        hits_d = d[hits]
        where_clusters = hits_d.argmin(axis=1)
        
        hit = 1 - len(np.unique(where_clusters))/len(user)
       
        measure.append((UID,recall,hit))
        
    return pd.DataFrame.from_records(measure,columns=['UID','recall','percent_empty'])