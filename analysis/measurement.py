from scipy.spatial.distance import cosine,cdist,squareform
import pandas as pd
import numpy as np
from analysis.common import getVectorColumns,initialUserImpression,initialUserHistory
from analysis.clustering import clusteringBatch

def baselineTestAvg(history,default_radius=0.3):
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_news_meta = pd.read_csv('generate/news_cleaned.csv')

    df_history = pd.read_csv(history)
    df_history = df_history.merge(df_news_embedding,on='NID')
    df_history = df_history.merge(df_news_meta,on='NID')
    
    #Sort by publishDate ascendingly
    df_history.sort_values('publishDate',inplace=True)

    vector_columns = getVectorColumns(df_history)

    records_avg = []
   
    for UID,g in df_history.groupby('UID'):
        if len(g)<2:
            continue

        avgVector = g[vector_columns].values.mean(axis=0)
        
        record = []
        record.append(UID)
        record += avgVector.tolist()
        records_avg.append(record)

    df_avg = pd.DataFrame.from_records(records_avg,columns=['UID']+vector_columns)
    df_avg['radius'] = default_radius
    
    return df_avg
def baselineTest(history,default_radius=0.3):
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_news_meta = pd.read_csv('generate/news_cleaned.csv')

    df_history = pd.read_csv(history)
    df_history = df_history.merge(df_news_embedding,on='NID')
    df_history = df_history.merge(df_news_meta,on='NID')
    
    #Sort by publishDate ascendingly
    df_history.sort_values('publishDate',inplace=True)

    vector_columns = getVectorColumns(df_history)

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


def measurement(df_user_representation,impression='',df_impression=None):
    if df_impression is None:
        df_impression = initialUserImpression(impression)
    
    vector_columns = getVectorColumns(df_impression)
        
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

def tuning(df_history,df_impression,t0,threshold,lam):
    res = []
    print("Clustering...")
    medoids,centroids = clusteringBatch(t0,df_history=df_history,threshold=threshold,lam=lam,with_centroid=True,constant_radius=0.3)
    print("Evaluating...")
    m_c = measurement(centroids,df_impression=df_impression)
    m_m = measurement(medoids,df_impression=df_impression)
    #print(m_m)
    res.append(threshold)
    res.append(lam)
    res.append(m_m.recall.mean()) 
    res.append(m_m.percent_empty.mean())
    res.append(medoids.radius.mean())
    res.append(medoids.groupby("UID").size().mean())
    res.append(m_c.recall.mean()) 
    res.append(m_c.percent_empty.mean())
    res.append(centroids.radius.mean())
    res.append(centroids.groupby("UID").size().mean())
    return res

def tuningParameters(subsetNr, lam, threshold,size=-1):
    history = 'generate/user_history_'+subsetNr+'.csv'
    impression = 'generate/user_impressions_'+subsetNr+'.csv'
    
    
    df_impression = initialUserImpression(impression)
    df_history = initialUserHistory(history)
        
    if size != -1:
        df_impression = df_impression.loc[0:size]
        
    t0 = 1575586800+1000
    result = []
    for t in threshold:
        for l in lam:
            print("Running with threshold", t, "and lambda", l)
            r = tuning(df_history, df_impression, t0, t, l)
            #print(r)
            result.append(r)
    return pd.DataFrame(result, columns=['Threshold','Lambda','Medoid Recall','Empty medoids','Medoid radius','Medoids per user','Centroid Recall','Empty centroids','Centroid radius','Centroids per user'])
   