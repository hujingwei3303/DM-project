from scipy.spatial.distance import cosine,cdist,squareform
import pandas as pd
import numpy as np
from analysis.common import getVectorColumns,initialUserImpression,initialUserHistory
from analysis.clustering import clusteringBatch

def baselineTestAvg(history='',df_history=None):
    if df_history is None:
        df_history = initialUserHistory(history)

    vector_columns = getVectorColumns(df_history)

    records_avg = []
   
    for UID,g in df_history.groupby('UID'):
        if len(g)<100:
            continue

        avgVector = g[vector_columns].values.mean(axis=0)
        
        record = []
        record.append(UID)
        record += avgVector.tolist()
        records_avg.append(record)

    df_avg = pd.DataFrame.from_records(records_avg,columns=['UID']+vector_columns)
    
    return df_avg

def baselineTest(history='',df_history=None,k=3):
    
    if df_history is None:
        df_history = initialUserHistory(history)
    
    #Sort by publishDate ascendingly
    df_history.sort_values('publishDate',inplace=True)

    vector_columns = getVectorColumns(df_history)

    records_random = []
    records_latest = []
   
    for UID,g in df_history.groupby('UID'):
        if len(g)<100:
            continue
            
        #randomly draw 3 samples from group
        randomMedoid = g.sample(n=min(len(g),k))[vector_columns].values
        
        for m in randomMedoid:
            record = []
            record.append(UID)
            record += m.tolist()
            records_random.append(record)

        #draw the latest sample from group
        latestMedoid = g.loc[g.index[max(-k,-len(g)):]][vector_columns].values
        
        for m in latestMedoid:
            record = []
            record.append(UID)
            record += m.tolist()
            records_latest.append(record)

    df_random = pd.DataFrame.from_records(records_random,columns=['UID']+vector_columns)
    
    df_latest = pd.DataFrame.from_records(records_latest,columns=['UID']+vector_columns)
    
    return df_random,df_latest


def measurement(df_user_representation,similarity_threshold=0.4,metric='cosine',impression='',df_impression=None):
    if df_impression is None:
        df_impression = initialUserImpression(impression)
    
    vector_columns = getVectorColumns(df_impression)
        
    measure = []
    for UID,g in df_impression.groupby('UID'):
        
        user = df_user_representation[df_user_representation.UID==UID]
        
        if len(user)==0:
            continue
        
        user = user[vector_columns].values
        
        positive = g[vector_columns]
        
        d = cdist(positive,user, metric=metric)
       
        #at least one distance consider similar
        #print(d.shape,len(user))
        hits = (d<similarity_threshold).sum(axis=1)>0
        #print((d<similarity_threshold).sum(axis=1))
        recall = hits.mean()
        
        hits_d = d[hits]
        #print(hits_d.shape)
        where_clusters = hits_d.argmin(axis=1)
        #print(where_clusters)
        hit = 1 - len(np.unique(where_clusters))/len(user)
       
        measure.append((UID,recall,hit))
        
    return pd.DataFrame.from_records(measure,columns=['UID','recall','percent_empty'])

def tuning(df_history,df_impression,t0,threshold,lam):
    res = []
    print("Clustering...")
    medoids,centroids = clusteringBatch(t0,df_history=df_history,threshold=threshold,lam=lam,with_centroid=True)
    print("Evaluating...")
    m_c = measurement(centroids,df_impression=df_impression,similarity_threshold=0.3)
    m_m = measurement(medoids,df_impression=df_impression,similarity_threshold=0.3)
    print(m_c.recall_mean())
    #print(m_m)
    res.append(threshold)
    res.append(lam)
    res.append(m_m.recall.mean()) 
    res.append(m_m.percent_empty.mean())
    res.append(medoids.groupby("UID").size().mean())
    res.append(m_c.recall.mean()) 
    res.append(m_c.percent_empty.mean())
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
    return pd.DataFrame(result, columns=['Threshold','Lambda','Medoid Recall','Empty medoids','Medoids per user','Centroid Recall','Empty centroids','Centroids per user'])
   