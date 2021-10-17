from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist,cosine,squareform
import numpy.ma as ma
import pandas as pd
import numpy as np
from analysis.common import getVectorColumns,initialUserHistory

def clustering(X,vector_columns,threshold,k=3,metric='euclidean',with_centroid=False):
    ''' X 
            [dataframe] group of user history
        vector_columns 
            [array like] embedding columns V1,V2....V100
        threshold
            [number] threshold for ward clustering
        k 
            [integer] choose k clusters
            
       '''
    m,n = X.shape
    
    #print(X)
    if m>1:
        pairwise_distance = pdist(X[vector_columns], metric=metric)
        
        labels = fcluster(ward(pairwise_distance), t=threshold, criterion='distance')
        #print(labels)
        num_clusters = labels.max()
        #print(num_clusters)
        scores = {}
        for i in range(num_clusters):
            c = i+1 #choose cluster
            #if we use sum, both consider number of item and time importance
            importance_score = X.loc[labels==c].importance.sum()
            scores[c] = importance_score
        
        if num_clusters>k:
            p = np.array(list(scores.values()))
            sum_score = p.sum()
            p /= sum_score
            chosed = np.random.choice(list(scores.keys()),p=p,size=k,replace=False)
        else:
            chosed = np.array(list(scores.keys()))
            
        
        medoids = []
        distance_upper_bound = []
        
        pairwise = squareform(pairwise_distance) 
        
        if with_centroid:
            centroids = []
           
            
        for c in chosed:
            idx = np.argwhere(labels==c).flatten()
            len_idx = len(idx)
        
            mask = np.ones(pairwise.shape,dtype=int)
            
            for j in idx:
                mask[j,idx]=0
                
            masked_pairwise = ma.array(pairwise, mask = mask)
            
            if with_centroid:
                mean_vector = X.loc[X.index[idx]][vector_columns].mean().values
            
            if len_idx<3:
                
                medoids.append(X.loc[X.index[idx[0]]][vector_columns].values)
                if with_centroid:
                    centroids.append(mean_vector) 
                continue
                
            min_distance_i = masked_pairwise.sum(axis=1).argmin()
            
            distances_i = masked_pairwise.compressed().reshape((len_idx,len_idx))[0]
            
            distances_i = distances_i[1:]
            
            medoids.append(X.loc[X.index[min_distance_i]][vector_columns].values)
            
            if with_centroid:
                centroids.append(mean_vector)
            
        if with_centroid:
            return medoids,centroids
        else:
            return medoids
    else:
        if with_centroid:
            return X[vector_columns].values,X[vector_columns].values
        else:
            return X[vector_columns].values


def clusteringBatch(t0,history='',lam=0.01,df_history=None,with_centroid=False,**kwargs):
    if df_history is None:
        df_history = initialUserHistory(history)
        
    df_history['importance'] = np.exp(-lam*(t0-df_history.publishDate)/24/3600)

    vector_columns = getVectorColumns(df_history)
    
    if with_centroid:
        records_medoid = []
        records_centroid = []
        
        for UID,g in df_history.groupby('UID'):
            if len(g)<100:
                continue
                
            medoid,centroid = clustering(X=g,vector_columns=vector_columns,with_centroid=with_centroid,**kwargs)
        
            len_centroid = len(centroid)
        
            for c in range(len_centroid):
                record = []
                record.append(UID)
                record+=centroid[c].tolist()
                records_centroid.append(record)
                
                record = []
                record.append(UID)
                record+=medoid[c].tolist()
                records_medoid.append(record)
        
        df_centroid = pd.DataFrame.from_records(records_centroid,columns=['UID']+vector_columns)
        df_medoid = pd.DataFrame.from_records(records_medoid,columns=['UID']+vector_columns)
        return df_medoid,df_centroid
    else:
        for UID,g in df_history.groupby('UID'):
            if len(g)<100:
                continue
                
            medoid = clustering(X=g,vector_columns=vector_columns,with_centroid=with_centroid,**kwargs)

            len_medoid = len(medoid)

            for c in range(len_centroid):
                record = []
                record.append(UID)
                record+=centroid[c].tolist()
                records_medoid.append(record)
                
        
        df_medoid = pd.DataFrame.from_records(records_medoid,columns=['UID']+vector_columns)
        return df_medoid