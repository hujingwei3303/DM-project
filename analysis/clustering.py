from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist,cosine,squareform
import numpy.ma as ma
import pandas as pd
import numpy as np
from analysis.common import get_vector_columns

def clustering(X,vector_columns,threshold,lam,radius_scale=1,default_radius=0.3,with_centroid=False):
    ''' X 
            [dataframe] group of user history
        vector_columns 
            [array like] embedding columns V1,V2....V300
        threshold
            [number] threshold for ward clustering
        lam 
            [number] paramter for importance sampling
       '''
    m,n = X.shape
    #print(X)
    if m>1:
        pairwise_distance = pdist(X[vector_columns], metric='cosine')
        
        labels = fcluster(ward(pairwise_distance), t=threshold, criterion='distance')
        #print(labels)
        num_clusters = labels.max()
        
        scores = {}
        for i in range(num_clusters):
            c = i+1 #choose cluster
            #if we use sum, both consider number of item and time importance
            importance_score = X.loc[labels==c].importance.sum()
            scores[c] = importance_score
        
        if num_clusters>3:
            p = np.array(list(scores.values()))
            sum_score = p.sum()
            p /= sum_score
            chosed = np.random.choice(list(scores.keys()),p=p,size=3,replace=False)
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
                distance_upper_bound.append(default_radius)   
                medoids.append(X.loc[X.index[idx[0]]][vector_columns].values)
                if with_centroid:
                    centroids.append(mean_vector) 
                continue
            min_distance_i = masked_pairwise.sum(axis=1).argmin()
            distances_i = masked_pairwise.compressed().reshape((len_idx,len_idx))[0]
            distances_i = distances_i[~(distances_i==0)]
            
            #68%-95%-99.8% for 1,2,3 std
            #!!!!!however
            #according to the result, larger cluster has lower std.
            
            distance_upper_bound.append(distances_i.mean()+radius_scale*distances_i.std())   
            medoids.append(X.loc[X.index[min_distance_i]][vector_columns].values)
            
            if with_centroid:
                centroids.append(mean_vector)
            
        if with_centroid:
            return medoids,centroids,distance_upper_bound
        else:
            return medoids,distance_upper_bound
    else:
        if with_centroid:
            return X[vector_columns].values,X[vector_columns].values,default_radius*np.ones(X.shape[0])
        else:
            return X[vector_columns].values,default_radius*np.ones(X.shape[0])


def clusteringBatch(history,t0,threshold=0.7,lam=0.01,radius_scale=1,default_radius=0.3, with_centroid=False):
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_news_meta = pd.read_csv('generate/news_cleaned.csv')
    
    df_history = pd.read_csv(history)
    
    #df_history = df_history[df_history.UID=='U1']
 
    df_history = df_history.merge(df_news_embedding,on='NID')
    df_history = df_history.merge(df_news_meta,on='NID')


    df_history['importance'] = np.exp(-lam*(t0-df_history.publishDate)/100000)

    vector_columns = get_vector_columns(df_history)

    if with_centroid:
        records_medoid = []
        records_centroid = []
        
        for UID,g in df_history.groupby('UID'):
            medoid,centroid,radius = clustering(g,vector_columns,\
                                                threshold,lam,radius_scale,default_radius,with_centroid)
        
            len_centroid = len(centroid)
        
            for c in range(len_centroid):
                record = []
                record.append(UID)
                record+=centroid[c].tolist()
                record.append(radius[c])
                records_centroid.append(record)
                
                record = []
                record.append(UID)
                record+=medoid[c].tolist()
                record.append(radius[c])
                records_medoid.append(record)
        
        df_centroid = pd.DataFrame.from_records(records_centroid,columns=['UID']+vector_columns+['radius'])
        df_medoid = pd.DataFrame.from_records(records_medoid,columns=['UID']+vector_columns+['radius'])
        return df_medoid,df_centroid
    else:
        medoid,radius = clustering(g,vector_columns,threshold,lam,radius_scale,default_radius,with_centroid)
        
        len_medoid = len(medoid)
        
        for c in range(len_centroid):
            record = []
            record.append(UID)
            record+=centroid[c].tolist()
            record.append(radius[c])
            records_medoid.append(record)
                
        
        df_medoid = pd.DataFrame.from_records(records_medoid,columns=['UID']+vector_columns+['radius'])
        return df_medoid