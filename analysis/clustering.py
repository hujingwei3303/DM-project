from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist,cosine,squareform
import numpy.ma as ma
import pandas as pd
import numpy as np
from analysis.common import getVectorColumns,initialUserHistory

def clustering(X,vector_columns,threshold,k=3,radius_scale=1,default_radius=0.3,constant_radius=0.0,with_centroid=False):
    ''' X 
            [dataframe] group of user history
        vector_columns 
            [array like] embedding columns V1,V2....V100
        threshold
            [number] threshold for ward clustering
        k 
            [integer] choose k clusters
        radius_scale
            [float] mean+radius_scale*std
        default_radius
            [float] default value for radius
        constant_radius
            [float] if larger than zero, then radius use a constant
            
       '''
    m,n = X.shape
    
    if constant_radius>0:
        default_radius = constant_radius
        
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
                
                distance_upper_bound.append(default_radius) 
                
                medoids.append(X.loc[X.index[idx[0]]][vector_columns].values)
                if with_centroid:
                    centroids.append(mean_vector) 
                continue
                
            min_distance_i = masked_pairwise.sum(axis=1).argmin()
            
            distances_i = masked_pairwise.compressed().reshape((len_idx,len_idx))[0]
            
            distances_i = distances_i[1:]
            
           
            #68%-95%-99.8% for 1,2,3 std
            #!!!!!however
            #according to the result, larger cluster has lower std.
            
            if constant_radius==0.0:
                if len(distances_i)==0 or distances_i.sum()==0:#all point in cluster have same entity embeddings
                    radius = default_radius
                else:
                    radius = distances_i.mean()+radius_scale*distances_i.std()
                distance_upper_bound.append(radius) 
            else:
                distance_upper_bound.append(default_radius)  
                
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


def clusteringBatch(t0,history='',lam=0.01,df_history=None,with_centroid=False,**kwargs):
    if df_history is None:
        df_history = initialUserHistory(history)
        
    df_history['importance'] = np.exp(-lam*(t0-df_history.publishDate)/100000)

    vector_columns = getVectorColumns(df_history)
    
    if with_centroid:
        records_medoid = []
        records_centroid = []
        
        for UID,g in df_history.groupby('UID'):
            
            medoid,centroid,radius = clustering(X=g,vector_columns=vector_columns,with_centroid=with_centroid,**kwargs)
        
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
        medoid,radius = clustering(X=g,vector_columns=vector_columns,with_centroid=with_centroid,**kwargs)
        
        len_medoid = len(medoid)
        
        for c in range(len_centroid):
            record = []
            record.append(UID)
            record+=centroid[c].tolist()
            record.append(radius[c])
            records_medoid.append(record)
                
        
        df_medoid = pd.DataFrame.from_records(records_medoid,columns=['UID']+vector_columns+['radius'])
        return df_medoid