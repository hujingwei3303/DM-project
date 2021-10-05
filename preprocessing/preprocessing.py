import pandas as pd
import numpy as np
import csv
import ast
import re
from preprocessing.common import url2Hash,time2stamp

output = 'generate/' 

def getEntityEmbedding(entity_embedding):
    '''
    entityEmbedding: path of MIND entity_embedding.vec
    return: 
        dataframe with readable column name
    '''
    df_embedding = pd.read_csv(entity_embedding,sep='\t',header=None)
    df_embedding = df_embedding[df_embedding.columns[:101]]
    return df_embedding.rename(columns = {c:('WikidataId' if c==0 else 'V'+str(c)) for c in df_embedding.columns}) 

def entity2Embedding(df_entitiyEmbedding,vectorColumns,row):
    '''
    mean entity embedding in title and abstract, remove possible duplicates, 
    '''
    try:
        entities = ast.literal_eval(row['titleEntities'])+ast.literal_eval(row['abstractEntities'])  
        labels = {entity["WikidataId"] for entity in entities}
        return df_entitiyEmbedding[df_entitiyEmbedding.WikidataId.isin(labels)][vectorColumns].mean(axis=0).values
    except:
        return np.full(len(vectorColumns),np.nan)
    
def createNews(news,newsTimes,entityEmbedding):
    '''
    news: path of original MIND news.tsv
    newsTimes: path of publish time of news fetch from web
    entityEmbedding: path of MIND entity_embedding.vec
    
    output:
    /generate/news_embedding.csv
    /generate/news_cleaned.csv
    '''
    df_news = pd.read_csv(news,sep='\t',header=None)
    df_news.rename(columns={0:'NID',1:'category',2:'subcat',3:'title'
                            ,4:'abstract',5:'url',6:'titleEntities',7:'abstractEntities'},inplace=True)
    
    df_news['newsHash'] = df_news.url.apply(url2Hash)
    
    #create embedding dataframe
    df_entityEmbedding = getEntityEmbedding(entityEmbedding)
    print(df_entityEmbedding.shape)
    vectorColumns = [c for c in df_entityEmbedding.columns if not c=='WikidataId']
    emb = df_news.apply(lambda r:entity2Embedding(df_entityEmbedding,vectorColumns,r),axis=1)  
 
    df_newsEmbedding = pd.DataFrame.from_records(emb,columns = vectorColumns)
    df_newsEmbedding = pd.concat([df_news.NID,df_newsEmbedding],axis=1)
    df_newsEmbedding = df_newsEmbedding[~df_newsEmbedding.V1.isnull()]
    df_newsEmbedding.to_csv(output+'news_embedding.csv',index=None)

    #create meta dataframe
    df_news = df_news[['NID','category','subcat','newsHash']]

    df_newstimes = pd.read_csv(newsTimes)
    df_news = df_news.merge(df_newstimes,on='newsHash')
    df_news.to_csv(output+'news_cleaned.csv',index=None)
    
def subcat2vector(subcat,nlp):
    tokens = [t for t in nlp(subcat) if t.has_vector]
    vec = np.zeros((300,1))
    if len(tokens)==0:
        return vec
    
    for token in tokens:
        vec+=token.vector.reshape(-1,1)
    
    vec = vec.mean(axis=1)
    return vec

def createCategoriyEmbeddingNLP(subcategories):
    '''subcategories: path to the cleaned subcategory list, it should under generate/ directory'''
    import spacy
    
    nlp = spacy.load("en_core_web_lg")
    
    vecSize = 300
    
    df_subcategories = pd.read_csv(subcategories,sep=';')
    embeddings = df_subcategories.subcat_tokens.apply(lambda x:subcat2vector(x,nlp))
    df_subcat_embeddings = pd.DataFrame.from_records(embeddings,columns=['V'+str(i) for i in range(1,vecSize+1)])
    pd.concat([df_subcategories,df_subcat_embeddings],axis=1).to_csv(output+'news_subcat_embedding_nlp.csv')
    
def createUsers(behaviors):
    df_behavior = pd.read_csv(behaviors,sep='\t',header=None)
    
    user_impressions_fp = open(output+'user_impressions.csv', 'w',encoding='utf-8')
    user_impressions_writer = csv.writer(user_impressions_fp)
    user_impressions_writer.writerow(['UID','timestamp','NID','attitude'])
        
    user_history_fp = open(output+'user_history.csv', 'w',encoding='utf-8')
    user_history_writer = csv.writer(user_history_fp)
    user_history_writer.writerow(['UID','NID'])
    
    for _,row in df_behavior.iterrows():
        user = row[1]
        ts = time2stamp(row[2])
        

        history = row[3]
        if type(history)==str and len(history)>0:
            for h in history.strip().split(' '):
                user_history_writer.writerow([user,h])
        
        impressions = row[4]
        if type(impressions)==str and len(impressions)>0: 
            for imp in impressions.strip().split(' '):
                i = imp.split('-')
                user_impressions_writer.writerow([user,ts,i[0],i[1]])
                
    user_impressions_fp.close()
    user_history_fp.close()
    
    