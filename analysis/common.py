import pandas as pd

def getVectorColumns(df):
    return [c for c in df.columns if c.startswith('V')]

def initialUserHistory(history):
    df_history = pd.read_csv(history)
    df_news_meta = pd.read_csv('generate/news_cleaned.csv')
    df_history = df_history.merge(df_news_meta,on='NID')
    
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_news_embedding.set_index('NID')
    df_history= df_history.merge(df_news_embedding,on='NID')
    
    df_history.set_index('UID',append=True)
    return df_history

def initialUserImpression(impression):
    df_impression = pd.read_csv(impression)
    df_impression = df_impression[df_impression.attitude==1]
    df_news_embedding = pd.read_csv('generate/news_embedding.csv')
    df_news_embedding.set_index('NID')
    df_impression = df_impression.merge(df_news_embedding,on='NID')
    df_impression.set_index('UID',append=True)
    return df_impression