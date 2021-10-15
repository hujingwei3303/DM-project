def get_vector_columns(df):
    return [c for c in df.columns if c.startswith('V')]