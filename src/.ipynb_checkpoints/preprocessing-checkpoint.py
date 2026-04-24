import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

def custom_feature_engineering(df):
    """Lógica extraída de notebooks 05 y 06"""
    df = df.copy()
    # Limpieza: Eliminar duplicados
    df = df.drop_duplicates()
    
    # Ingeniería de variables: Intensidad y Préstamos
    if 'campaign' in df.columns and 'pdays' in df.columns:
        df['campaign_intensity'] = df['campaign'] / (df['pdays'] + 1)
    
    if 'loan' in df.columns and 'housing' in df.columns:
        df['has_loan_or_housing'] = ((df['loan'] == 'yes') | (df['housing'] == 'yes')).astype(int)
    
    return df

def get_preprocessor(num_cols, cat_cols):
    """Construcción del ColumnTransformer con Yeo-Johnson"""
    
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('yeo', PowerTransformer(method='yeo-johnson')), # Notebook 07
        ('sc', StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    
    return preproc