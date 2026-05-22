import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

def custom_feature_engineering(df):
    """
    Punto 3: Encapsular lógica reutilizable.
    Aquí integramos lo importante de los notebooks 05 y 06.
    """
    df = df.copy()
    
    # 1. Limpieza básica (del Notebook 05)
    df = df.drop_duplicates()
    
    # 2. Ingeniería de variables (del Notebook 06)
    if 'campaign' in df.columns and 'pdays' in df.columns:
        df['campaign_intensity'] = df['campaign'] / (df['pdays'] + 1)
        
    if 'loan' in df.columns and 'housing' in df.columns:
        df['has_loan_or_housing'] = ((df['loan'] == 'yes') | (df['housing'] == 'yes')).astype(int)
    
    return df

def get_preprocessor(num_cols, cat_cols):
    """
    Punto 2: Usar ColumnTransformer para aplicar distintas transformaciones.
    """
    # Transformación para números (Notebook 07 usó Yeo-Johnson para normalizar)
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('yeo', PowerTransformer(method='yeo-johnson')),
        ('sc', StandardScaler())
    ])
    
    # Transformación para categorías
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Unión de ambos en el ColumnTransformer
    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    
    return preproc