import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

class TransactionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime", amount_col="Amount", customer_id_col="CustomerId"):
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.customer_id_col = customer_id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert datetime column
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df['transaction_hour'] = df[self.datetime_col].dt.hour
        df['transaction_day'] = df[self.datetime_col].dt.day
        df['transaction_month'] = df[self.datetime_col].dt.month
        df['transaction_year'] = df[self.datetime_col].dt.year

        # Group by customer
        agg_df = df.groupby(self.customer_id_col)[self.amount_col].agg(
            total_transaction_amount='sum',
            average_transaction_amount='mean',
            transaction_count='count',
            std_transaction_amount='std'
        ).reset_index()

        # Merge
        df = df.merge(agg_df, on=self.customer_id_col, how='left')

        return df

# encoding

def one_hot_encode(df, columns):

    df_encoded = pd.get_dummies(df, columns=columns, drop_first=False)

    return df_encoded

# dropping columns
def drop_columns(df, columns):
    
    return df.drop(columns=columns, errors='ignore')
                             
# Scaling
def standard_scale(df, columns):
   
    scaler = StandardScaler()

    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    return df_scaled, scaler

# proxy target variable engineering
def create_proxy_target(df, target_col='Target', threshold=0.5):

    snapshot_date = df['TransactionStartTime'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'CustomerId': 'count',                                         # Frequency
        'Amount': 'sum'                                     # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'CustomerId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    df = df.merge(rfm, on='CustomerId', how='left')

    return df