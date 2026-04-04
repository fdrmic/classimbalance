import pandas as pd
import numpy as np
from pathlib import Path
from aml_benchmark.features.pipeline import FeaturePipeline, FEATURE_NAMES

print(f'Total Features: {len(FEATURE_NAMES)}')
assert len(FEATURE_NAMES) == 30, f'Erwartet 30, got {len(FEATURE_NAMES)}'

df = pd.read_csv('data/raw/LI-Small_Trans.csv', nrows=50000, header=0)
df.columns = ['timestamp','from_bank','from_account','to_bank','to_account',
              'amount_received','receiving_currency','amount_paid',
              'payment_currency','payment_format','is_laundering']
df['timestamp'] = pd.to_datetime(df['timestamp'])

pipeline = FeaturePipeline(accounts_path='data/raw/LI-Small_accounts.csv')
X_train = pipeline.fit_transform(df)
print(f'fit_transform shape: {X_train.shape}')
assert X_train.shape[1] == 30

X_val = pipeline.transform(df.tail(5000))
print(f'transform shape: {X_val.shape}')
assert X_val.shape[1] == 30

fan_in_idx = FEATURE_NAMES.index('fan_in_score')
fan_out_idx = FEATURE_NAMES.index('fan_out_score')
assert X_train[:, fan_in_idx].sum() > 0, 'fan_in_score ist überall 0!'
assert X_train[:, fan_out_idx].sum() > 0, 'fan_out_score ist überall 0!'
print(f'fan_in_score mean: {X_train[:, fan_in_idx].mean():.4f}')
print(f'fan_out_score mean: {X_train[:, fan_out_idx].mean():.4f}')

curr_idx = FEATURE_NAMES.index('currency_mismatch')
print(f'currency_mismatch rate: {X_train[:, curr_idx].mean()*100:.2f}%')

print()
print('=== ALLE TESTS BESTANDEN ===')