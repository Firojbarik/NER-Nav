import pandas as pd
import numpy as np

ds = pd.read_parquet('data/processed/ml/real_temporal_risk_dataset.parquet')

# Check terrain by event
for eid, grp in ds.groupby('event_id'):
    elev_mean = grp['elevation_m'].mean()
    slope_mean = grp['slope_degrees'].mean()
    label = grp['label'].iloc[0] if len(grp) == 1 else 'mixed'
    print(f'{eid:45s} n={len(grp):2d} elev={elev_mean:7.1f} slope={slope_mean:6.2f} pos={int(grp["label"].sum())} neg={int((grp["label"]==0).sum())}')

print('\n--- Test set ---')
# Recreate the split
from scripts.train_production_risk_model import split_by_events
train, val, test = split_by_events(ds, n_val_events=4, n_test_events=4)

for eid, grp in test.groupby('event_id'):
    elev_mean = grp['elevation_m'].mean()
    slope_mean = grp['slope_degrees'].mean()
    print(f'{eid:45s} n={len(grp):2d} elev={elev_mean:7.1f} slope={slope_mean:6.2f} pos={int(grp["label"].sum())} neg={int((grp["label"]==0).sum())}')

print('\n--- Train set ---')
for eid, grp in train.groupby('event_id'):
    elev_mean = grp['elevation_m'].mean()
    slope_mean = grp['slope_degrees'].mean()
    print(f'{eid:45s} n={len(grp):2d} elev={elev_mean:7.1f} slope={slope_mean:6.2f} pos={int(grp["label"].sum())} neg={int((grp["label"]==0).sum())}')

# Check feature distributions
print('\n--- Elevation by label ---')
print(ds.groupby('label')['elevation_m'].describe())
print('\n--- Slope by label ---')
print(ds.groupby('label')['slope_degrees'].describe())