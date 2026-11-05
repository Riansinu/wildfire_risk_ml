from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
MONTHS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()
RAW_FEATURES = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'month']
FEATURES = ['temp', 'RH', 'wind', 'log_rain', 'DMC', 'DC', 'FFMC', 'ISI', 'month_sin', 'month_cos']
BLOCKS = {'weather': [0, 1, 2, 3], 'moisture': [4, 5], 'fire_indices': [6, 7], 'season': [8, 9]}


def validate_inputs(frame):
    missing = set(RAW_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    result = frame[RAW_FEATURES].copy()
    result['month'] = result['month'].astype(str).str.lower().str.strip()
    if not result['month'].isin(MONTHS).all():
        raise ValueError('month must be a three-letter English month, jan through dec.')
    for col in RAW_FEATURES[:-1]:
        result[col] = pd.to_numeric(result[col], errors='raise')
        if not np.isfinite(result[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{col} must contain finite numeric values.')
    if not result['RH'].between(0, 100).all():
        raise ValueError('RH must be between 0 and 100 percent.')
    if (result[['wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI']] < 0).any().any():
        raise ValueError('Wind, rain and fire indices must be nonnegative.')
    if not result['FFMC'].between(0, 101).all():
        raise ValueError('FFMC must be between 0 and 101.')
    return result


def feature_frame(frame):
    raw = validate_inputs(frame)
    result = raw.drop(columns=['month', 'rain']).copy()
    result['log_rain'] = np.log1p(raw['rain'])
    phase = raw['month'].map({m: i for i, m in enumerate(MONTHS)}) * 2 * np.pi / 12
    result['month_sin'], result['month_cos'] = np.sin(phase), np.cos(phase)
    return result[FEATURES]


def load_data(path=None):
    path = Path(path or ROOT / 'data/forestfires.csv')
    frame = pd.read_csv(path)
    validate_inputs(frame)
    if not {'X', 'Y', 'area'}.issubset(frame.columns):
        raise ValueError('Training data requires X, Y and area.')
    if not np.isfinite(frame[['X', 'Y', 'area']].to_numpy(dtype=float)).all() or (frame.area < 0).any():
        raise ValueError('Invalid location or area values.')
    return frame


def grouped_split(frame, seed=42):
    # Fixed split selected without searching seeds for favorable label balance or scores.
    groups = frame.X.astype(str) + '_' + frame.Y.astype(str)
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    dev, test = next(first.split(frame, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
    train_rel, val_rel = next(second.split(frame.iloc[dev], groups=groups.iloc[dev]))
    splits = {'train': dev[train_rel], 'validation': dev[val_rel], 'test': test}
    for name, ids in splits.items():
        if (frame.iloc[ids].area > 0).nunique() != 2:
            raise ValueError(f'{name} lacks a target class; revise the prespecified split design.')
    return splits, groups


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
