import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wildfire.bma import BayesianModelAverage
from wildfire.data import feature_frame, grouped_split, load_data


def test_spatial_splits_cover_rows_and_never_share_groups():
    df = load_data(); splits, groups = grouped_split(df)
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(df)))
    sets = [set(groups.iloc[ids]) for ids in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))


def test_target_and_feature_leakage():
    df = load_data(); X = feature_frame(df)
    altered = df.copy(); altered['area'] = 99999; altered['X'] = -100; altered['day'] = 'invalid'
    pd.testing.assert_frame_equal(X, feature_frame(altered))
    assert 'area' not in X.columns and np.isfinite(X.to_numpy()).all()
    df.loc[0, 'RH'] = 101
    with pytest.raises(ValueError): feature_frame(df)


def test_bma_posterior_is_normalized_and_predictive_mixture_matches():
    df = load_data(); splits, _ = grouped_split(df)
    ids = splits['train']; X = feature_frame(df); y = (df.area > 0).astype(int)
    pipe = Pipeline([('scale', StandardScaler()), ('model', BayesianModelAverage())])
    pipe.fit(X.iloc[ids], y.iloc[ids]); bma = pipe.named_steps['model']
    assert len(bma.weights_) == 16 and np.isclose(bma.weights_.sum(), 1)
    assert (bma.weights_ > 0).all()
    p = pipe.predict_proba(X.iloc[splits['test']])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    scaled = pipe.named_steps['scale'].transform(X.iloc[splits['test']])
    np.testing.assert_allclose(p[:, 1], bma.candidate_probabilities(scaled) @ bma.weights_)


def test_inference_does_not_refit_scaler_or_models():
    df = load_data(); X = feature_frame(df); splits, _ = grouped_split(df)
    ids = splits['train']; pipe = Pipeline([('scale', StandardScaler()), ('model', BayesianModelAverage())])
    pipe.fit(X.iloc[ids], (df.iloc[ids].area > 0).astype(int))
    mean = pipe.named_steps['scale'].mean_.copy(); weights = pipe.named_steps['model'].weights_.copy()
    pipe.predict_proba(X.iloc[splits['test']])
    np.testing.assert_array_equal(pipe.named_steps['scale'].mean_, mean)
    np.testing.assert_array_equal(pipe.named_steps['model'].weights_, weights)
