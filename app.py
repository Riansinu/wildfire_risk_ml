"""Launch: python -m streamlit run app.py"""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wildfire.data import MONTHS, RAW_FEATURES, feature_frame, validate_inputs

st.set_page_config(page_title='Ember | Bayesian Wildfire Study', page_icon='🌲', layout='wide')


@st.cache_resource
def load_bundle():
    # Load only the project's locally produced artifact. Never accept uploaded pickle files.
    return joblib.load(ROOT / 'artifacts/models.joblib')


def range_messages(frame, bundle):
    messages = []
    for col, bounds in bundle['training_ranges'].items():
        outside = ~frame[col].between(bounds['min'], bounds['max'])
        if outside.any():
            messages.append(f"{col}: {outside.sum()} row(s) outside training range {bounds['min']:g}–{bounds['max']:g}")
    if not frame.month.isin(bundle['training_months']).all():
        messages.append('Some months were absent from the training split.')
    return messages


st.title('Ember · Bayesian Wildfire Study')
st.caption('UNIT 5 / BAYESIAN MODEL AVERAGING / MONTESINHO, PORTUGAL')
st.info('Research proxy: probability that a recorded fire has area > 0 ha. '
        'This dataset contains fire records, not representative non-fire days. '
        'Area = 0 does not establish absence of fire. This app does not estimate daily ignition probability.')
if not (ROOT / 'artifacts/models.joblib').exists():
    st.error('Models are missing. From the project folder run: python -m wildfire.train')
    st.stop()
bundle = load_bundle()
models = bundle['models']
with st.sidebar:
    st.header('Study controls')
    selected = st.selectbox('Probability model', list(models), key='prediction_model')
    st.caption('BMA is the primary model. Select alternatives for the same inputs.')
    st.write(f"Validation-selected threshold: **{bundle['thresholds'][selected]:.2f}**")
    st.caption('Threshold maximizes validation F1. It is not an operational warning level.')
    st.write('Data SHA-256')
    st.code(bundle['metadata']['data_sha256'][:24] + '…')

predict_tab, batch_tab, compare_tab, weights_tab, methods_tab = st.tabs(
    ['Predict', 'Batch CSV', 'Measured performance', 'Bayesian weights', 'Study design'])

with predict_tab:
    st.subheader('Explore a recorded-fire scenario')
    st.caption('Enter measured weather and externally computed Fire Weather Index components. '
               'The app does not derive the indices from a single day of weather.')
    with st.form('scenario_form'):
        a, b, c = st.columns(3)
        temp = a.number_input('Temperature (°C)', value=20.0, step=.5)
        rh = b.number_input('Relative humidity (%)', min_value=0.0, max_value=100.0, value=45.0)
        wind = c.number_input('Wind speed (km/h)', min_value=0.0, value=4.0, step=.1)
        rain = a.number_input('Rain (mm/m², UCI units)', min_value=0.0, value=0.0, step=.1)
        ffmc = b.number_input('FFMC · fine fuel moisture code', min_value=0.0, max_value=101.0, value=90.0)
        dmc = c.number_input('DMC · duff moisture code', min_value=0.0, value=110.0)
        dc = a.number_input('DC · drought code', min_value=0.0, value=550.0)
        isi = b.number_input('ISI · initial spread index', min_value=0.0, value=8.0)
        month = c.selectbox('Month', MONTHS, index=7)
        submitted = st.form_submit_button('Estimate probability', type='primary')
    if submitted:
        st.session_state['scenario'] = pd.DataFrame([dict(temp=temp, RH=rh, wind=wind, rain=rain,
                                                         FFMC=ffmc, DMC=dmc, DC=dc, ISI=isi, month=month)])
    if 'scenario' in st.session_state:
        raw = validate_inputs(st.session_state['scenario'])
        X = feature_frame(raw)
        p = float(models[selected].predict_proba(X)[0, 1])
        left, right = st.columns([1, 2])
        left.metric('Positive recorded-area probability', f'{p:.1%}')
        left.progress(p)
        left.write('Predicted class: **' + ('area > 0**' if p >= bundle['thresholds'][selected] else 'area = 0**'))
        comparison = pd.DataFrame({'Model': list(models), 'Probability': [float(m.predict_proba(X)[0, 1]) for m in models.values()]})
        right.bar_chart(comparison.set_index('Model'), horizontal=True, color='#377c75')
        for message in range_messages(raw, bundle): st.warning(message)
        bma_pipe = models['BMA']; bma = bma_pipe.named_steps['model']
        candidate_p = bma.candidate_probabilities(bma_pipe.named_steps['scale'].transform(X))[0]
        mean = float(candidate_p @ bma.weights_)
        spread = float(np.sqrt(np.sum(bma.weights_ * (candidate_p - mean)**2)))
        st.caption(f'BMA between-model probability SD: {spread:.3f}. This describes disagreement '
                   'between candidate means; it is not a confidence interval or total predictive uncertainty.')

with batch_tab:
    st.subheader('Score multiple scenarios')
    sample = pd.DataFrame([dict(temp=20, RH=45, wind=4, rain=0, FFMC=90, DMC=110, DC=550, ISI=8, month='aug')])
    st.download_button('Download CSV template', sample.to_csv(index=False), 'scenario_template.csv', 'text/csv')
    uploaded = st.file_uploader('Upload CSV with the nine template columns (maximum 5,000 rows)', type='csv')
    if uploaded is not None:
        try:
            if uploaded.size > 2_000_000: raise ValueError('CSV must be smaller than 2 MB.')
            raw = pd.read_csv(uploaded)
            if not 1 <= len(raw) <= 5000: raise ValueError('Upload between 1 and 5,000 rows.')
            raw = validate_inputs(raw); X = feature_frame(raw); scored = raw.copy()
            for name, model in models.items():
                scored[name + '_probability'] = model.predict_proba(X)[:, 1]
            scored['selected_model_class'] = (scored[selected + '_probability'] >= bundle['thresholds'][selected]).astype(int)
            for message in range_messages(raw, bundle): st.warning(message)
            st.dataframe(scored, hide_index=True)
            st.download_button('Download scored CSV', scored.to_csv(index=False), 'scored_scenarios.csv', 'text/csv')
        except (ValueError, TypeError, pd.errors.ParserError) as exc:
            st.error(str(exc))

with compare_tab:
    st.subheader('One held-out test set, measured results')
    results = pd.read_csv(ROOT / 'artifacts/model_comparison.csv')
    st.dataframe(results, hide_index=True)
    st.info('Measured signal is weak on this split: BMA ROC-AUC is close to 0.5. '
            'A high F1 can result from predicting every record positive; compare the prevalence baseline '
            'and F1 at 0.5 before interpreting performance.')
    st.download_button('Download comparison table', results.to_csv(index=False), 'model_comparison.csv', 'text/csv')
    st.caption('Lower Brier is better; higher ROC-AUC and F1 are better. Validation F1 selected thresholds '
               'and is optimistic. Validation Brier was not used for tuning. Timings depend on hardware; '
               'inference is warmed batch time divided by batch size.')
    st.image(str(ROOT / 'artifacts/roc_calibration.png'))
    st.image(str(ROOT / 'artifacts/comparison.png'))
    st.dataframe(pd.read_csv(ROOT / 'artifacts/calibration_bins.csv'), hide_index=True)
    st.caption('Brier assesses overall probability quality, including discrimination; it is not a pure '
               'calibration measure. Small bin counts limit calibration conclusions.')

with weights_tab:
    st.subheader('How the Bayesian average is assembled')
    st.image(str(ROOT / 'artifacts/model_weights.png'))
    st.dataframe(pd.read_csv(ROOT / 'artifacts/bma_weights.csv'), hide_index=True)
    st.bar_chart(pd.read_csv(ROOT / 'artifacts/block_inclusion.csv').set_index('block'))
    st.caption('All 16 block subsets include an intercept. Equal model priors; independent Gaussian '
               'coefficient priors. Laplace marginal likelihoods yield approximate posterior weights. '
               'Weights indicate support within this model family, not causal effects.')

with methods_tab:
    st.subheader('Target, leakage controls and reproducibility')
    st.json(bundle['metadata']['split_summary'])
    st.markdown('''
* **Target:** `1[area > 0]`, a positive recorded-area proxy conditional on a recorded fire.
* **Spatial split:** entire `(X, Y)` cells assigned to train, validation or test. Fractions are by cells, so row counts vary.
* **No chronology claim:** month/day names lack year and exact dates; a reliable chronological split is unavailable.
* **Weather:** rain uses `log1p`; month uses sine/cosine. Scaling is fit within each training fold.
* **Features:** temperature, RH, wind, rain, FFMC, DMC, DC, ISI and month. Area is only the label; X/Y are only split groups; day is excluded.
* **Selection:** four-fold grouped CV within training minimizes Brier. Validation chooses F1 thresholds. Test is evaluated after decisions are fixed.
* **Calibration:** inspect held-out reliability curves, bin counts and Brier; no test-fitted calibration transformation.
* **Limits:** 517 records from one park, coarse groups, uncertain event dependence and no representative non-fire controls. Spatial grouping cannot remove all shared-weather dependence.
''')
    st.link_button('UCI dataset and attribution', 'https://archive.ics.uci.edu/dataset/162/forest%2Bfires')
    st.code('python -m wildfire.train\npython -m streamlit run app.py')
