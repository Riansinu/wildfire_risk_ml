from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_loads_and_scenario_prediction():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.metric and '%' in app.metric[0].value
    app.selectbox(key='prediction_model').select('Random Forest').run(timeout=30)
    assert not app.exception and app.metric
