import unittest

from streamlit.testing.v1 import AppTest


class PendingOriginalsTests(unittest.TestCase):
    def test_selection_and_zoom(self):
        app = AppTest.from_string('''
from pathlib import Path
import pandas as pd
from src.common import LAB_ROOT
from src.region_pilot_ui import render_pending_originals
checks = pd.DataFrame([
    dict(id=i, modalidade='cranio', regiao=r, status='ATENCAO', motivo='Ponto ausente')
    for i in ['2017_41_GP', '2017_76_MVXP'] for r in ['nariz', 'central']])
render_pending_originals(LAB_ROOT/'resultados/piloto_regioes/piloto_20260929_v3', checks)
''').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.selectbox[0].options), 2)
        app.slider[0].set_value(2.).run()
        self.assertFalse(app.exception)
        app.selectbox[0].select(('2017_76_MVXP', 'cranio')).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.slider[0].value, 1.)


if __name__ == '__main__':
    unittest.main()
