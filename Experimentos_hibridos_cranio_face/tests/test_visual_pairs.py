import hashlib
import json
import unittest

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from scripts.lote_pares_visuais import ROOT, PROPOSALS, original_xy
from src.common import LAB_ROOT, REPO_ROOT


class VisualPairsTest(unittest.TestCase):
    def test_saved_proposals_and_originals(self):
        folder = sorted((ROOT/'runs').glob('*/lote.json'))[-1].parent
        data = json.loads((folder/'lote.json').read_text())
        frame = pd.read_csv(folder/'pontos.csv')
        self.assertEqual(len(data['entradas']), 10)
        self.assertEqual(len(frame), 220)
        self.assertFalse(frame.validado.any())
        self.assertFalse(data['treinamento_autorizado'])
        self.assertFalse(frame.duplicated(['id_pessoa','modalidade','numero']).any())
        self.assertEqual(int(frame.estado.eq('DUVIDA').sum()), 5)
        for entry in data['entradas']:
            for kind, view in entry['views'].items():
                self.assertEqual(hashlib.sha256((REPO_ROOT/view['path']).read_bytes()).hexdigest(), view['hash'])
                with Image.open(REPO_ROOT/view['path']) as source:
                    image = source.copy()
                for i, coord in enumerate(PROPOSALS[entry['id_pessoa']][kind], 1):
                    row = frame[(frame.id_pessoa==entry['id_pessoa'])&(frame.modalidade==kind)&(frame.numero==i)].iloc[0]
                    mapped = original_xy(view,coord,kind)
                    if mapped is None:
                        self.assertTrue(pd.isna(row.x_px) and pd.isna(row.y_px))
                    else:
                        self.assertEqual(mapped,(row.x_px,row.y_px))
                        self.assertTrue(0 <= row.x_px < image.width and 0 <= row.y_px < image.height)
                with Image.open(folder/entry['id_pessoa']/(kind+'_marcado.jpg')) as preview:
                    self.assertEqual(preview.size,image.size)

    def test_panel_navigation_filter_and_zoom(self):
        app = AppTest.from_file(str(LAB_ROOT/'revisao_cranios.py')).run(timeout=30)
        self.assertFalse(app.exception)
        app.radio[0].set_value('Pares face e crânio').run()
        select=lambda label: next(s for s in app.selectbox if s.label==label)
        self.assertEqual(select('Par face e crânio').value,'2017_41_GP')
        app.button[1].click().run()
        self.assertEqual(select('Par face e crânio').value,'2017_73_MMPCC')
        app.button[0].click().run()
        self.assertEqual(select('Par face e crânio').value,'2017_41_GP')
        select('Ano dos pares').set_value(2019).run()
        self.assertEqual(len(select('Par face e crânio').options),5)
        select('Número do ponto').set_value(7).run()
        app.radio[1].set_value('Original').run()
        self.assertFalse(app.exception)


if __name__=='__main__':
    unittest.main()
