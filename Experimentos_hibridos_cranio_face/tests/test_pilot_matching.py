import unittest
import json
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.region_pilot import RUNS, LAB_ROOT, key
from src.pilot_matching_ui import geometry, ranked


class MatchingPilotTests(unittest.TestCase):
    def test_matrix(self):
        run=RUNS/'piloto_20260929_v3'
        items=json.loads((run/'entradas.json').read_text())
        extracts={(i['id'],i['modalidade']):json.loads((run/key(i)/'extracao/data.json').read_text()) for i in items}
        result=geometry(items,extracts)
        self.assertEqual(len(result),121)
        self.assertEqual(result.correto.sum(),11)
        self.assertFalse(result.duplicated(['face_id','cranio_id']).any())
        self.assertTrue((result.medidas>=2).all())

    def test_ties(self):
        result=ranked(pd.DataFrame({'face_id':['a','a'],'cranio_id':['a','b'],'score':[1.,1.]}))
        self.assertEqual(result['rank'].tolist(),[2.,2.])

    def test_pages(self):
        app=AppTest.from_file(str(LAB_ROOT/'revisao_cranios.py')).run(timeout=30)
        app.radio[0].set_value('Imagens teste piloto').run()
        self.assertFalse(app.exception)
        for section in ['Crânios marcados','Comparação dos pares','Matching']:
            next(r for r in app.radio if r.label=='Etapa do piloto').set_value(section).run()
            self.assertFalse(app.exception)
        next(s for s in app.selectbox if s.label=='Método').set_value('Duas torres existente').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)


if __name__=='__main__': unittest.main()
