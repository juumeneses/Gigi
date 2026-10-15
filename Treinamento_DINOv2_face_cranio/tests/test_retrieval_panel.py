import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from Treinamento_DINOv2_face_cranio import core, retrieval_controls
from Treinamento_DINOv2_face_cranio.tests.test_pipeline import fixture


class RetrievalPanelTest(unittest.TestCase):
    def test_one_click_uses_defaults_and_always_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,_=fixture(Path(tmp))
            code=f'''
import streamlit as st
from pathlib import Path
from Treinamento_DINOv2_face_cranio.retrieval_controls import controls
def launch(action,args):
    st.session_state['launched']=(action,args)
controls(Path({str(manifest)!r}),launch,lambda:False)
'''
            app=AppTest.from_string(code,default_timeout=30).run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(next(s for s in app.multiselect if s.label=='Extratores do pipeline Top-10').value,['dinov2_vitb14','convnext_tiny'])
            next(b for b in app.button if b.label=='Iniciar pipeline Top-1 / 3 / 5').click().run()
            action,args=app.session_state['launched']
            self.assertEqual(action,'retrieval_pipeline')
            self.assertNotIn('run_dir',args)
            self.assertEqual(args['seeds'],[42,43,44])
            self.assertEqual(args['epochs'],100)
            self.assertEqual(set(args['learning_rates']),{.001,.0003})
            self.assertEqual(set(args['margins']),{.1,.2,.3})
            self.assertTrue(args['tune'])

    def test_results_separate_provisional_and_recommended_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'retrieval_pipelines/20261015T000000000000Z';run.mkdir(parents=True)
            model=root/'fusions/example';model.mkdir(parents=True)
            core.save(run/'request.json',dict(selection_metric='recall_10'))
            core.save(run/'status.json',dict(state='CONCLUIDO',done=10,total=10))
            row=dict(backbone='convnext_tiny',lr=.001,margin=.2,kind='ensemble',n_seeds=3,n=103,
                recall_1=.1,recall_3=.3,recall_5=.4,recall_10=.6,mrr=.2,delta_recall_10=.05,
                conclusion='ganho inconclusivo',confirmed=True)
            provisional=dict(row,kind='individual',n_seeds=1,confirmed=False)
            core.save(run/'summary.json',dict(rows=[row,provisional],examples=[],subsets=[],note='Teste sintetico'))
            core.save(run/'recommended.json',dict(metrics=row,output=str(model)))
            (run/'results.csv').write_text('synthetic\n');(run/'report.html').write_text('<p>synthetic</p>')
            with patch.object(retrieval_controls,'ARTIFACTS',root):
                app=AppTest.from_string('from Treinamento_DINOv2_face_cranio.retrieval_controls import results\nresults()',default_timeout=30).run()
                self.assertFalse(app.exception,[e.message for e in app.exception])
                self.assertTrue(any(m.label=='Rodada selecionada: Recall@1' and m.value=='10.0%' for m in app.metric))
                self.assertTrue(any(t.label=='Confirmacao (3 seeds)' for t in app.tabs))
                next(b for b in app.button if b.label=='Usar recomendado desta rodada no Matching').click().run()
                self.assertEqual(app.session_state['main_area'],'Matching')
                self.assertEqual(app.session_state['fusion_model_choice'],model)
