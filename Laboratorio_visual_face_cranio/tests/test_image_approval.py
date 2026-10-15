import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from streamlit.testing.v1 import AppTest

from Laboratorio_visual_face_cranio.core import save, load, digest, sha, write_image
from Laboratorio_visual_face_cranio.references import confirm_image_points, review, promote, current_reviews
from Laboratorio_visual_face_cranio.data import training_samples


class ImageApprovalTests(unittest.TestCase):
    def test_bulk_approval_preserves_corrections_and_absences(self):
        points=[dict(numero=n,x=10.,y=20.,visibility='NAO_AVALIADO') for n in range(1,12)]
        points[1].update(x=33.,y=44.,visibility='VISIVEL')
        points[2].update(x=None,y=None,visibility='OCULTO')
        points[3].update(x=None,y=None)
        confirmed=confirm_image_points(points)
        self.assertEqual(confirmed[0]['visibility'],'VISIVEL')
        self.assertEqual((confirmed[1]['x'],confirmed[1]['y']),(33.,44.))
        self.assertEqual(confirmed[2]['visibility'],'OCULTO')
        self.assertEqual(confirmed[3]['visibility'],'INDETERMINADO')
        self.assertEqual(points[0]['visibility'],'NAO_AVALIADO')

    def test_confirmed_image_enters_training_reference_with_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            batch=root/'annotation_batches'/'batch'
            samples=[]
            proposals=[]
            for key,split in [('a','treino'),('b','validacao')]:
                original=root/(key+'.png')
                write_image(original,np.full((80,100,3),30 if key=='a' else 60,np.uint8))
                points=[dict(numero=n,x=20.+n,y=30.,visibility='NAO_AVALIADO') for n in range(1,12)]
                points[1].update(x=None,y=None)
                row=dict(key=key,path=str(original),hash=sha(original),person=key,split=split,
                         modality='face',view='FFN',filename=key+'.png',width=100,height=80,points=points,status='PROPOSTA',bank_keys=[key])
                proposal=batch/key/'proposal.json'
                save(proposal,row)
                samples.append(row)
                proposals.append(proposal)
            base=dict(samples=samples,failures=[])
            base['fingerprint']=digest(base)
            reference=root/'references'/'base'/'references.json'
            save(reference,base)
            with patch('Laboratorio_visual_face_cranio.core.ARTIFACTS',root):
                decision=review(proposals[0],confirm_image_points(samples[0]['points']),'Test reviewer')
                ready=promote(reference,batch,log=lambda message:None)/'references.json'
            latest=current_reviews(batch)
            self.assertIn(str(proposals[0]),latest)
            self.assertNotIn(str(proposals[1]),latest)
            data,tr,val,supported=training_samples(ready,'face','FFN')
            self.assertEqual(tr[0]['source'],str(decision))
            self.assertEqual(tr[0]['points'][0]['visibility'],'VISIVEL')
            self.assertIsNone(tr[0]['points'][1]['x'])
            self.assertFalse(supported[1])
            self.assertEqual(load(proposals[0])['points'][0]['visibility'],'NAO_AVALIADO')
            save(batch/'config.json',dict(refinement='test'))
            script=f'''
from pathlib import Path
import streamlit as st
from Laboratorio_visual_face_cranio.core import load
from Laboratorio_visual_face_cranio.workflows_ui import expansion_panel
reference=Path(st.session_state.pop('next_reference',{str(ready)!r}))
expansion_panel(reference,load(reference)['samples'],lambda *a,**kw:None)
'''
            with patch('Laboratorio_visual_face_cranio.core.ARTIFACTS',root), \
                 patch('Laboratorio_visual_face_cranio.workflows_ui.ARTIFACTS',root), \
                 patch('Laboratorio_visual_face_cranio.point_editor.ARTIFACTS',root):
                app=AppTest.from_string(script).run()
                self.assertFalse(app.exception)
                next(b for b in app.button if b.label=='OK · salvar e preparar para treinamento').click().run()
                self.assertFalse(app.exception)
                metrics={m.label:m.value for m in app.metric}
                self.assertEqual(metrics['Pendentes de revisão'],'0')
                self.assertEqual(metrics['Prontas para treinamento'],'2')
                self.assertEqual(current_reviews(batch)[str(proposals[1])][1]['author'],'Usuário do painel')
            write_image(Path(samples[0]['path']),np.zeros((80,100,3),np.uint8))
            self.assertNotIn(str(proposals[0]),current_reviews(batch))


if __name__=='__main__':
    unittest.main()
