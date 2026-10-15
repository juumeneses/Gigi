import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from Treinamento_DINOv2_face_cranio import core, matching_breakdown as breakdown


class BreakdownTest(unittest.TestCase):
    def test_seven_subsets_missing_views_and_primary_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();out=root/'output';out.mkdir()
            image=root/'image.png';Image.new('RGB',(20,20)).save(image)
            row=dict(path=str(image),hash=core.sha(image))
            people=[dict(person=str(i),year='2022',split='validacao',views={v:dict(row) for v in ('FFN','FFS','FL','CF','CD')}) for i in range(2)]
            core.save(model/'config.json',dict(weights=[1,0,0],normalization=[dict(mean=0,std=1)]*3))
            core.save(model/'cohort.json',dict(people=people))
            matrix=np.array([[.1,.9],[.3,.7]])
            core.np_save(model/'scores.npz',frontal=matrix,sorriso=matrix[:,::-1],lateral=matrix)
            result=dict(model=str(model),query=people[0],candidates=people,weights=[1,0,0],ranking=breakdown.ranking(matrix[0:1],people))
            core.save(out/'result.json',result)
            before=core.sha(out/'result.json')
            with patch.object(breakdown.fusion,'compare_fusion',side_effect=AssertionError('Cached scores must avoid inference')):
                breakdown.build(out,log=lambda *a,**kw:None)
                report=core.load(out/'model_breakdown.json')
                self.assertEqual(len(report['rows']),8)
                self.assertEqual(report['rows'][0]['Posicao do correto'],2)
                self.assertEqual(next(r for r in report['rows'] if r['Combinacao']=='Sorrindo')['Posicao do correto'],1)
                self.assertEqual(core.sha(out/'result.json'),before)
                token=report['signature']
                del result['query']['views']['FFS']
                core.save(out/'result.json',result)
                breakdown.build(out,log=lambda *a,**kw:None)
                report=core.load(out/'model_breakdown.json')
                self.assertNotEqual(token,report['signature'])
                self.assertIsNone(next(r for r in report['rows'] if r['Combinacao']=='Sorrindo')['Posicao do correto'])

    def test_unknown_upload_never_reuses_identity_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'image.png';Image.new('RGB',(10,10)).save(image)
            people=[dict(person='known',split='validacao',views=dict(FFN=dict(path=str(image),hash=core.sha(image))))]
            core.save(root/'cohort.json',dict(people=people))
            self.assertIsNone(breakdown.cached_matrices(root,dict(person='upload',views={}),people))
            query=dict(person='known',views=dict(FFN=dict(path=str(image))))
            Image.new('RGB',(10,10),'red').save(image)
            self.assertIsNone(breakdown.cached_matrices(root,query,people))

    def test_upload_fallback_is_explicit_and_keeps_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();out=root/'out';out.mkdir();local=root/'local';local.mkdir()
            core.save(model/'config.json',dict(weights=[1,0,0]))
            core.save(model/'cohort.json',dict(people=[]))
            query=dict(person='upload',views={});candidates=[dict(person='a',views={}),dict(person='b',views={})]
            ranked=[dict(person='a',rank=1,score=.8),dict(person='b',rank=2,score=.2)]
            core.save(out/'result.json',dict(model=str(model),query=query,candidates=candidates,weights=[1,0,0],ranking=ranked))
            before=core.sha(out/'result.json')
            self.assertIsNone(breakdown.build(out,log=lambda *a,**kw:None))
            combos={bits:dict(weights=[float(b)*w for b,w in zip(bits,[1,.3,.2])],ranking=ranked) for bits in breakdown.SUBSETS}
            core.save(local/'result.json',dict(combinations=combos))
            with patch.object(breakdown.fusion,'compare_fusion',return_value=local) as compare:
                breakdown.build(out,allow_inference=True,log=lambda *a,**kw:None)
                self.assertEqual(compare.call_count,1)
                self.assertTrue(compare.call_args.kwargs['all_subsets'])
            self.assertEqual(core.sha(out/'result.json'),before)
            self.assertEqual(len(core.load(out/'model_breakdown.json')['rows']),8)
