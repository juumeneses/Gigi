import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch

from Treinamento_DINOv2_face_cranio import core, engine, ensemble, fusion, improvement_pipeline as pipeline
from Treinamento_DINOv2_face_cranio.preprocessing import prepare_image
from Treinamento_DINOv2_face_cranio.tests.test_pipeline import FakeExtractor, fixture


class ImprovementPipelineTest(unittest.TestCase):
    def test_numerical_zero_in_confidence_interval_is_inconclusive(self):
        baseline=dict(galleries={'a':['a','b'],'b':['a','b']},rows=[dict(person='a',rank=2),dict(person='b',rank=2)])
        selected=dict(baseline,rows=[dict(person='a',rank=1),dict(person='b',rank=2)])
        with patch.object(pipeline.np,'quantile',return_value=np.array([6.9e-18,.5])):
            result=pipeline.paired_gain([selected],[baseline])
        self.assertEqual(result['ci95'][0],0.)
        self.assertEqual(result['conclusion'],'ganho inconclusivo')
        with patch.object(pipeline.np,'quantile',return_value=np.array([.01,.5])):
            self.assertEqual(pipeline.paired_gain([selected],[baseline])['conclusion'],'melhora')

    def test_trim_preserves_foreground_and_falls_back(self):
        im = Image.new('RGB',(200,300),'black')
        ImageDraw.Draw(im).rectangle((60,70,140,230),fill='white')
        crop,meta = prepare_image(im,'uniform_background')
        self.assertTrue(meta['applied'])
        self.assertLess(meta['box'][0],60)
        self.assertGreater(meta['box'][2],140)
        self.assertGreater(crop.height,crop.width)
        original,meta = prepare_image(Image.new('RGB',(100,200),'white'),'uniform_background')
        self.assertFalse(meta['applied'])
        self.assertEqual(original.size,(100,200))
        original,meta = prepare_image(im)
        self.assertEqual(meta['box'],[0,0,200,300])

    def test_hard_mining_excludes_validation_and_has_no_duplicates(self):
        model = engine.Towers()
        usable = [0,1,2,3]
        matrix = np.array([[1.,.99,.1,.2],[.99,1.,.1,.2],[.1,.1,1.,.98],[.2,.2,.98,1.]])
        with patch.object(engine,'score_matrix',return_value=matrix) as score:
            batches = engine.training_batches(model,{},'E1',usable,2,42,0,True,1)
        self.assertEqual(score.call_args.args[-1],usable)
        self.assertEqual(sorted(i for b in batches for i in b),usable)
        self.assertEqual({frozenset(b) for b in batches},{frozenset((0,1)),frozenset((2,3))})

    @patch.object(engine,'Extractor',FakeExtractor)
    def test_full_factorial_resume_and_same_ensemble_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest,people = fixture(root)
            def fresh(kind):
                path = root/(kind+'_'+str(len(list(root.iterdir()))))
                path.mkdir()
                return path
            args = dict(manifest=str(manifest),run_dir=root/'pipeline',losses=['infonce'],seeds=[42,43,44],
                        epochs=1,batch=2,hard_warmup=0,smoke=True,log=lambda *a,**k:None)
            with patch.object(fusion,'new_run',side_effect=fresh),patch.object(ensemble,'new_run',side_effect=fresh):
                run = pipeline.run_pipeline(**args)
                self.assertEqual(core.load(run/'status.json')['state'],'CONCLUIDO')
                summary = core.load(run/'summary.json')
                self.assertEqual(len(summary['rows']),8)
                self.assertEqual(len(summary['subsets']),112)
                original = next(r for r in summary['rows'] if r['condition']=='original' and r['kind']=='ensemble')
                model = Path(original['outputs'][0])
                val = [p for p in people if p['split']=='validacao']
                output = fusion.compare_fusion(model,val[0],val,weights=[.5,.3,.2],log=lambda *a,**k:None)
                with np.load(model/'scores.npz') as arrays:
                    scores = arrays['combinado']
                for row in core.load(output/'result.json')['ranking']:
                    j = next(i for i,p in enumerate(val) if p['person']==row['person'])
                    self.assertAlmostEqual(row['score'],float(scores[0,j]),places=5)
                before = core.load(run/'ledger.json')
                with patch.object(fusion,'train_components',side_effect=AssertionError('nao deve treinar novamente')):
                    pipeline.run_pipeline(**args)
                self.assertEqual(before,core.load(run/'ledger.json'))
                mining = core.load(run/'infonce/hard/seed42/frontal/mining_last.json')
                self.assertTrue(all(pid in {p['person'] for p in people if p['split']=='treino'} for b in mining['batches'] for pid in b))
                model_file = run/'infonce/original/seed42/frontal/best.pt'
                model_file.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'alterada'):
                    pipeline.run_pipeline(**args)

    @patch.object(engine,'Extractor',FakeExtractor)
    def test_interrupt_then_resume_does_not_duplicate_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest,_=fixture(root)
            stop=root/'stop.flag'
            def fresh(kind):
                target=root/(kind+'_'+str(len(list(root.iterdir()))))
                target.mkdir()
                return target
            def log(message,**kwargs):
                if '/scores' in message:
                    stop.touch()
            args=dict(manifest=str(manifest),run_dir=root/'pipeline',conditions=[],losses=['triplet'],seeds=[42,43,44],epochs=1,batch=2,smoke=True)
            with patch.object(fusion,'new_run',side_effect=fresh),patch.object(ensemble,'new_run',side_effect=fresh):
                run=pipeline.run_pipeline(**args,stop=stop,log=log)
                self.assertEqual(core.load(run/'status.json')['state'],'INTERROMPIDO')
                initial=core.sha(run/'triplet/original/seed42/frontal/best.pt')
                stop.unlink()
                run=pipeline.run_pipeline(**args,log=lambda *a,**k:None)
                self.assertEqual(core.load(run/'status.json')['state'],'CONCLUIDO')
                self.assertEqual(initial,core.sha(run/'triplet/original/seed42/frontal/best.pt'))
                self.assertEqual(len(core.load(run/'ledger.json')),7)


if __name__=='__main__':
    unittest.main()
