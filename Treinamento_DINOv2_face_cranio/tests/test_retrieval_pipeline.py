import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Treinamento_DINOv2_face_cranio import core, engine, fusion, ensemble, retrieval_pipeline as pipeline
from Treinamento_DINOv2_face_cranio.tests.test_pipeline import FakeExtractor, fixture


class RetrievalPipelineTest(unittest.TestCase):
    def test_top10_selection_and_mrr_tiebreak(self):
        self.assertGreater(engine.selection_key(dict(recall_1=.1,recall_10=.8,mrr=.2),'recall_10'),
                           engine.selection_key(dict(recall_1=.5,recall_10=.7,mrr=.6),'recall_10'))
        self.assertGreater(engine.selection_key(dict(recall_1=.1,recall_10=.8,mrr=.4),'recall_10'),
                           engine.selection_key(dict(recall_1=.5,recall_10=.8,mrr=.3),'recall_10'))
        with self.assertRaises(ValueError):engine.selection_key({},'unknown')

    @patch.object(engine,'Extractor',FakeExtractor)
    def test_staged_comparison_grid_confirmation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,people=fixture(root)
            # Frozen test identity is catalogued but must never be read/extracted.
            manifest_data=core.load(manifest)
            test_person=dict(people[-1],person='frozen_test',split='teste')
            manifest_data['people'].append(test_person)
            manifest_data['fingerprint']=core.digest({k:v for k,v in manifest_data.items() if k!='fingerprint'})
            core.save(manifest,manifest_data)
            def prepare(model,log):
                folder=root/model;folder.mkdir(exist_ok=True);core.save(folder/'model.json',dict(synthetic=True))
                return folder
            def fresh(kind):
                path=root/(kind+'_'+str(len(list(root.iterdir()))));path.mkdir();return path
            args=dict(manifest=str(manifest),run_dir=root/'pipeline',backbones=['dinov2_vitb14','convnext_tiny'],
                learning_rates=[.001,.0003],margins=[.2,.1],epochs=1,batch=2,smoke=True,log=lambda *a,**kw:None)
            original_features=engine.features
            def features(people,*a,**kw):
                self.assertFalse(any(p['split']=='teste' for p in people))
                return original_features(people,*a,**kw)
            with patch.object(pipeline.backbone,'prepare_variant',side_effect=prepare),patch.object(fusion,'new_run',side_effect=fresh),patch.object(ensemble,'new_run',side_effect=fresh),patch.object(engine,'features',side_effect=features):
                run=pipeline.run_pipeline(**args)
                self.assertEqual(core.load(run/'status.json')['state'],'CONCLUIDO')
                result=core.load(run/'summary.json')
                self.assertEqual(result['primary_metric'],'recall_10')
                self.assertEqual(len(core.load(run/'backbone_comparison.json')),2)
                self.assertEqual(len(core.load(run/'parameter_screening.json')['rows']),4)
                recommendation=core.load(run/'recommended.json')
                self.assertEqual(recommendation['metrics']['n_seeds'],3)
                self.assertTrue(Path(recommendation['output']).exists())
                self.assertTrue(all(core.load(p)['selection_metric']=='recall_10' for p in run.glob('trials/*/seed*/*/config.json')))
                self.assertTrue(all('val_recall_10' in core.load(p)[0] for p in run.glob('trials/*/seed*/*/history.json')))
                baseline=next(r for r in result['rows'] if r['backbone']=='dinov2_vitb14' and r['kind']=='ensemble')
                self.assertTrue(baseline['confirmed'])
                self.assertEqual(baseline['ci95'],[0.,0.])
                ledger=core.load(run/'ledger.json')
                with patch.object(fusion,'train_components',side_effect=AssertionError('Must reuse completed stages')):
                    pipeline.run_pipeline(**args)
                self.assertEqual(ledger,core.load(run/'ledger.json'))
                self.assertEqual(core.load(run/'status.json')['done'],core.load(run/'status.json')['total'])
                checkpoint=next(run.glob('trials/*/seed*/frontal/best.pt'));checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'alterada'):pipeline.run_pipeline(**args)

    @patch.object(engine,'Extractor',FakeExtractor)
    def test_interrupt_resume_and_no_parameter_sweep_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=fixture(root);stop=root/'stop.flag'
            def prepare(model,log):
                folder=root/model;folder.mkdir(exist_ok=True);core.save(folder/'model.json',dict(synthetic=True));return folder
            def fresh(kind):
                path=root/(kind+str(len(list(root.iterdir()))));path.mkdir();return path
            def log(message,**kw):
                if '/scores' in message:stop.touch()
            args=dict(manifest=str(manifest),run_dir=root/'pipeline',backbones=['dinov2_vitb14'],tune=False,epochs=1,batch=2,smoke=True)
            with patch.object(pipeline.backbone,'prepare_variant',side_effect=prepare),patch.object(fusion,'new_run',side_effect=fresh),patch.object(ensemble,'new_run',side_effect=fresh):
                run=pipeline.run_pipeline(**args,stop=stop,log=log)
                self.assertEqual(core.load(run/'status.json')['state'],'INTERROMPIDO')
                checkpoint=next(run.glob('trials/*/seed*/frontal/best.pt'));before=core.sha(checkpoint)
                stop.unlink()
                pipeline.run_pipeline(**args,log=lambda *a,**kw:None)
                self.assertEqual(core.sha(checkpoint),before)
                self.assertEqual(core.load(run/'status.json')['state'],'CONCLUIDO')
                self.assertFalse((run/'parameter_screening.json').exists())


if __name__=='__main__':unittest.main()
