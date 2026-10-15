import tempfile
import unittest
from unittest.mock import patch

from Treinamento_DINOv2_face_cranio import matching_batch as batch
from Treinamento_DINOv2_face_cranio import core,engine
from Treinamento_DINOv2_face_cranio.reporting import newest_first


class BatchTest(unittest.TestCase):
    def test_samples_are_stable_and_distinct(self):
        people=[dict(person=str(i)) for i in range(15)]
        selections=batch.samples(people,10,10,42)
        self.assertEqual(selections,batch.samples(list(reversed(people)),10,10,42))
        self.assertEqual(len({s['query']['person'] for s in selections}),10)
        for s in selections:
            ids=[p['person'] for p in s['candidates']]
            self.assertEqual(len(set(ids)),10)
            self.assertEqual(ids.count(s['query']['person']),1)
        with self.assertRaises(ValueError):batch.samples(people,20,10)

    def test_saved_inference_and_failures_count_in_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=core.Path(tmp);model=root/'model';model.mkdir()
            core.save(model/'config.json',dict(experiment='E1'))
            core.save(model/'cohort.json',dict(people=[dict(person=str(i),split='validacao') for i in range(12)]))
            def fresh(kind):
                p=root/kind/str(len(list(root.rglob('result.json'))));p.mkdir(parents=True,exist_ok=True);return p
            calls=[]
            def compare(run,query,candidates,**kwargs):
                calls.append(query['person'])
                if len(calls)==2:raise ValueError('Imagem ilegivel')
                ranking=[dict(person=query['person'],score=.9,rank=1)]
                return dict(model=run,query=query,candidates=candidates,ranking=ranking)
            with patch.object(batch,'new_run',side_effect=fresh),patch.object(engine,'compare',side_effect=compare):
                run=batch.run_batch(str(model),log=lambda *a,**k:None)
            report=core.load(run/'summary.json')
            self.assertEqual(report['top_1'],dict(hits=9,total=10,recall=.9))
            self.assertEqual(report['failures'],1)
            self.assertEqual(report['processed'],10)
            self.assertEqual(len(set(calls)),10)
            self.assertEqual(core.load(run/'status.json')['state'],'CONCLUIDO')

    def test_nested_runs_sort_by_time_not_directory(self):
        old=core.Path('/suites/20261014T100000000000Z/z/config.json')
        new=core.Path('/runs/20261015T100000000000Z/config.json')
        self.assertEqual(newest_first([old,new]),[new,old])

    def test_batch_panel_shows_saved_counts(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as tmp:
            root=core.Path(tmp);run=root/'matching_batches/20261015T130000000000Z';run.mkdir(parents=True)
            rows=[{'Top-1':'Sim' if i<6 else 'Nao','Top-3':'Sim' if i<8 else 'Nao','Top-5':'Sim',
                   'Posicao do correto':1 if i<6 else 3 if i<8 else 5,'output':None} for i in range(10)]
            core.save(run/'summary.json',dict(**batch.summarize(rows,10),rows=rows,combinations={}))
            core.save(run/'status.json',dict(state='CONCLUIDO'))
            core.save(run/'request.json',dict(seed=42))
            with patch.object(core,'ARTIFACTS',root):
                app=AppTest.from_string('from Treinamento_DINOv2_face_cranio.matching_batch import results\nresults()').run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual([m.value for m in app.metric],['6/10','8/10','10/10'])


if __name__=='__main__':unittest.main()
