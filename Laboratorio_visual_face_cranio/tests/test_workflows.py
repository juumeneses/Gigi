import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from Laboratorio_visual_face_cranio import core, matching, references, cranial_sex


class WorkflowTests(unittest.TestCase):
    def test_gallery_is_stable_mixed_and_correct_once(self):
        items = [dict(person=f'p{i}', year=str(2017+i%3)) for i in range(20)]
        a = matching.sample_gallery(items, 10, 42)
        self.assertEqual(a, matching.sample_gallery(list(reversed(items)), 10, 42))
        self.assertEqual(sum(r['person'] == a['query']['person'] for r in a['candidates']), 1)
        self.assertEqual(len({r['person'] for r in a['candidates']}), 10)
        self.assertGreater(len({r['year'] for r in a['candidates']}), 1)
        with self.assertRaises(ValueError):
            matching.sample_gallery(items, 21)

    def test_geometry_preserves_scale_and_missingness(self):
        points = [dict(numero=i+1, x=i*5, y=i*8) for i in range(11)]
        a, mask = matching.geometry(points, 100, 200)
        scaled = [dict(numero=p['numero'], x=p['x']*2, y=p['y']*2) for p in points]
        b, _ = matching.geometry(scaled, 200, 400)
        np.testing.assert_allclose(a, b)
        points[0].update(x=None, y=None)
        _, missing = matching.geometry(points, 100, 200)
        self.assertEqual(int(mask.sum()-missing.sum()), 10)

    def test_multiview_slots_keep_missing_side_and_never_transplant_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cranio.png'
            core.write_image(path, np.full((60, 100, 3), 150, np.uint8))
            item = dict(person='p', views={v: dict(path=str(path)) for v in ['CF', 'CD']})
            output = matching.extract(item, 'cranio', None, multiview=True)
            self.assertEqual(output['view_names'], ['CF', 'CE', 'CD'])
            self.assertEqual(output['views'], [1., 0., 1.])
            self.assertEqual(output['points']['CD'], [])

    def test_triplet_gradients_and_failure_denominator(self):
        f = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        s = torch.tensor([[.7, .7], [.7, .7]], requires_grad=True)
        loss = matching.triplet_semihard(f, s)
        loss.backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        metrics = matching.retrieval(np.array([[1., 0.], [0., -np.inf]]), ['a', 'b'])
        self.assertEqual(metrics['recall_1'], .5)
        self.assertEqual(metrics['mrr'], .5)

    def test_batch_excludes_final_test_and_unknown_splits(self):
        rows = []
        ledger = {}
        for i in range(20):
            person = f'p{i}'
            ledger[person] = ['treino', 'validacao', 'teste'][i%3]
            for view in ['FFN', 'CF']:
                rows.append(dict(person=person, year=str(2017+i%2), view=view))
        batch = references.select_batch(rows, count=8, ledger=ledger)
        self.assertEqual(len(batch), 8)
        self.assertTrue(all(r['split'] != 'teste' for r in batch))

    def test_unreviewed_points_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'ARTIFACTS', Path(tmp)):
            path = Path(tmp)/'original.png'
            core.write_image(path, np.full((100, 100, 3), 128, np.uint8))
            proposal = Path(tmp)/'batch'/'p'/'proposal.json'
            core.save(proposal, dict(path=str(path), hash=core.sha(path), width=100, height=100))
            points = [dict(numero=i+1, x=20., y=20., visibility='NAO_AVALIADO') for i in range(11)]
            with self.assertRaises(ValueError):
                references.review(proposal, points, 'tester')
            for p in points:
                p['visibility'] = 'VISIVEL'
            points[0]['visibility'] = 'OCULTO'
            output = references.review(proposal, points, 'tester')
            self.assertIsNone(core.load(output)['points'][0]['x'])

    def test_automatic_labels_are_rejected_as_documented_sex(self):
        frame = pd.DataFrame([dict(id_pessoa='p', sexo='MASCULINO', fonte='FairFace', tipo_referencia='AUTOMATICO')])
        with self.assertRaises(ValueError):
            cranial_sex.import_reference(frame, 'tester', True)
        with self.assertRaises(ValueError):
            cranial_sex.import_reference(frame.assign(tipo_referencia='DOCUMENTAL'), 'tester', False)

    def test_cross_split_identical_image_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.png'
            core.write_image(path, np.zeros((30, 30, 3), np.uint8))
            items = [dict(person='a', split='treino', views={'CF': dict(path=str(path))}),
                     dict(person='b', split='validacao', views={'CF': dict(path=str(path))})]
            with self.assertRaises(ValueError):
                matching.validate_items(items)

    def test_cranial_three_view_training_uses_documented_splits(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'ARTIFACTS', Path(tmp)):
            rows, labels, ledger = [], [], {}
            for i in range(6):
                person = f'p{i}'
                ledger[person] = ['treino', 'validacao', 'teste'][i//2]
                labels.append(dict(id_pessoa=person, sexo='MASCULINO' if i%2 else 'FEMININO', fonte='Registro de teste', tipo_referencia='DOCUMENTAL'))
                for j, view in enumerate(['FFN', 'CF', 'CE', 'CD']):
                    path = Path(tmp)/f'{person}_{view}.png'
                    im = np.zeros((60+i, 60+j, 3), np.uint8)
                    im[5:45+i, 10:40+j] = 150+i+j
                    core.write_image(path, im)
                    rows.append(dict(person=person, year='2017', view=view, path=str(path), key=person+view))
            reference = cranial_sex.import_reference(pd.DataFrame(labels), 'tester', True)
            with patch.object(references, 'splits', return_value=ledger):
                run = cranial_sex.train(rows, str(reference), log=lambda _: None)
                config = core.load(run/'config.json')
                self.assertEqual(config['train_ids'], ['p0', 'p1'])
                self.assertEqual(config['validation_ids'], ['p2', 'p3'])
                out = cranial_sex.apply(run, [r for r in references.paired(rows) if r['split'] == 'teste'], evaluate_test=True, log=lambda _: None)
                predictions = core.load(out/'predictions.json')
                self.assertTrue(all(set(r['per_view']) == {'CF', 'CE', 'CD'} for r in predictions))
                self.assertTrue(all(r['evaluation'] == 'TESTE_INDEPENDENTE' for r in predictions))
                self.assertEqual(core.load(out/'metrics.json')['n'], 2)

    def test_ridge_does_not_reward_an_empty_candidate(self):
        stats = dict(mean=[0.]*55, scale=[1.]*55, supported=[True]*55)
        state = dict(stats=dict(face=stats, cranio=stats), ridge=np.zeros((111, 55)))
        query = [dict(face=dict(geometry=[0.]*55, availability=[True]*55))]
        candidates = [dict(cranio=dict(geometry=[0.]*55, availability=[False]*55)),
                      dict(cranio=dict(geometry=[0.]*55, availability=[True]*55))]
        scores = matching.ridge_scores(query, candidates, state)
        self.assertFalse(np.isfinite(scores[0, 0]))
        self.assertTrue(np.isfinite(scores[0, 1]))

    def test_matching_resume_matches_full_and_inference_filters_keep_original_rank(self):
        class TinyTowers(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.face = nn.Linear(2, 2)
                self.skull = nn.Linear(2, 2)
            def encode(self, features, views, geom, modality):
                head = self.face if modality == 'face' else self.skull
                return torch.nn.functional.normalize(head(features[:, 0, :]), dim=-1)
            def head_state(self):
                return self.state_dict()
            def load_heads(self, values):
                self.load_state_dict(values)

        def extract(item, modality, *args):
            i = int(item['person'][1:])
            return dict(features=[[i/4., 1.]], views=[1.], geometry=[0.]*55,
                        availability=[False]*55, points={}, failures=[])

        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'ARTIFACTS', Path(tmp)), \
                patch.object(matching, 'Towers', TinyTowers), patch.object(matching, 'extract', extract):
            rows, ledger = [], {}
            for i in range(4):
                person = f'p{i}'
                ledger[person] = 'treino' if i < 2 else 'validacao'
                for view in ['FFN', 'CF']:
                    path = Path(tmp)/f'{person}_{view}.png'
                    core.write_image(path, np.full((30, 30, 3), i*20+(1 if view == 'CF' else 0), np.uint8))
                    rows.append(dict(person=person, year='2017', view=view, path=str(path), key=person+view))
            with patch.object(references, 'splits', return_value=ledger):
                full = matching.train(rows, epochs=2, seed=42, log=lambda _: None)
                partial = matching.train(rows, epochs=1, seed=42, log=lambda _: None)
                resumed = matching.train(rows, epochs=2, seed=42, resume=str(partial), log=lambda _: None)
                a = matching.checkpoint_load(full/'last.pt')
                b = matching.checkpoint_load(resumed/'last.pt')
                self.assertEqual(a['history'], b['history'])
                for key in a['heads']:
                    self.assertTrue(torch.equal(a['heads'][key], b['heads'][key]))
                gallery = matching.sample_gallery(references.paired(rows), 2)
                unfiltered = matching.compare(full, gallery['query'], gallery['candidates'], log=lambda _: None)
                sexes = {r['person']: 'FEMININO' for r in gallery['candidates']}
                filtered = matching.compare(full, gallery['query'], gallery['candidates'], reference_sex='MASCULINO', candidate_sexes=sexes, log=lambda _: None)
                self.assertEqual([r['original_rank'] for r in filtered['ranking']], [r['original_rank'] for r in unfiltered['ranking']])
                self.assertTrue(all(r['excluded'] and r['rank'] is None for r in filtered['ranking']))


if __name__ == '__main__':
    unittest.main()
