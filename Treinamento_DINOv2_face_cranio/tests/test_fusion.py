import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Treinamento_DINOv2_face_cranio import core, engine, fusion
from Treinamento_DINOv2_face_cranio.tests.test_pipeline import FakeExtractor, fixture


class FusionTest(unittest.TestCase):
    def test_active_protocol_excludes_ce(self):
        self.assertEqual(core.EXPERIMENTS[fusion.COMPONENTS['lateral']], (['FL'], ['CD']))
        self.assertNotIn('CE', core.EXPERIMENTS['E6D'][1])
        with self.assertRaisesRegex(ValueError, 'legado'):
            engine.train('unused.json', experiment='L1')

    def test_weights_and_missing(self):
        a = np.array([[1., 2.], [3., -np.inf]])
        b = np.ones((2, 2))
        stats = [dict(mean=0., std=1.)]*2
        out = fusion.combine([a, b], stats, [1, 1])
        self.assertEqual(out[0, 1], 1.5)
        self.assertTrue(np.isneginf(out[1, 1]))
        self.assertTrue(np.isfinite(fusion.combine([a, b], stats, [0, 1])).all())
        for weights in ([0, 0], [-1, 2], [float('nan'), 1]):
            with self.assertRaises(ValueError): fusion.combine([a, b], stats, weights)
        s = fusion.normalization(np.array([[999., 1.], [3., 999.]]))
        self.assertEqual(s['mean'], 2.)

    @patch.object(engine, 'Extractor', FakeExtractor)
    @patch.object(fusion, 'Extractor', FakeExtractor)
    def test_training_fusion_inference_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, people = fixture(root)
            bundle = fusion.train_components(str(manifest), run_dir=root/'bundle', epochs=1, batch=2, smoke=True)
            before = core.sha(bundle/'frontal/best.pt')
            fusion.train_components(str(manifest), run_dir=bundle, epochs=1, batch=2, smoke=True)
            self.assertEqual(before, core.sha(bundle/'frontal/best.pt'))
            def fresh(kind):
                path = root/(kind+'_'+str(len(list(root.iterdir()))))
                path.mkdir()
                return path
            with patch.object(fusion, 'new_run', fresh):
                run = fusion.fuse(bundle)
                results = core.load(run/'results.json')
                self.assertEqual(set(results), {'frontal', 'sorriso', 'lateral', 'combinado'})
                val = [p for p in people if p['split'] == 'validacao']
                output = fusion.compare_fusion(str(run), val[0], val)
                ranking = core.load(output/'result.json')['ranking']
                matrix = np.load(run/'scores.npz')['combinado']
                for row in ranking:
                    index = next(i for i, p in enumerate(val) if p['person'] == row['person'])
                    self.assertAlmostEqual(row['score'], matrix[0, index], places=5)
                    self.assertAlmostEqual(row['score'], sum(row[n+'_contribuicao'] for n in fusion.COMPONENTS), places=5)
                self.assertEqual(core.load(run/'config.json')['normalization'][0]['n'], 6)
                with patch.object(engine, 'compare', wraps=engine.compare) as compare:
                    selected = fusion.compare_fusion(str(run), val[0], val, weights=[1, 0, 0])
                    self.assertEqual(compare.call_count, 1)
                    record = core.load(selected/'result.json')
                    self.assertEqual(record['weights'], [1, 0, 0])
                    self.assertTrue(all(r['sorriso'] is None and r['lateral_contribuicao'] == 0 for r in record['ranking']))
                with self.assertRaises(ValueError):
                    fusion.compare_fusion(str(run), val[0], val, weights=[0, 0, 0])
                with patch.object(engine, 'compare', wraps=engine.compare) as compare:
                    combined = fusion.compare_fusion(str(run), val[0], val, all_subsets=True)
                    self.assertEqual(compare.call_count, 3)
                combined = core.load(combined/'result.json')
                self.assertEqual(len(combined['combinations']), 7)
                weighted = core.load(fusion.compare_fusion(str(run), val[0], val,
                    weights=[1, 1, .25], all_subsets=True)/'result.json')
                self.assertEqual(weighted['combinations']['101']['weights'], [1, 0, .25])
                self.assertEqual(weighted['ranking'], weighted['combinations']['111']['ranking'])
                self.assertEqual(combined['ranking'], combined['combinations']['111']['ranking'])
                self.assertEqual([r['person'] for r in record['ranking']],
                                 [r['person'] for r in combined['combinations']['100']['ranking']])
                for entry in combined['combinations'].values():
                    self.assertEqual(len(entry['ranking']), len(val))
                    for r in entry['ranking']:
                        self.assertAlmostEqual(r['score'], sum(r[n+'_contribuicao'] for n in fusion.COMPONENTS), places=5)
                fusion.all_combinations(bundle)
                ledger = core.load(bundle/'combinations.json')
                self.assertEqual(len(ledger['outputs']), 7)
                for key, output in ledger['outputs'].items():
                    cfg = core.load(Path(output)/'config.json')
                    self.assertEqual(cfg['weights'], [int(k) for k in key])
                    self.assertEqual(core.load(Path(output)/'results.json')['combinado']['full']['n'], 3)
                fusion.all_combinations(bundle)
                self.assertEqual(ledger, core.load(bundle/'combinations.json'))


if __name__ == '__main__':
    unittest.main()
