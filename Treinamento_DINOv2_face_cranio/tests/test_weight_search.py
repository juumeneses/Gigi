import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Treinamento_DINOv2_face_cranio import core, engine, fusion, weight_search


class WeightSearchTest(unittest.TestCase):
    def test_top1_is_prioritized_over_higher_mrr(self):
        self.assertGreater(engine.selection_key(dict(recall_1=.4, mrr=.52)),
                           engine.selection_key(dict(recall_1=.2, mrr=.6)))
        ids = list('abcde')
        def scores(ranks):
            matrix = np.full((5,5), -1.)
            for i, rank in enumerate(ranks):
                matrix[i, [j for j in range(5) if j != i][:rank-1]] = 1.
                matrix[i,i] = 0.
            return matrix
        a, b = scores([1,1,5,5,5]), scores([1,2,2,2,2])
        datasets = [dict(ids=ids, matrices=[a,b,b], stats=[dict(mean=0.,std=1.)]*3, seed=s) for s in (42,43,44)]
        rows, selected, _ = weight_search.optimize(datasets, step=1., log=lambda *a, **k: None)
        self.assertEqual(rows[selected]['frontal'], 1.)
        rows, selected, _ = weight_search.optimize(datasets, step=1., objective='mrr', log=lambda *a, **k: None)
        self.assertAlmostEqual(rows[selected]['recall_1'], .2)
        self.assertGreater(rows[selected]['mrr'], .52)

    def test_ranks_match_existing_evaluator_with_ties_and_missing(self):
        ids = ['z', 'a', 'm']
        people = [dict(person=p, year='2020') for p in ids]
        matrix = np.array([[.4, .4, .2], [.3, .3, .1], [.1, .2, -np.inf]])
        fast = weight_search.rank_metrics(matrix, ids)
        expected = engine.evaluation(matrix, people)['full']
        for key in fast:
            self.assertEqual(fast[key], expected[key])
        self.assertEqual(fast['failures'], 1)
        for step in (0, -.1, float('nan'), .3):
            with self.assertRaises(ValueError):
                weight_search.grid_weights(step)
        self.assertTrue(any(np.allclose(w, [1/3]*3) for w in weight_search.grid_weights(.1)))

    def test_search_exports_matching_runs_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            people = [dict(person=p, year='2020', split='validacao', views={}) for p in ('a','b','c')]
            co = dict(fingerprint='same', people=people)
            good = np.eye(3)
            bad = np.roll(good, 1, axis=1)*2
            sources = []
            checkpoints = []
            for seed in (42,43,44):
                source = root/str(seed)
                source.mkdir()
                children = {}
                for name, exp in fusion.COMPONENTS.items():
                    child = source/name
                    child.mkdir()
                    (child/'best.pt').write_bytes(b'unchanged')
                    checkpoints.append(child/'best.pt')
                    core.save(child/'config.json', dict(seed=seed, cohort='same', experiment=exp, code_hash='fixed'))
                    children[name] = dict(run=str(child), checkpoint_hash=core.sha(child/'best.pt'))
                cfg = dict(split='validacao', smoke=False, cohort='same', children=children,
                    weights=[1,1,1], normalization=[dict(mean=0., std=1.)]*3)
                core.save(source/'config.json', cfg)
                core.save(source/'cohort.json', co)
                core.np_save(source/'scores.npz', frontal=good, sorriso=bad, lateral=bad)
                evaluation = engine.evaluation((good+bad+bad)/3, people)
                core.save(source/'results.json', dict(combinado=evaluation))
                sources.append(source)
            def fresh(kind):
                output = root/(kind+'_'+str(len(list(root.iterdir()))))
                output.mkdir()
                return output
            with patch.object(weight_search, 'new_run', side_effect=fresh):
                run = weight_search.search_weights(sources, step=.5, log=lambda *a, **k: None)
            result = core.load(run/'summary.json')
            self.assertEqual(result['best']['mrr'], 1.)
            self.assertGreater(result['delta_mrr'], 0)
            self.assertEqual(len(result['outputs']), 3)
            self.assertEqual(len(result['held_out_seed']), 3)
            for output in result['outputs']:
                config = core.load(Path(output)/'config.json')
                self.assertEqual(config['weights'], result['weights'])
                self.assertEqual(config['weights_selected_on'], 'validacao')
                self.assertEqual(core.load(Path(output)/'status.json')['state'], 'CONCLUIDO')
            self.assertTrue(all(p.read_bytes() == b'unchanged' for p in checkpoints))
            with self.assertRaisesRegex(ValueError, 'tres seeds'):
                weight_search.read_sources(sources[:2])
            cfg = core.load(sources[0]/'config.json')
            core.save(sources[0]/'config.json', dict(cfg, split='teste'))
            with self.assertRaisesRegex(ValueError, 'validacao'):
                weight_search.read_sources(sources)
            core.save(sources[0]/'config.json', cfg)
            checkpoints[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint'):
                weight_search.read_sources(sources)


if __name__ == '__main__':
    unittest.main()
