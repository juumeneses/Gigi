import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from Treinamento_DINOv2_face_cranio import core, fusion, rank_blend, model_profile, quality


class RankBlendTest(unittest.TestCase):
    def test_ranks_missing_ties_and_weights(self):
        ids = ['z','a','m']
        scores = np.array([[.1,.1,np.nan]])
        np.testing.assert_equal(rank_blend.ordered_ranks(scores,ids),[[2,1,np.inf]])
        b = np.array([[.4,.3,.9]])
        result, parts = rank_blend.combine([scores,b],ids,[.65,.35],1)
        self.assertTrue(np.isneginf(result[0,2]))
        self.assertAlmostEqual(result[0,0],sum(p[1][0,0] for p in parts))
        self.assertTrue(np.isfinite(rank_blend.combine([scores,b],ids,[0,1],1)[0]).all())
        for w in ([0,0],[-1,1],[1,float('nan')]):
            with self.assertRaises(ValueError):rank_blend.combine([scores,b],ids,w)

    def test_search_profile_and_same_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            people = [dict(person=str(i),year='2022',split='validacao',views={}) for i in range(12)]
            rng = np.random.default_rng(15)
            sources = []
            matrices = []
            for j in range(2):
                source = root/f'source{j}'; source.mkdir();sources.append(source)
                matrix = rng.normal(size=(12,12)); matrices.append(matrix)
                core.save(source/'config.json',dict(cohort='same',split='validacao',weights=[1,0,0],normalization=[dict(mean=0,std=1)]*3))
                core.save(source/'cohort.json',dict(people=people))
                core.np_save(source/'scores.npz',frontal=matrix,sorriso=matrix,lateral=matrix)
                core.save(source/'results.json',{})
            def fresh(kind):
                path = root/(kind+str(len(list(root.iterdir()))));path.mkdir();return path
            with patch.object(rank_blend,'new_run',fresh):
                search = rank_blend.search(*sources,view_weights_a=[1,0,0],view_weights_b=[1,0,0],log=lambda *a,**kw:None)
                output = Path(core.load(search/'summary.json')['output'])
                cfg = core.load(output/'config.json')
                weights = [m['weight'] for m in cfg['rank_fusion_members']]
                result = model_profile.evaluation(output,cfg['weights'])
                scores = model_profile.validation_scores(output)[0]
                def compared(run, query, candidates, *args, **kwargs):
                    i = int(query['person']); matrix = matrices[[p.resolve() for p in sources].index(Path(run).resolve())]
                    target = fresh('comparison')
                    core.save(target/'result.json',dict(failures=[],ranking=[dict(person=c['person'],score=float(matrix[i,int(c['person'])]),frontal_vistas=['CF']) for c in candidates]))
                    return target
                with patch.object(fusion,'compare_fusion',side_effect=compared):
                    for i in (0,3,8):
                        out = rank_blend.compare(output,people[i],people)
                        rows = core.load(out/'result.json')['ranking']
                        for row in rows:self.assertAlmostEqual(row['score'],scores[i,int(row['person'])])
                        small = [p for p in people if p['person'] in result['galleries'][str(i)]]
                        out = rank_blend.compare(output,people[i],small)
                        true = next(r for r in core.load(out/'result.json')['ranking'] if r['person']==str(i))
                        self.assertEqual(true['rank'],result['rows'][i]['gallery_rank'])
                with self.assertRaises(ValueError):fusion.compare_fusion(output,people[0],people,all_subsets=True)
                with self.assertRaises(ValueError):model_profile.evaluation(output,[0,1,0])

    def test_quality_is_a_proxy_not_a_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'uniform.png'
            Image.new('RGB',(400,300),'gray').save(path)
            measured = quality.measure(path)
            self.assertEqual((measured['width'],measured['height']),(400,300))
            self.assertEqual(measured['sharpness'],0)
            self.assertNotIn('confidence',measured)
            bad = Path(tmp)/'bad.png';bad.write_bytes(b'version https://git-lfs.github.com/spec/v1')
            with self.assertRaises(ValueError):quality.measure(bad)
