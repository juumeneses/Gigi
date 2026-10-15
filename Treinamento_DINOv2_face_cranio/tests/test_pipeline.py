import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from Treinamento_DINOv2_face_cranio import core, data, engine, backbone


class FakeExtractor:
    fingerprint = 'synthetic-only'
    device = 'cpu'

    def __init__(self, *args, **kwargs):
        torch.set_num_threads(1)

    def feature(self, row):
        if row.get('broken'):
            raise ValueError('unreadable')
        rng = np.random.default_rng(int(core.digest(row['path'])[:8], 16))
        return rng.normal(size=384).astype(np.float32)


def fixture(root, n=6):
    people = []
    for i in range(n):
        views = {v: dict(path=str(root/f'{i}_{v}.jpg'), hash='fixture') for v in core.VIEWS}
        people.append(dict(person=f'p{i}', year='2017', split='treino' if i < 3 else 'validacao', views=views, conflicts={}))
    value = dict(people=people, identity_hash_collisions={}, split_hash='fixed', rows=[], version=core.VERSION)
    value['fingerprint'] = core.digest(value)
    path = root/'manifest.json'
    core.save(path, value)
    return path, people


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_proportions_exif_and_lfs(self):
        image = Image.new('RGB', (100, 200))
        exif = image.getexif(); exif[274] = 6
        image.save(self.root/'exif.jpg', exif=exif)
        restored = core.read_image(self.root/'exif.jpg')
        self.assertEqual(restored.size, (200, 100))
        _, transform = core.letterbox(restored)
        self.assertEqual(transform['scale_x'], transform['scale_y'])
        self.assertEqual(transform['offset_y'], 56)
        core.atomic(self.root/'fake.jpg', b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError, 'LFS'): core.read_image(self.root/'fake.jpg')
        core.atomic(self.root/'bad.jpg', b'invalid')
        with self.assertRaises(OSError): core.read_image(self.root/'bad.jpg')

    def test_overlay_endpoints_and_no_input_changes(self):
        face = Image.new('RGB', (100, 200), 'red')
        skull = Image.new('RGB', (200, 100), 'blue')
        before = face.tobytes(), skull.tobytes()
        base = core.visual_overlay(face, skull, opacity=0.)
        self.assertTrue(np.array_equal(np.array(base), np.array(core.letterbox(face, 640)[0])))
        shown = core.visual_overlay(face, skull, opacity=1.)
        self.assertEqual(shown.getpixel((320, 320)), (0, 0, 255))
        self.assertEqual(shown.size, (640, 640))
        self.assertEqual(before, (face.tobytes(), skull.tobytes()))

    def test_gallery_unique_stable_and_positive_once(self):
        _, people = fixture(self.root)
        a = engine.sample_gallery(people, 4, 42)
        self.assertEqual(a, engine.sample_gallery(people, 4, 42))
        self.assertEqual(sum(x['person'] == a['query']['person'] for x in a['candidates']), 1)
        self.assertEqual(len({x['person'] for x in a['candidates']}), 4)

    def test_inventory_copies_conflicts_and_originals_untouched(self):
        bank = self.root/'Banco'
        source = bank/'2017'/'2017_1_A'/'2017_1_A_FFN.png'
        source.parent.mkdir(parents=True)
        Image.new('RGB', (20, 20), 'red').save(source)
        duplicate = bank/'2017'/'_separados'/'FFN'/source.name
        core.atomic(duplicate, source.read_bytes())
        alternative = source.with_suffix('.jpg')
        Image.new('RGB', (20, 20), 'blue').save(alternative)
        ledger = self.root/'splits.csv'
        core.atomic(ledger, b'id_pessoa,ano,split\n2017_1_A,2017,treino\n')
        before = core.sha(source)
        with patch.object(data, 'SPLITS', ledger), patch.object(data, 'new_run', lambda k: self.root/'inventory'):
            run = data.inventory(bank)
        m = core.load(run/'manifest.json')
        self.assertEqual(len(m['people']), 1)
        self.assertEqual(len(m['people'][0]['conflicts']['FFN']), 3)
        with patch.object(data, 'SPLITS', ledger), patch.object(data, 'new_run', lambda k: self.root/'resolved'):
            run = data.inventory(bank, resolutions={'2017_1_A/FFN': str(source)})
        self.assertEqual(core.load(run/'manifest.json')['people'][0]['views']['FFN']['selection'], 'explicit')
        self.assertEqual(before, core.sha(source))

    def test_missing_stays_in_evaluation_denominator(self):
        _, people = fixture(self.root, 3)
        scores = np.eye(3)
        scores[1, :] = -np.inf
        result = engine.evaluation(scores, people)
        self.assertEqual(result['full']['n'], 3)
        self.assertEqual(result['full']['failures'], 1)
        self.assertAlmostEqual(result['full']['recall_1'], 2/3)

    def test_cohorts_and_identity_quarantine(self):
        path, people = fixture(self.root)
        doc = core.load(path)
        doc['people'][0]['views'].pop('FL')
        doc['identity_hash_collisions'] = {'same': ['p1', 'p2']}
        doc['fingerprint'] = core.digest({k: v for k, v in doc.items() if k != 'fingerprint'})
        core.save(path, doc)
        matched = data.cohort(path, 'E3')
        self.assertEqual([p['person'] for p in matched['people']], ['p3', 'p4', 'p5'])
        available = data.cohort(path, 'E3', 'available')
        self.assertEqual(len(available['people']), 4)

    def test_view_mask_and_view_changes(self):
        torch.manual_seed(42)
        model = engine.Towers()
        x = torch.randn(2, 2, 384)
        mask = torch.tensor([[1., 0.], [1., 1.]])
        a = model(x, mask, ['FFN', 'FL'], 'face')
        changed = x.clone(); changed[0, 1] += 1000
        self.assertTrue(torch.equal(a[0], model(changed, mask, ['FFN', 'FL'], 'face')[0]))
        changed[1, 1] += 5
        self.assertFalse(torch.allclose(a[1], model(changed, mask, ['FFN', 'FL'], 'face')[1]))

    @patch.object(engine, 'Extractor', FakeExtractor)
    def test_resume_matches_uninterrupted_and_data_changes_blocked(self):
        path, _ = fixture(self.root)
        full, resumed = self.root/'full', self.root/'resume'
        args = dict(manifest=str(path), batch=2, patience=10, log=lambda *a, **k: None)
        engine.train(**args, epochs=3, run_dir=full)
        engine.train(**args, epochs=1, run_dir=resumed)
        engine.train(**args, epochs=3, run_dir=resumed, resume=True)
        x = torch.load(full/'last.pt', weights_only=True)
        y = torch.load(resumed/'last.pt', weights_only=True)
        self.assertEqual(x['history'], y['history'])
        for key in x['model']:
            self.assertTrue(torch.equal(x['model'][key], y['model'][key]))
        with self.assertRaisesRegex(ValueError, 'incompativel'):
            engine.train(**{**args, 'lr': .02}, epochs=3, run_dir=resumed, resume=True)

    @patch.object(engine, 'Extractor', FakeExtractor)
    def test_stop_and_resume(self):
        path, _ = fixture(self.root)
        stop = self.root/'stop'; stop.touch()
        run = self.root/'run'
        engine.train(str(path), run_dir=run, stop=stop, epochs=2)
        self.assertEqual(core.load(run/'status.json')['state'], 'INTERROMPIDO')
        stop.unlink()
        engine.train(str(path), run_dir=run, resume=True, epochs=2)
        self.assertEqual(core.load(run/'status.json')['state'], 'CONCLUIDO')

    @patch.object(engine, 'Extractor', FakeExtractor)
    def test_three_seeds_suite_has_paired_summary(self):
        path, _ = fixture(self.root)
        with patch.object(engine, 'new_run', lambda k: self.root/'suite'):
            run = engine.suite(str(path), experiment='E2', seeds=[42, 43, 44], epochs=1,
                               log=lambda *a, **k: None)
        summary = core.load(run/'summary.json')
        self.assertTrue(summary['confirmed_three_seeds'])
        self.assertEqual(summary['n_identities'], 3)
        self.assertEqual(len(core.load(run/'children.json')), 9)
        engine.suite(str(path), experiment='E2', seeds=[42, 43, 44], epochs=1,
                     resume_dir=run, log=lambda *a, **k: None)
        self.assertEqual(summary, core.load(run/'summary.json'))

    @patch.object(engine, 'Extractor', FakeExtractor)
    def test_shared_compare_independent_of_ids_and_no_landmarks(self):
        path, people = fixture(self.root)
        run = self.root/'run'
        engine.train(str(path), epochs=1, run_dir=run)
        original = engine.compare(run, people[3], people[3:])
        uploaded = [dict(p, person='upload'+str(i)) for i, p in enumerate(people[3:])]
        result = engine.compare(run, dict(people[3], person='face'), uploaded)
        self.assertEqual([r['score'] for r in original['ranking']], [r['score'] for r in result['ranking']])
        self.assertEqual(sorted(r['rank'] for r in result['ranking']), [1, 2, 3])

    @patch.object(engine, 'Extractor', FakeExtractor)
    def test_cohort_comparison_and_test_freeze_gate(self):
        path, _ = fixture(self.root)
        a, b = self.root/'a', self.root/'b'
        engine.train(str(path), experiment='E1', cohort_experiment='E2', epochs=1, run_dir=a)
        engine.train(str(path), experiment='E2', epochs=1, run_dir=b)
        self.assertEqual(len(engine.paired_comparison(a, b)['rows']), 3)
        with self.assertRaisesRegex(ValueError, 'Congele'): engine.evaluate(a, split='teste')
        config = core.load(b/'config.json'); config['seed'] = 99; core.save(b/'config.json', config)
        with self.assertRaisesRegex(ValueError, 'seed'): engine.paired_comparison(a, b)

    def test_triplet_finite_gradients(self):
        x = torch.randn(5, 128, requires_grad=True)
        y = torch.randn(5, 128, requires_grad=True)
        loss = engine.triplet(torch.nn.functional.normalize(x), torch.nn.functional.normalize(y))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_corrupt_cache_rebuilt_and_original_hash_checked(self):
        path = self.root/'image.png'; Image.new('RGB', (20, 40), 'red').save(path)
        extractor = object.__new__(backbone.Extractor)
        extractor.dimension = 384
        extractor.fingerprint = 'test'; extractor.device = 'cpu'
        calls = []
        extractor.encode_image = lambda im: (calls.append(1) or np.ones(384, np.float32), {})
        with patch.object(backbone, 'ARTIFACTS', self.root):
            row = dict(path=str(path), hash=core.sha(path))
            first = extractor.feature(row); extractor.feature(row)
            self.assertEqual(len(calls), 1)
            cache = next((self.root/'features').rglob('*.npz'))
            core.np_save(cache, embedding=np.zeros(384), key='corrupt')
            self.assertTrue(np.array_equal(first, extractor.feature(row)))
            self.assertEqual(len(calls), 2)
            Image.new('RGB', (20, 40), 'blue').save(path)
            with self.assertRaisesRegex(ValueError, 'alterada'): extractor.feature(row)


if __name__ == '__main__':
    unittest.main()
