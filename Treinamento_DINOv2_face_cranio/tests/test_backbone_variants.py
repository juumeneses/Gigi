import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Treinamento_DINOv2_face_cranio import engine, core
from Treinamento_DINOv2_face_cranio.tests.test_pipeline import FakeExtractor, fixture


class WideExtractor(FakeExtractor):
    dimension = 768
    fingerprint = 'wide-test-features'

    def feature(self, row):
        return np.tile(super().feature(row), 2)


class BackboneVariantsTest(unittest.TestCase):
    def test_wide_train_reload_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine, 'Extractor', WideExtractor):
            root = Path(tmp)
            manifest, people = fixture(root)
            run = engine.train(str(manifest), run_dir=root/'run', backbone='dinov2_vitb14',
                               epochs=1, batch=2, smoke=True)
            config, model = engine.load_run(run)
            self.assertEqual(config['dimension'], 768)
            self.assertEqual(config['backbone'], 'dinov2_vitb14')
            self.assertEqual(model.face[0].in_features, 768)
            result = engine.compare(run, people[3], people[3:])
            self.assertEqual(len(result['ranking']), 3)
            self.assertTrue(all(r['score'] is not None for r in result['ranking']))
