import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cranio_runs as runs
from src.common import write_image


class CheckpointTests(unittest.TestCase):
    def test_interrupted_run_preserves_completed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'dataset/splits').mkdir(parents=True)
            pd.DataFrame([{'id_pessoa': 'a', 'split': 'treino'}]).to_csv(root/'dataset/splits/split_identidades.csv', index=False)
            original = root/'original.png'
            image = np.full((40, 40, 3), 180, dtype=np.uint8)
            write_image(original, image)
            manifest = pd.DataFrame([{'id_pessoa': identity, 'ano': 2020, 'imagem_cf': str(original), 'usar_matching': True} for identity in ['a', 'b']])
            result = ({'status_cranio_geo': 'OK', 'regioes_centrais_detectadas': 4}, [], image, image)
            with patch.object(runs, 'LAB_ROOT', root), patch.object(runs, 'detect_geometry_v2', return_value=result), patch.object(runs, 'detect_geometry_v3', side_effect=[result, KeyboardInterrupt, result]) as detect:
                with self.assertRaises(KeyboardInterrupt):
                    runs.run_resumable(manifest, {'cranios': {'gate_cobertura': .8}})
                completed, _ = runs.run_resumable(manifest, {'cranios': {'gate_cobertura': .8}})
                self.assertEqual(len(completed), 2)
                self.assertEqual(detect.call_count, 3)

    def test_resume_force_and_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dataset/splits').mkdir(parents=True)
            pd.DataFrame([{'id_pessoa': 'a', 'split': 'treino'}]).to_csv(root / 'dataset/splits/split_identidades.csv', index=False)
            original = root / 'original.png'
            write_image(original, np.full((40, 40, 3), 180, dtype=np.uint8))
            lfs = root / 'lfs.jpg'
            lfs.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc')
            manifest = pd.DataFrame([{'id_pessoa': identity, 'ano': 2020, 'imagem_cf': str(path), 'usar_matching': True}
                                     for identity, path in [('a', original), ('b', root/'missing.jpg'), ('c', lfs)]])
            config = {'cranios': {'gate_cobertura': .8}}
            def detector(image, *args):
                return {'status_cranio_geo': 'OK', 'regioes_centrais_detectadas': 4}, [], image, image
            with patch.object(runs, 'LAB_ROOT', root), patch.object(runs, 'detect_geometry_v2', side_effect=detector), patch.object(runs, 'detect_geometry_v3', side_effect=detector) as detect:
                first, _ = runs.run_resumable(manifest, config)
                self.assertEqual(len(first), 3)
                self.assertEqual(first.motivo_cranio_geo.dropna().tolist(), ['IMAGEM_AUSENTE', 'GIT_LFS_NAO_BAIXADO'])
                resumed, _ = runs.run_resumable(manifest, config)
                pd.testing.assert_frame_equal(first, resumed)
                self.assertEqual(detect.call_count, 1)
                runs.run_resumable(manifest, config, force=True)
                self.assertEqual(detect.call_count, 2)
                self.assertEqual(len(list((root/'resultados/cranios/runs').iterdir())), 2)

    def test_empty_train_does_not_learn_from_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'dataset/splits').mkdir(parents=True)
            pd.DataFrame([{'id_pessoa': 'a', 'split': 'teste'}]).to_csv(root/'dataset/splits/split_identidades.csv', index=False)
            with patch.object(runs, 'LAB_ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'Split de treino vazio'):
                    runs.run_resumable(pd.DataFrame(), {})


if __name__ == '__main__':
    unittest.main()
