import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from Treinamento_DINOv2_face_cranio import visuals
from Treinamento_DINOv2_face_cranio.core import save


class VisualTests(unittest.TestCase):
    def test_colors_and_zero_effect(self):
        original = Image.new('RGB', (200, 200), (100, 100, 100))
        out = np.asarray(visuals.overlay(original, [[1, -1], [0, 0]]))
        self.assertGreater(out[50, 50, 0], out[50, 50, 2])
        self.assertGreater(out[50, 150, 2], out[50, 150, 0])
        np.testing.assert_array_equal(out[150, 50], [100, 100, 100])
        with self.assertRaises(ValueError):
            visuals.overlay(original, [[float('nan')]])

    def test_cache_invalidates_checkpoint_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'run'; run.mkdir()
            (run/'best.pt').write_bytes(b'weights')
            save(run/'config.json', {'size': 224})
            image = root/'image.png'; Image.new('RGB', (20,20)).save(image)
            item = dict(person='one', views={'FFN': {'path': str(image)}})
            def new_run(kind):
                path = root/kind/str(len(list((root/kind).glob('*'))))
                path.mkdir(parents=True)
                return path
            with patch.object(visuals, 'ARTIFACTS', root), patch.object(visuals, 'new_run', new_run), \
                    patch.object(visuals.engine, 'explain', return_value={'maps': []}) as explain:
                first = visuals.cached_map(run, item, item)
                self.assertEqual(first, visuals.cached_map(run, item, item))
                self.assertEqual(explain.call_count, 1)
                (run/'best.pt').write_bytes(b'new weights')
                self.assertNotEqual(first, visuals.cached_map(run, item, item))
                Image.new('RGB', (20,20), 'white').save(image)
                visuals.cached_map(run, item, item)
                self.assertEqual(explain.call_count, 3)


if __name__ == '__main__':
    unittest.main()
