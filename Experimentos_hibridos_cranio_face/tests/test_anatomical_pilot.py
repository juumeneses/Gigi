import unittest
import numpy as np
import cv2
from src.anatomical_pilot import edge_point, propose
from src.cranios import _multi_threshold_candidates


class PilotTests(unittest.TestCase):
    def test_dark_outline_does_not_hide_internal_cavities(self):
        image = np.full((300, 300, 3), 180, np.uint8)
        cv2.rectangle(image, (1, 1), (298, 298), (0, 0, 0), 4)
        for x in (90, 210):
            cv2.ellipse(image, (x, 120), (40, 40), 0, 0, 360, (0, 0, 0), -1)
        mask = np.full((300, 300), 255, np.uint8)
        box = (0, 0, 299, 299)
        legacy = _multi_threshold_candidates(image, mask, box)
        nested = _multi_threshold_candidates(image, mask, box, nested=True)
        self.assertEqual(len(legacy), 0)
        self.assertGreaterEqual(len(nested), 2)
        self.assertTrue(all(60 < c['cx'] < 240 and 90 < c['cy'] < 150 for c in nested))

    def test_border_not_center(self):
        candidate = {'contour': np.array([[[10, 20]], [[30, 20]], [[30, 40]], [[10, 40]]], dtype=np.int32)}
        np.testing.assert_equal(edge_point(candidate, 'left', True), [10, 30])
        np.testing.assert_equal(edge_point(candidate, 'right', True), [30, 30])

    def test_blank_abstains(self):
        points, mask, preview = propose(np.full((200, 200, 3), 255, np.uint8))
        self.assertEqual(len(points), 4)
        self.assertFalse(any(p['visivel'] for p in points))
        self.assertTrue(all(p['x_px'] is None and not p['anotacao_validada'] for p in points))
        self.assertEqual(preview.shape, (200, 200, 3))


if __name__ == '__main__':
    unittest.main()
