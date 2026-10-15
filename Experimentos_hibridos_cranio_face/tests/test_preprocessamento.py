import unittest
import numpy as np
from scripts.testar_preprocessamento import variants, original_coordinates


class PreprocessingTests(unittest.TestCase):
    def test_preserves_input_and_maps_scale(self):
        image = np.full((101, 203, 3), 100, np.uint8)
        before = image.copy()
        versions = list(variants(image))
        self.assertEqual(len(versions), 5)
        np.testing.assert_array_equal(image, before)
        _, derived, transform = versions[-1]
        p = original_coordinates(dict(visivel=True, x_px=derived.shape[1]/2, y_px=derived.shape[0]/2), transform)
        self.assertAlmostEqual(p['x_px'], 101.5)
        self.assertAlmostEqual(p['y_px'], 50.5)

    def test_absence_stays_absent(self):
        point = dict(visivel=False, x_px=None, y_px=None)
        self.assertEqual(original_coordinates(point, dict(scale_x=.85, scale_y=.85)), point)


if __name__ == '__main__':
    unittest.main()
