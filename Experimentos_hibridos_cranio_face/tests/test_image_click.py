import unittest
from src.image_click import map_click


class ClickTests(unittest.TestCase):
    def test_full_image(self):
        self.assertEqual(map_click(.5, .5, [0,0,2000,1000]), (1000,500))

    def test_zoom_offset_and_edges(self):
        self.assertEqual(map_click(.5,.5,[800,600,1200,1000]), (1000,800))
        self.assertEqual(map_click(1,1,[800,600,1200,1000]), (1199,999))

    def test_invalid(self):
        for u in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                map_click(u,.5,[0,0,100,100])


if __name__ == '__main__':
    unittest.main()
