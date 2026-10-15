import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.technical_audit import inspect_geometry, missing_requirements


class TechnicalAuditTests(unittest.TestCase):
    def test_malformed_coordinate_does_not_abort(self):
        for value in ('invalido', [1, 2]):
            rows = inspect_geometry({'points': [{'x': value, 'y': 1}]}, {}, (20, 30), Path('.'))
            self.assertEqual(rows[0]['status'], 'ERRO')

    def test_missing_requirements(self):
        ex = {'item': {'modalidade': 'cranio'}, 'proposals': [
            {'numero': 2, 'x': None, 'y': None}, {'numero': 3, 'x': 5, 'y': 8}]}
        self.assertEqual(missing_requirements(ex, {'required': [2, 3, 4]}), [2, 4])
        face = {'points': [{'idx': 64, 'x': float('nan'), 'y': 2}]}
        self.assertEqual(missing_requirements(face, {'required': [64]}), [64])

    def test_points(self):
        rows = inspect_geometry({'points': [{'x': float('nan'), 'y': 1}]}, {}, (20, 30), Path('.'))
        self.assertEqual(rows[0]['status'], 'ERRO')
        rows = inspect_geometry({'points': [{'x': 30, 'y': 1}]}, {}, (20, 30), Path('.'))
        self.assertEqual(rows[0]['status'], 'ATENCAO')

    def test_crop_and_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cv2.imwrite(str(folder/'crop.png'), np.zeros((256, 256, 3), np.uint8))
            box = dict(crop_bounds=[0, 0, 20, 10], arquivo='crop.png',
                       transform=dict(scale=12.8, offset=[0, 64], input_size=[20, 10], size=[256, 128]))
            regions = {'regions': {'region': box}}
            self.assertEqual(inspect_geometry({}, regions, (20, 30), folder)[-1]['status'], 'OK_TECNICO')
            box['transform']['scale'] = 0
            self.assertEqual(inspect_geometry({}, regions, (20, 30), folder)[-1]['status'], 'ERRO')

    def test_absent(self):
        rows = inspect_geometry({}, {'regions': {'nose': {'reason': 'PONTOS_AUSENTES'}}}, (20, 30), Path('.'))
        self.assertEqual(rows[-1]['status'], 'ATENCAO')


if __name__ == '__main__':
    unittest.main()
