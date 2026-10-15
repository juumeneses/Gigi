import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import pandas as pd

from Laboratorio_visual_face_cranio import core, learning, processing
from Laboratorio_visual_face_cranio.data import training_samples


class LabTests(unittest.TestCase):
    def test_registered_labels_are_not_promoted_to_official(self):
        from Laboratorio_visual_face_cranio.registered import table
        curated = pd.DataFrame([dict(ano=2017, id_imagem='A_FFN', genero_final='FEMININO',
                                     origem_genero_final='FAIRFACE_ALTA_CONFIANCA', classe_fairface='FEMININO_PROVAVEL')])
        face = pd.DataFrame([dict(ano=2017, id_imagem='A_FFN', classe_modelo='FEMININO_PROVAVEL')])
        skull = pd.DataFrame([dict(ano=2017, id_pessoa='A', classe_cranio='MASCULINO_PROVAVEL'),
                              dict(ano=2019, id_pessoa='B', classe_cranio='FEMININO_PROVAVEL')])
        with patch('Laboratorio_visual_face_cranio.registered.pd.read_csv', side_effect=[curated, face, skull]), \
                patch('pathlib.Path.exists', return_value=True):
            output = table().set_index('id_pessoa')
        self.assertEqual(len(output), 2)
        self.assertTrue(output.referencia_documentada.eq('NÃO DISPONÍVEL NESTAS FONTES').all())
        self.assertEqual(output.loc['A', 'concordancia_curadoria_cranio'], 'DIVERGE')
        self.assertEqual(output.loc['B', 'concordancia_curadoria_cranio'], 'SEM COMPARAÇÃO')
        self.assertEqual(output.loc['B', 'ano'], 2019)

    def test_uniform_coordinates_roundtrip(self):
        im = np.zeros((173, 431, 3), np.uint8)
        _, matrix = core.letterbox(im, 128)
        points = [[0, 0], [100.2, 85.7], [430, 172]]
        result = core.transform(core.transform(points, matrix), cv2.invertAffineTransform(matrix))
        np.testing.assert_allclose(result, points, atol=1e-10)
        self.assertEqual(matrix[0, 0], matrix[1, 1])

    def test_missing_labels_do_not_change_loss(self):
        pred = torch.tensor([[[.2, .3], [.5, .6]]], requires_grad=True)
        target = torch.tensor([[[.1, .1], [float('nan'), float('nan')]]])
        loss = learning.masked_loss(pred, target, torch.tensor([[True, False]]))
        self.assertAlmostEqual(loss.item(), .05, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[0, 1].abs().sum().item(), 0)

    def test_missing_region_and_clipping(self):
        points = [dict(numero=i+1, x=None, y=None) for i in range(11)]
        self.assertEqual(core.regions_for(points, 100, 100)['nariz']['status'], 'REGIAO_INDISPONIVEL')
        for i in [1, 2, 3]:
            points[i].update(x=float(i), y=float(i))
        nose = core.regions_for(points, 100, 100)['nariz']
        self.assertGreater(nose['outside_fraction'], 0)
        self.assertGreaterEqual(min(nose['box']), 0)

    def references(self, root):
        samples = []
        for n in range(3):
            path = root / f'{n}.png'
            core.write_image(path, np.full((50 + n, 80, 3), 80 + n*10, np.uint8))
            samples.append(dict(key=str(n), path=str(path), hash=core.sha(path),
                                person=str(n), modality='cranio', view='CF',
                                split='validacao' if n == 2 else 'treino',
                                points=[dict(numero=i+1, x=float(10+i*2), y=float(12+i)) for i in range(11)]))
        payload = dict(samples=samples)
        core.save(root/'refs.json', dict(payload, fingerprint=core.digest(payload)))
        return root/'refs.json'

    def test_resume_matches_full_training_and_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            refs = self.references(root)
            with patch.object(core, 'ARTIFACTS', root/'out'):
                full = learning.train(refs, 'cranio', 'CF', epochs=2, size=32, batch_size=1, log=lambda _: None)
                partial = learning.train(refs, 'cranio', 'CF', epochs=1, size=32, batch_size=1, log=lambda _: None)
                learning.train(refs, 'cranio', 'CF', epochs=2, size=32, batch_size=1, resume=partial, log=lambda _: None)
                one, two = [learning.checkpoint_load(p/'last.pt') for p in (full, partial)]
                self.assertEqual(one['history'], two['history'])
                for key in one['model']:
                    self.assertTrue(torch.equal(one['model'][key], two['model'][key]))
                with self.assertRaisesRegex(ValueError, 'Retomada'):
                    learning.train(refs, 'cranio', 'CF', epochs=3, size=32, lr=.002, batch_size=1, resume=partial)
            core.write_image(root/'0.png', np.zeros((50, 80, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, 'Original alterado'):
                training_samples(refs, 'cranio', 'CF')

    def test_wrong_view_and_split_leak_are_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.references(Path(folder))
            with self.assertRaisesRegex(ValueError, 'faltam referencias'):
                training_samples(path, 'cranio', 'CE')
            data = core.load(path)
            data.pop('fingerprint')
            data['samples'][2]['person'] = '0'
            core.save(path, dict(data, fingerprint=core.digest(data)))
            with self.assertRaisesRegex(ValueError, 'mais de um split'):
                training_samples(path, 'cranio', 'CF')

    def test_processing_resumes_and_keeps_unreadable_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good, bad = root/'a.png', root/'b.png'
            core.write_image(good, np.zeros((50, 80, 3), np.uint8))
            bad.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:x')
            rows = [dict(key=str(n), path=str(p), filename=p.name) for n, p in enumerate([good, bad])]
            stop = root/'stop'
            def log(line):
                if line.startswith('[1/'):
                    stop.touch()
            partial = processing.process(rows, 'contours', run=root/'partial', stop=stop, log=log)
            self.assertEqual(core.load(partial/'status.json')['state'], 'INTERROMPIDO')
            stop.unlink()
            processing.process(rows, 'contours', run=partial, resume=True, log=lambda _: None)
            full = processing.process(rows, 'contours', run=root/'full', log=lambda _: None)
            a, b = core.load(partial/'results.json'), core.load(full/'results.json')
            self.assertEqual(len(a), 2)
            self.assertEqual([r['status'] for r in a], [r['status'] for r in b])
            self.assertEqual(a[0]['reason'], 'MASCARA_VAZIA')
            self.assertIn('LFS_NAO_BAIXADO', a[1]['reason'])
            (partial/'0'/'preview.png').write_bytes(b'corrupt')
            processing.process(rows, 'contours', run=partial, resume=True, log=lambda _: None)
            self.assertIsNotNone(cv2.imread(str(partial/'0'/'preview.png')))


if __name__ == '__main__':
    unittest.main()
