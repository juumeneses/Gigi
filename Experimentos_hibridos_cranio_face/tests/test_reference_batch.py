import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import LAB_ROOT, write_image, sha256_file
from src.reference_batch import point_schema, select_batch, validate_draft, save_draft
from src.validation import review_sample


class ReferenceTests(unittest.TestCase):
    def draft(self):
        schema = point_schema()
        return schema, pd.DataFrame(dict(ponto=schema.ponto, estado='PENDENTE', x_px=np.nan, y_px=np.nan, observacao=''))

    def test_schema(self):
        schema = point_schema()
        self.assertEqual(len(schema), 17)
        self.assertEqual(schema.ponto.nunique(), 17)
        self.assertEqual(int(schema.conjunto.eq(9).sum()), 9)

    def test_draft_coordinates_and_absence(self):
        schema, draft = self.draft()
        validate_draft(draft, schema, 100, 100)
        draft.loc[0, 'estado'] = 'LOCALIZADO'
        with self.assertRaises(ValueError):
            validate_draft(draft, schema, 100, 100)
        draft.loc[0, ['x_px','y_px']] = [10, 20]
        validate_draft(draft, schema, 100, 100)
        draft.loc[0, 'estado'] = 'NAO_VISIVEL'
        with self.assertRaises(ValueError):
            validate_draft(draft, schema, 100, 100)

    def test_sample_excludes_acceptance_and_test(self):
        manifest = pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
        splits = pd.read_csv(LAB_ROOT/'dataset/splits/split_identidades.csv')
        batch = select_batch(manifest, splits)
        sample = review_sample(manifest, splits)
        self.assertEqual(len(batch), 30)
        self.assertTrue(batch.groupby('ano').size().eq(6).all())
        excluded = set(sample.loc[sample.grupo_revisao.eq('aceitacao'), 'id_pessoa']) | set(splits.loc[splits.split.eq('teste'), 'id_pessoa'])
        self.assertFalse(set(batch.id_pessoa) & excluded)
        self.assertEqual(batch.id_pessoa.nunique(), 30)

    def test_save_does_not_validate_and_checks_hash(self):
        schema, draft = self.draft()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/'original.png'
            write_image(path, np.zeros((100,100,3), dtype=np.uint8))
            digest = sha256_file(path)
            output = save_draft(folder, 'example', draft, schema, path, digest, 'test')
            self.assertFalse(pd.read_csv(output).validado.any())
            self.assertEqual(sha256_file(path), digest)
            with self.assertRaises(ValueError):
                save_draft(folder, 'example', draft, schema, path, 'invalid', 'test')


if __name__ == '__main__':
    unittest.main()
