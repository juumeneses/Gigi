import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from Treinamento_DINOv2_face_cranio.sex_controls import filter_ranking


class DocumentedFilterTest(unittest.TestCase):
    def test_experimental_reference_is_not_displayed_as_documented(self):
        from Laboratorio_visual_face_cranio.registered import table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root/'Treinamento_faces/resultados/genero_faces'
            base.mkdir(parents=True)
            pd.DataFrame([dict(ano=2019, id_imagem='2019_a_FFN', genero_final='FEMININO',
                origem_genero_final='CORRIGIDO_MANUALMENTE', classe_fairface='FEMININO_PROVAVEL')]).to_csv(base/'genero_faces_curado.csv', index=False)
            ref = root/'artifacts/sex_references/experimental'
            ref.mkdir(parents=True)
            (ref/'reference.csv').write_text('id_pessoa,sexo\n2019_a,FEMININO\n')
            (ref/'config.json').write_text(json.dumps(dict(reference_kind='CURADORIA_EXPLORATORIA')))
            with patch('Laboratorio_visual_face_cranio.registered.REPO', root), \
                 patch('Laboratorio_visual_face_cranio.registered.ARTIFACTS', root/'artifacts'):
                result = table()
            self.assertEqual(result.iloc[0].referencia_documentada, 'NÃO DISPONÍVEL NESTAS FONTES')
            self.assertNotIn('sexo_documentado', result.columns)

    def test_cached_module_without_curated_import_is_reloaded(self):
        from Laboratorio_visual_face_cranio import cranial_sex
        from Treinamento_DINOv2_face_cranio.sex_controls import cranial_module
        original = cranial_sex.import_curated_reference
        try:
            del cranial_sex.import_curated_reference
            module = cranial_module()
            self.assertTrue(callable(module.import_curated_reference))
            self.assertIs(cranial_module(), module)
        finally:
            if not hasattr(cranial_sex, 'import_curated_reference'):
                cranial_sex.import_curated_reference = original

    def test_curated_import_preserves_provenance(self):
        from Laboratorio_visual_face_cranio.cranial_sex import import_curated_reference, references
        from Laboratorio_visual_face_cranio.core import load
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'curated.csv'
            pd.DataFrame([
                dict(id_imagem='2019_a_FFN', genero_final='FEMININO', origem_genero_final='CORRIGIDO_MANUALMENTE'),
                dict(id_imagem='2019_b_FFN', genero_final='MASCULINO', origem_genero_final='FAIRFACE_ALTA_CONFIANCA'),
            ]).to_csv(source, index=False)
            with patch('Laboratorio_visual_face_cranio.cranial_sex.new_run', return_value=root/'manual'):
                run = import_curated_reference(source, 'Revisora', True)
            self.assertEqual(list(references(run).index), ['2019_a'])
            self.assertFalse(load(run/'config.json')['documented'])
            self.assertEqual(references(run).iloc[0].tipo_referencia, 'CURADORIA_EXPLORATORIA')
            with patch('Laboratorio_visual_face_cranio.cranial_sex.new_run', return_value=root/'all'):
                run = import_curated_reference(source, 'Revisora', False)
            self.assertEqual(len(references(run)), 2)
            self.assertEqual(references(run).loc['2019_b','origem_genero_final'], 'FAIRFACE_ALTA_CONFIANCA')
            with self.assertRaises(ValueError):
                import_curated_reference(source, '', True)

    def test_unknown_kept_original_unchanged(self):
        rows = [dict(person='a',rank=1,score=.9), dict(person='b',rank=2,score=.8),
                dict(person='c',rank=3,score=.7)]
        kept, excluded = filter_ranking(rows, {'a':'FEMININO','b':'MASCULINO'}, 'MASCULINO')
        self.assertEqual([r['person'] for r in kept], ['b','c'])
        self.assertEqual(excluded[0]['rank'], 1)
        self.assertNotIn('sexo_documentado', rows[0])
        self.assertEqual(kept[0]['rank'], 2)

    def test_missing_query_reference(self):
        with self.assertRaises(ValueError):
            filter_ranking([], {}, None)

    def test_estimates_not_mislabeled_as_documented(self):
        rows = [dict(person='correct',rank=1,score=.9)]
        kept, excluded = filter_ranking(rows, {'correct':'FEMININO'}, 'MASCULINO', 'sexo_estimado_cranio')
        self.assertFalse(kept)
        self.assertEqual(excluded[0]['sexo_estimado_cranio'], 'FEMININO')
        self.assertNotIn('sexo_documentado', excluded[0])
