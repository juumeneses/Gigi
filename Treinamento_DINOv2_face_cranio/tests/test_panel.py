import unittest
import json
import ast
from pathlib import Path
from streamlit.testing.v1 import AppTest
from Treinamento_DINOv2_face_cranio.core import uses_ce


class PanelTest(unittest.TestCase):
    def test_blend_matching_metrics_parameters_and_history(self):
        root = Path(__file__).parents[1]
        blends = [p.parent for p in sorted((root/'artifacts/fusions').glob('*/config.json'),reverse=True)
                  if json.loads(p.read_text()).get('rank_fusion_members')]
        if not blends:self.skipTest('Nenhuma mistura concluida')
        app = AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
        app.radio[0].set_value('Matching').run()
        next(s for s in app.selectbox if s.label=='Execucao dos modelos').set_value(blends[0]).run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(all(any(m.label==f'Recall@{k}' for m in app.metric) for k in (1,3,5,10)))
        self.assertFalse(any(s.label=='Como comparar' for s in app.selectbox))
        self.assertTrue(any(e.label=='Parametros do treino e protocolo' for e in app.expander))
        breakdowns=[d.value for d in app.dataframe if 'Fonte' in d.value and 'Combinacao' in d.value]
        if breakdowns:
            self.assertTrue(any('Mistura RRF' in set(d['Fonte']) for d in breakdowns))
            self.assertTrue(any('Sorrindo' in set(d['Combinacao']) for d in breakdowns))
        next(s for s in app.segmented_control if s.label=='Consulta').set_value('Enviar imagens').run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        next(s for s in app.segmented_control if s.label=='Consulta').set_value('Teste conhecido').run()
        next(b for b in app.button if b.label=='Sortear amostra').click().run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any(b.label=='Comparar' for b in app.button))

    def test_improvement_pipeline_controls_and_results(self):
        root=Path(__file__).parents[1]
        app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
        app.radio[0].set_value('Treinamentos').run()
        self.assertTrue(any(b.label=='Iniciar pipeline completo' for b in app.button))
        self.assertEqual(len(next(s for s in app.multiselect if s.label=='Cenarios do pipeline').value),4)
        app.radio[0].set_value('Resultados').run()
        self.assertFalse(app.exception,[e.message for e in app.exception])
        if any(s.label=='Indicador do pipeline' for s in app.selectbox):
            next(s for s in app.selectbox if s.label=='Indicador do pipeline').set_value('Recall@5 (%)').run()
            next(s for s in app.selectbox if s.label=='Exemplo do pipeline').set_value(1).run()
            next(s for s in app.selectbox if s.label=='Previa do enquadramento').set_value(1).run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
        ensembles=[p.parent for p in (root/'artifacts/fusions').glob('*/config.json') if json.loads(p.read_text()).get('ensemble_members')]
        if ensembles:
            app.radio[0].set_value('Matching').run()
            next(s for s in app.selectbox if s.label=='Execucao dos modelos').set_value(ensembles[0]).run()
            self.assertFalse(app.exception,[e.message for e in app.exception])

    def test_selected_weight_search_is_available_in_matching(self):
        root = Path(__file__).parents[1]
        paths = sorted((root/'artifacts/weight_searches').glob('*/summary.json'), reverse=True)
        if not paths:
            self.skipTest('Nenhuma busca de pesos concluida')
        summary = json.loads(paths[0].read_text())
        model = Path(summary['outputs'][0])
        app = AppTest.from_file(str(root/'app.py'), default_timeout=60).run()
        app.radio[0].set_value('Matching').run()
        next(s for s in app.selectbox if s.label == 'Execucao dos modelos').set_value(model).run()
        for label, weight in zip(('Peso neutra','Peso sorrindo','Peso lateral'), summary['weights']):
            self.assertEqual(next(n for n in app.number_input if n.label == label).value, weight)
        next(b for b in app.button if b.label == 'Usar pesos desta execucao').click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        for label, weight in zip(('Peso neutra','Peso sorrindo','Peso lateral'), summary['weights']):
            self.assertEqual(next(n for n in app.number_input if n.label == label).value, weight)
        original = next(p.parent for p in sorted((root/'artifacts/fusions').glob('*/config.json'), reverse=True)
                        if not json.loads(p.read_text()).get('weight_search') and not json.loads(p.read_text()).get('rank_fusion_members') and not uses_ce(p.parent))
        config = json.loads((original/'config.json').read_text())
        next(s for s in app.selectbox if s.label == 'Execucao dos modelos').set_value(original).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        for label, weight in zip(('Peso neutra','Peso sorrindo','Peso lateral'), config['weights']):
            self.assertEqual(next(n for n in app.number_input if n.label == label).value, float(weight))
        app.radio[0].set_value('Resultados').run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any(t.label == 'Pesos das vistas' for t in app.tabs))

    def test_home_and_cranial_workflow_navigation(self):
        app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py'), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, 'Início')
        next(b for b in app.button if b.key == 'home_Sexo registrado').click().run()
        self.assertEqual(app.radio[0].value, 'Sexo registrado')
        self.assertTrue(any(s.label == 'Etapa do modelo craniano' for s in app.segmented_control))
        self.assertFalse(any(u.label == 'Referências de sexo (CSV)' for u in app.get('file_uploader')))
        for step in ('2 · Treinar', '3 · Aplicar e consultar', '1 · Referências'):
            next(s for s in app.segmented_control if s.label == 'Etapa do modelo craniano').set_value(step).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
        next(r for r in app.radio if r.label == 'Origem dos rótulos').set_value('Importar CSV com fonte').run()
        self.assertFalse(app.exception)
        for section in ('Resultados de faces', 'Resultados de crânios', 'Sobre os métodos', 'Modelo craniano'):
            next(s for s in app.segmented_control if s.label == 'O que você quer fazer?').set_value(section).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_start_training_never_passes_existing_directory(self):
        tree = ast.parse((Path(__file__).parents[1]/'app.py').read_text())
        page = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_page')
        start = next(n for n in ast.walk(page) if isinstance(n, ast.If)
                     and 'Iniciar treinamento selecionado' in ast.unparse(n.test))
        body = '\n'.join(ast.unparse(n) for n in start.body)
        self.assertNotIn('run_dir', body)
        self.assertNotIn('target', body)
        params = next(n for n in ast.walk(page) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'params' for t in n.targets))
        self.assertNotIn('run_dir', ast.unparse(params.value))

    def test_seven_rankings_visual_selection(self):
        root = Path(__file__).parents[1]
        records = [json.loads(p.read_text()) for p in sorted((root/'artifacts/fusion_comparisons').glob('*/result.json'), reverse=True)]
        result = next((r for r in records if r.get('all_subsets') and not uses_ce(r['model'])), None)
        if result is None:
            self.skipTest('Nenhuma comparacao real com sete combinacoes disponivel')
        app = AppTest.from_file(str(root/'app.py'), default_timeout=60).run()
        app.radio[0].set_value('Matching').run()
        next(s for s in app.selectbox if s.label == 'Execucao dos modelos').set_value(Path(result['model'])).run()
        next(s for s in app.selectbox if s.label == 'Como comparar').set_value('Todas as 7 combinacoes').run()
        for label, value in zip(('Peso neutra', 'Peso sorrindo', 'Peso lateral'), result.get('weights', [1,1,1])):
            next(n for n in app.number_input if n.label == label).set_value(float(value)).run()
        app.session_state['gallery'] = dict(run=result['model'],query=result['query'],candidates=result['candidates'],seed=42)
        app.run()
        self.assertEqual([t.label for t in app.tabs][:2], ['Ranking', 'Comparador'])
        next(c for c in app.checkbox if c.label == 'Ver historico completo deste modelo').check().run()
        self.assertFalse(app.exception)
        next(c for c in app.checkbox if c.label == 'Ver historico completo deste modelo').uncheck().run()
        for bits in ('100','010','001','110','101','011','111'):
            next(s for s in app.selectbox if s.label == 'Ranking da combinacao').set_value(bits).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertTrue(any(s.label == 'Cranio no ranking' for s in app.selectbox))
        for view in ('FFN','FFS','FL'):
            next(s for s in app.selectbox if s.label == 'Foto da pessoa').set_value(view).run()
            self.assertFalse(app.exception)
        next(c for c in app.checkbox if c.label == 'Mostrar quadrados de sensibilidade').uncheck().run()
        next(c for c in app.checkbox if c.label == 'Sobrepor face e cranio').check().run()
        self.assertFalse(app.exception)
        next(s for s in app.slider if s.label == 'Opacidade do cranio').set_value(.7).run()
        next(c for c in app.checkbox if c.label == 'Mostrar quadrados de sensibilidade').check().run()
        self.assertFalse(app.exception)
        next(c for c in app.checkbox if c.label == 'Sobrepor face e cranio').uncheck().run()
        self.assertFalse(app.exception)
        next(s for s in app.selectbox if s.label == 'Sexo provavel da consulta (informado por voce)').set_value('MASCULINO').run()
        next(r for r in app.radio if r.label == 'Referencia dos cranios').set_value('Estimativa craniana').run()
        next(c for c in app.checkbox if c.label == 'Mostrar ranking considerando o sexo informado').check().run()
        self.assertFalse(app.exception)
        next(c for c in app.checkbox if c.label == 'Mostrar ranking considerando o sexo informado').uncheck().run()
        self.assertFalse(app.exception)

    def test_existing_real_outputs_all_pages(self):
        root = Path(__file__).parents[1]
        app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        for area in ('Treinamentos', 'Resultados', 'Matching', 'Banco'):
            app.radio[0].set_value(area).run()
            self.assertFalse(app.exception, [(e.message) for e in app.exception])
        app.radio[0].set_value('Treinamentos').run()
        for backbone in ('dinov2_vitb14', 'convnext_tiny', 'dinov2_vits14'):
            next(s for s in app.selectbox if s.label == 'Extrator visual').set_value(backbone).run()
            self.assertFalse(app.exception)
        toggle = next((t for t in app.toggle if t.label == 'Rolar logs automaticamente'), None)
        if toggle is not None:
            toggle.set_value(False).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Leitura pausada' in c.value for c in app.caption))
            next(t for t in app.toggle if t.label == 'Rolar logs automaticamente').set_value(True).run()
        training = next(s for s in app.selectbox if s.label == 'O que treinar')
        self.assertEqual(len(training.options), 5)
        self.assertFalse(any(s.label == 'Experimento' for s in app.selectbox))
        for choice in ('frontal', 'sorriso', 'lateral', 'todos', 'sete'):
            next(s for s in app.selectbox if s.label == 'O que treinar').set_value(choice).run()
            self.assertFalse(app.exception)
            self.assertTrue(any(b.label == 'Iniciar treinamento selecionado' for b in app.button))
        app.radio[0].set_value('Resultados').run()
        self.assertEqual([t.label for t in app.tabs][:5], ['Visao geral', 'Resumo simples', 'Detalhes tecnicos', 'Pesos das vistas', 'Pipeline de melhorias'])
        next(r for r in app.radio if r.label == 'Galeria dos graficos').set_value('Gallery reduzida').run()
        self.assertFalse(app.exception)
        next(c for c in app.checkbox if c.label == 'Incluir testes tecnicos').uncheck().run()
        self.assertFalse(app.exception)
        for mode in ('Graficos', 'Combinacoes', 'Tabela'):
            next(s for s in app.segmented_control if s.label == 'Visualizar resultados').set_value(mode).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            if mode == 'Combinacoes' and any(s.label == 'Detalhe da combinacao' for s in app.segmented_control):
                for detail in ('Mapas', 'Parametros', 'Indicadores'):
                    next(s for s in app.segmented_control if s.label == 'Detalhe da combinacao').set_value(detail).run()
                    self.assertFalse(app.exception, [e.message for e in app.exception])
        for category in ('Acertou em primeiro', 'Nao acertou em primeiro', 'Sem ranking'):
            next(s for s in app.selectbox if s.label == 'Casos').set_value(category).run()
            self.assertFalse(app.exception)
        app.radio[0].set_value('Matching').run()
        if any(s.label == 'Tipo de modelo' for s in app.selectbox):
            plain = next(p.parent for p in sorted((root/'artifacts/fusions').glob('*/config.json'),reverse=True)
                         if not json.loads(p.read_text()).get('rank_fusion_members') and not uses_ce(p.parent))
            next(s for s in app.selectbox if s.label == 'Execucao dos modelos').set_value(plain).run()
            self.assertFalse(any(e.label == 'Matching por scores separados' for e in app.expander))
            for label in ('Neutra', 'Sorrindo', 'Lateral'):
                next(c for c in app.checkbox if c.label == label).uncheck().run()
            self.assertTrue(any('Selecione pelo menos' in i.value for i in app.info))
            next(c for c in app.checkbox if c.label == 'Sorrindo').check().run()
            next(s for s in app.segmented_control if s.label == 'Consulta').set_value('Enviar imagens').run()
            self.assertFalse(app.exception)
            next(s for s in app.segmented_control if s.label == 'Consulta').set_value('Teste conhecido').run()
        sample = next(b for b in app.button if b.label == 'Sortear amostra')
        sample.click().run()
        self.assertFalse(app.exception)
        first = app.session_state['gallery']['query']['person']
        next(b for b in app.button if b.label == 'Sortear amostra').click().run()
        self.assertNotEqual(first, app.session_state['gallery']['query']['person'])
        next(c for c in app.checkbox if c.label == 'Repetir amostra usando a seed informada').check().run()
        next(b for b in app.button if b.label == 'Sortear amostra').click().run()
        fixed = app.session_state['gallery']
        next(b for b in app.button if b.label == 'Sortear amostra').click().run()
        self.assertEqual(fixed, app.session_state['gallery'])
        self.assertTrue(any(b.label == 'Comparar' for b in app.button))
        if any(s.label == 'Vista facial' for s in app.selectbox):
            next(s for s in app.slider if s.label == 'Opacidade do cranio').set_value(.7).run()
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
