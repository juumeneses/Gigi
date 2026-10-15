"""Compact controls and results for validation score-weight selection."""
from .portable import Path

import pandas as pd
import streamlit as st

from .core import ARTIFACTS, load
from .backbone import MODELS
from .weight_search import source_groups


def search_controls(launch, busy):
    with st.expander('Buscar pesos das vistas na validacao', expanded=False):
        groups = [g for g in source_groups() if len(g['sources']) >= 3]
        if not groups:
            st.info('Conclua tres seeds do mesmo extrator, loss e parametros, com as sete combinacoes avaliadas.')
            return
        selected = st.selectbox('Modelos para buscar pesos', range(len(groups)),
            format_func=lambda i: MODELS[groups[i]['backbone']][0]+' | '+groups[i]['loss']+' | seeds '+
                ', '.join(map(str, sorted(groups[i]['sources'])))+' | '+groups[i]['protocol'][:8])
        group = groups[selected]
        step = st.select_slider('Passo dos pesos', options=[.05, .1, .2], value=.05,
                               format_func=lambda x: f'{x:.0%}')
        st.caption('Prioridade: maior Recall@1 medio das seeds na galeria completa de validacao. MRR desempata. Inclui pesos iguais e vistas com peso zero. Reutiliza scores salvos; nao precisa treinar novamente.')
        if st.button('Buscar melhores pesos', disabled=busy(), type='primary', icon=':material/tune:'):
            launch('search_weights', dict(sources=[group['sources'][s] for s in sorted(group['sources'])], step=step))


def search_results():
    paths = sorted((ARTIFACTS/'weight_searches').glob('*/summary.json'), reverse=True)
    if not paths:
        st.info('Em Treinamentos, abra Buscar pesos das vistas na validacao para iniciar.')
        return
    path = st.selectbox('Busca de pesos concluida', paths, format_func=lambda p: p.parent.name)
    summary = load(path)
    objective = summary.get('objective', 'mrr')
    config = load(path.parent/'config.json')
    from .reporting import metadata
    info = metadata(Path(summary['outputs'][0]))
    st.caption(f'{info["Extrator"]} | {info["Loss"]} | seeds '+', '.join(str(r['seed']) for r in summary['seeds']))
    st.caption('Criterio da busca: '+('Recall@1, com MRR como desempate' if objective == 'recall_1' else 'MRR (busca anterior)'))
    columns = st.columns(3)
    for column, label, weight in zip(columns, ('Neutra', 'Sorrindo', 'Lateral'), summary['weights']):
        column.metric('Peso '+label, f'{weight:.0%}')
    st.caption(summary['interpretation'])
    rows = []
    for label, metrics in [('Pesos iguais', summary['equal_weights']), ('Pesos escolhidos', summary['best'])]:
        rows.append(dict(Combinacao=label, MRR=metrics['mrr'], **{f'Recall@{k} (%)': 100*metrics[f'recall_{k}'] for k in (1,3,5,10)}))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.metric('Ganho de Recall@1 na validacao', f'{100*(summary["best"]["recall_1"]-summary["equal_weights"]["recall_1"]):+.2f} p.p.')
    st.caption(f'Variacao de MRR: {summary["delta_mrr"]:+.4f}')
    rows = [dict(Seed=r['seed'], MRR=r['selected']['mrr'],
            **{'MRR pesos iguais':r['baseline']['mrr'], 'Ganho MRR':r['selected']['mrr']-r['baseline']['mrr']},
            **{f'Recall@{k} (%)':100*r['selected'][f'recall_{k}'] for k in (1,3,5,10)}) for r in summary['seeds']]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander('Estabilidade dos pesos entre seeds'):
        st.caption('Em cada linha, os pesos foram escolhidos nas outras seeds. As pessoas de validacao continuam sendo as mesmas; isto verifica variacao do treinamento.')
        held_out_mrr = sum(r['mrr'] for r in summary['held_out_seed'])/len(summary['held_out_seed'])
        held_out_top1 = sum(r['recall_1'] for r in summary['held_out_seed'])/len(summary['held_out_seed'])
        st.metric('Recall@1 medio na seed excluida da escolha', f'{held_out_top1:.1%}',
                  delta=f'{100*(held_out_top1-summary["equal_weights"]["recall_1"]):+.2f} p.p. vs pesos iguais')
        st.metric('MRR medio na seed excluida da escolha', f'{held_out_mrr:.4f}',
                  delta=f'{held_out_mrr-summary["equal_weights"]["mrr"]:+.4f} vs pesos iguais')
        st.dataframe(pd.DataFrame([dict(Seed=r['seed'], Neutra=r['weights'][0], Sorrindo=r['weights'][1], Lateral=r['weights'][2],
            MRR=r['mrr'], **{'Recall@1 (%)':100*r['recall_1'], 'Recall@10 (%)':100*r['recall_10']}) for r in summary['held_out_seed']]), hide_index=True)
    with st.expander('Todas as combinacoes de pesos'):
        frame = pd.DataFrame(load(path.parent/'grid.json')).sort_values([objective, 'mrr'] if objective != 'mrr' else ['mrr'], ascending=False)
        st.dataframe(frame, hide_index=True)
        metric = st.selectbox('Metrica da busca', ['recall_1','mrr','recall_5','recall_10'])
        pivot = frame.pivot_table(index='frontal', columns='lateral', values=metric, aggfunc='max')
        st.dataframe(pivot.style.background_gradient(cmap='YlGnBu'))
    st.caption('No Matching, selecione uma execucao marcada "pesos ajustados": os pesos sao carregados automaticamente. Use "Usar pesos desta execucao" para restaura-los depois de editar.')
    st.download_button('Baixar tabela dos pesos', (path.parent/'grid.csv').read_bytes(), 'busca_pesos.csv')
