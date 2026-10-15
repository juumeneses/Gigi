"""Validation performance and training provenance beside the matching controls."""
from .portable import Path
import streamlit as st

from .core import ARTIFACTS, load, sha, uses_ce
from .model_profile import evaluation, leaves


@st.cache_data(show_spinner=False)
def measured(run, weights, token):
    return evaluation(Path(run), weights)


def profile(run, weights=None, key='draft'):
    run = Path(run)
    if not (run/'scores.npz').exists():
        st.info('Esta execucao nao possui scores salvos para recalcular estas metricas.')
        return
    try:
        result = measured(str(run), tuple(weights) if weights is not None else None,
                          (sha(run/'scores.npz'), sha(run/'config.json')))
        metrics = result['full']
        st.markdown('**Desempenho de validacao desta selecao**')
        for col, k in zip(st.columns(4), (1, 3, 5, 10)):
            col.metric(f'Recall@{k}', f'{metrics[f"recall_{k}"]:.1%}')
        st.caption(f'{metrics["n"]} pessoas / galeria completa de validacao. Recall@K: fracao de consultas com o correto nas primeiras K posicoes. Nao e certeza para uma consulta nova.')
        with st.expander('Parametros do treino e protocolo', expanded=False):
            st.write('Execucao:', run.name)
            cfg = load(run/'config.json')
            if cfg.get('rank_fusion_members'):
                st.dataframe([dict(Fonte=i+1, Execucao=Path(m['run']).name,
                    Peso=m['weight'], Pesos_das_vistas=str(m['view_weights']))
                    for i,m in enumerate(cfg['rank_fusion_members'])], hide_index=True)
                st.caption(f'Fusao de posicoes RRF, constante {cfg["rank_constant"]}. Pesos escolhidos nesta validacao; resultado exploratorio, nao teste independente.')
            else:
                st.write('Pesos neutra / sorriso / lateral:', list(weights) if weights is not None else cfg.get('weights'))
            st.dataframe(leaves(run, weights), hide_index=True)
            st.write('MRR:', round(metrics['mrr'], 4))
            st.write('Gallery-10 fixa:', {f'Recall@{k}': f'{result["gallery"][f"recall_{k}"]:.1%}' for k in (1,3,5,10)})
    except (OSError, ValueError, KeyError) as exc:
        st.warning('Metricas desta selecao indisponiveis: '+str(exc))


def blend_controls(launch, busy):
    with st.expander('Misturar dois modelos: prioridade Top-1 e depois Top-10'):
        runs = [p.parent for p in sorted((ARTIFACTS/'fusions').glob('*/results.json'), reverse=True)
                if not load(p.parent/'config.json').get('rank_fusion_members')
                and not load(p.parent/'config.json').get('smoke') and not uses_ce(p.parent)]
        if len(runs) < 2:
            st.info('Conclua dois modelos na mesma coorte para avaliar uma mistura.'); return
        label = lambda p: p.name+' | '+str(load(p/'config.json').get('policy','modelo'))
        a = st.selectbox('Fonte A', runs, format_func=label)
        b = st.selectbox('Fonte B', runs, index=1, format_func=label)
        weights = []
        for name, run in [('A',a),('B',b)]:
            cfg = load(run/'config.json')
            weights.append([c.number_input(f'{name}: peso {v}', min_value=0., max_value=10., value=float(w), step=.05)
                for c,v,w in zip(st.columns(3),('neutra','sorriso','lateral'),cfg['weights'])])
        st.caption('Testa 105 misturas de rankings nos scores salvos. Maximiza Recall@1; Recall@10 e MRR desempatam. Nao treina o extrator nem usa o teste final.')
        if st.button('Avaliar e salvar mistura', disabled=busy() or a == b or not all(map(sum,weights))):
            launch('blend_models', dict(source_a=str(a), source_b=str(b), view_weights_a=weights[0], view_weights_b=weights[1]))


def view_controls(config, selector):
    if config.get('rank_fusion_members'):
        st.caption('Mistura RRF: vistas fixas de cada fonte, detalhadas abaixo. Para mudar as fontes ou seus pesos, use Treinamentos > Misturar dois modelos.')
        return config['weights'], False
    if st.button('Usar pesos desta execucao', icon=':material/tune:'):
        st.session_state['matching_combination_mode'] = 'Selecionar modelos'
        for name, weight in zip(('frontal','sorriso','lateral'), config['weights']):
            st.session_state['matching_use_'+name] = weight > 0
            st.session_state['match_weight_'+name] = float(weight)
        st.rerun()
    if st.button('Aplicar pesos 50% neutra / 30% sorrindo / 20% lateral', icon=':material/tune:'):
        st.session_state['matching_combination_mode'] = 'Selecionar modelos'
        for name, weight in zip(('frontal','sorriso','lateral'), (.5,.3,.2)):
            st.session_state['matching_use_'+name] = True
            st.session_state['match_weight_'+name] = weight
        st.rerun()
    mode = selector.selectbox('Como comparar', ['Selecionar modelos','Todos os modelos','Todas as 7 combinacoes'],key='matching_combination_mode')
    if mode == 'Selecionar modelos':
        active = [int(col.checkbox(label,key='matching_use_'+name)) for col,label,name in
                  zip(st.columns(3),('Neutra','Sorrindo','Lateral'),('frontal','sorriso','lateral'))]
    else:
        active = [1,1,1]
    with st.expander('Pesos da combinacao'):
        values = [col.number_input(label,min_value=0.,max_value=10.,step=.05,key='match_weight_'+name)
                  for col,label,name in zip(st.columns(3),('Peso neutra','Peso sorrindo','Peso lateral'),('frontal','sorriso','lateral'))]
        weights = [a*v for a,v in zip(active,values)]
        if sum(weights):
            st.caption('Participacao normalizada: '+', '.join(f'{n}: {w/sum(weights):.1%}' for n,w in zip(('Neutra','Sorrindo','Lateral'),weights)))
        st.caption('Pesos alteram a combinacao dos scores, nao o treino. Nao sao probabilidades de identidade. As metricas abaixo sao recalculadas para esta selecao.')
    return weights,mode == 'Todas as 7 combinacoes'
