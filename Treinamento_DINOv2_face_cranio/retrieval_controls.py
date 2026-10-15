"""One-click Top-10 experiment sequence and comparable validation results."""
from .portable import Path
import pandas as pd
import streamlit as st

from .backbone import MODELS, model_dir
from .core import ARTIFACTS, letterbox, load, read_image
from .data import cohort
from .reporting import started


def controls(manifest,launch,busy):
    if not manifest:return
    with st.expander('Pipeline Top-10: DINOv2 x ConvNeXt e parametros',expanded=True):
        st.caption('Objetivo: encontrar o correto entre os dez primeiros da galeria completa. Checkpoints por Recall@10; MRR desempata. Sem CE e sem usar o teste final.')
        models=st.multiselect('Extratores do pipeline Top-10',['dinov2_vitb14','convnext_tiny'],
            default=['dinov2_vitb14','convnext_tiny'],format_func=lambda k:MODELS[k][0])
        tune=st.checkbox('Depois comparar learning rate e margem',value=True,key='retrieval_tune')
        st.caption('1. Extratores com parametros iguais e tres seeds. 2. Grade no extrator mais promissor, seed 42. 3. Confirmacao da configuracao escolhida nas seeds 43/44 e ensembles. Pesos fixos: 50% neutra / 30% sorriso / 20% lateral.')
        with st.expander('Parametros do pipeline Top-10'):
            a,b,c=st.columns(3)
            epochs=a.number_input('Top-10: limite de epocas',1,1000,100)
            patience=b.number_input('Top-10: paciencia',1,100,15)
            batch=c.number_input('Top-10: batch',2,256,32)
            rates=st.multiselect('Top-10: learning rates',[.001,.0003],default=[.001,.0003],format_func=lambda v:f'{v:g}')
            margins=st.multiselect('Top-10: margens triplet',[.2,.1,.3],default=[.2,.1,.3])
            device=st.selectbox('Top-10: dispositivo',['auto','cpu','mps','cuda'])
            st.caption('Triplet; resolucao 224; extratores pre-treinados congelados e duas projecoes treinaveis. Baseline lr=0,001 / margem=0,2 sempre incluido. Seeds fixas 42, 43 e 44.')
        c=cohort(manifest,'E6D','matched')
        counts={s:sum(p['split']==s for p in c['people']) for s in ('treino','validacao','teste')}
        st.write(f'{counts["treino"]} treino | {counts["validacao"]} validacao | {counts["teste"]} teste reservado')
        st.caption('Comparacao nas mesmas identidades com FFN, FFS, FL, CF e CD disponiveis; arquivos ilegiveis continuam contando como falhas. Pessoas com vistas ausentes ou conflitos ficam fora desta coorte comparavel.')
        if c['excluded']:
            with st.expander(f'Exclusoes da coorte Top-10 ({len(c["excluded"])})'):st.dataframe(c['excluded'],hide_index=True)
        combinations=len(set([.001,*rates]))*len(set([.2,*margins])) if tune else 1
        maximum=3*(len(models)*3+(combinations-1)+(2 if tune and combinations>1 else 0)) if models else 0
        st.info(f'Ate {maximum} novos modelos de projecao por vista. Caracteristicas e etapas concluidas sao reutilizadas; cada clique inicia uma nova execucao.')
        pending=[MODELS[m][0] for m in models if not (model_dir(m)/'model.json').exists()]
        if pending:st.caption('Serao preparados automaticamente (primeira vez requer rede e espaco): '+', '.join(pending))
        if st.button('Iniciar pipeline Top-10',type='primary',icon=':material/play_arrow:',
                     disabled=busy() or not models or min(counts['treino'],counts['validacao'])<2):
            launch('retrieval_pipeline',dict(manifest=str(manifest),backbones=models,tune=tune,
                learning_rates=rates,margins=margins,seeds=[42,43,44],epochs=epochs,patience=patience,batch=batch,device=device))
        st.caption('Acompanhe em Acompanhamento e logs. Resultados > Pipeline Top-10 mostra comparacao, parametros e o modelo para usar no Matching. Nao iniciarei o treino completo por aqui.')


def use_matching(path):
    st.session_state['main_area']='Matching'
    st.session_state['matching_model_type']='Modelos por vista'
    st.session_state['fusion_model_choice']=Path(path)


def results():
    paths=sorted((ARTIFACTS/'retrieval_pipelines').glob('*/request.json'),reverse=True)
    if not paths:
        st.info('Inicie em Treinamentos > Pipeline Top-10.');return
    def label(path):
        date=started(path.parent)
        if date:
            from zoneinfo import ZoneInfo
            return date.astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y %H:%M:%S')+' | '+path.parent.name
        return path.parent.name
    path=st.selectbox('Execucao Top-10',paths,format_func=label)
    run=path.parent;state=load(run/'status.json') if (run/'status.json').exists() else {}
    st.caption(f'{state.get("state","Preparando")} | {state.get("done",0)}/{state.get("total",0)} etapas | {state.get("phase","")}')
    if not (run/'summary.json').exists():
        st.info('As metricas aparecerao apos o primeiro conjunto de modelos.');return
    report=load(run/'summary.json');frame=pd.DataFrame(report['rows'])
    if frame.empty:return
    st.caption(report['note'])
    if (run/'recommended.json').exists():
        recommended=load(run/'recommended.json');row=recommended['metrics']
        for col,k in zip(st.columns(4),(1,3,5,10)):col.metric(f'Top-10 pipeline: Recall@{k}',f'{row[f"recall_{k}"]:.1%}')
        st.caption(f'{row["n"]} candidatos por consulta | {MODELS[row["backbone"]][0]} | lr {row["lr"]:g} | margem {row["margin"]} | ensemble de tres seeds')
        st.button('Usar recomendado Top-10 no Matching',icon=':material/compare:',on_click=use_matching,args=(recommended['output'],))
    for k in (1,3,5,10):frame[f'Recall@{k} (%)']=100*frame[f'recall_{k}']
    frame['Ganho Top-10 (p.p.)']=100*pd.to_numeric(frame['delta_recall_10'],errors='coerce')
    frame['Extrator']=frame['backbone'].map(lambda k:MODELS[k][0])
    comparison,screening,plots,examples=st.tabs(['Confirmacao (3 seeds)','Triagem e parametros','Graficos Top-10','Exemplos Top-10'])
    columns=['Extrator','lr','margin','kind','n_seeds','n','Recall@1 (%)','Recall@3 (%)','Recall@5 (%)','Recall@10 (%)','mrr','Ganho Top-10 (p.p.)','conclusion']
    with comparison:
        st.dataframe(frame[frame.confirmed][columns],hide_index=True)
        st.caption('Individual = media das seeds; ensemble = ranking combinado. Ganho contra o baseline do primeiro extrator no mesmo tipo. Intervalos sao exploratorios, pois esta validacao orienta as escolhas.')
        with st.expander('Intervalos e todas as metricas'):st.dataframe(frame,hide_index=True)
    with screening:
        st.dataframe(frame[columns+['confirmed']],hide_index=True)
        st.caption('Linhas nao confirmadas representam apenas a seed 42. Nao declare ganho com base somente nelas.')
        with st.expander('Parametros e protocolo completos'):st.json(load(path),expanded=False)
        with st.expander('Sete combinacoes por modelo'):st.dataframe(pd.DataFrame(report['subsets']),hide_index=True)
    with plots:
        metric=st.selectbox('Indicador do pipeline Top-10',[f'Recall@{k} (%)' for k in (10,1,3,5)]+['mrr'])
        confirmed=frame[frame.confirmed].copy()
        confirmed['Modelo']=confirmed.Extrator+' | lr '+confirmed.lr.astype(str)+' | margem '+confirmed.margin.astype(str)+' | '+confirmed.kind
        st.bar_chart(confirmed.set_index('Modelo')[[metric]],height=320)
    with examples:
        entries=report['examples']
        if entries:
            i=st.selectbox('Consulta do pipeline Top-10',range(len(entries)),format_func=lambda i:f'{entries[i]["backbone"]} | seed {entries[i]["seed"]} | {entries[i]["person"]} | posicao {entries[i]["rank"]}')
            entry=entries[i];people={p['person']:p for p in load(Path(entry['output'])/'cohort.json')['people']}
            for col,pid,view,title in zip(st.columns(3),(entry['person'],entry['person'],entry['top_candidate']),('FFN','CF','CF'),('Face','Cranio correto','Primeiro colocado')):
                image=people.get(pid,{}).get('views',{}).get(view)
                if image:
                    try:col.image(letterbox(read_image(image['path']),320)[0],caption=title+' | '+str(pid),width='stretch')
                    except (OSError,ValueError) as exc:col.warning(str(exc))
    st.download_button('Baixar resultados Top-10',(run/'results.csv').read_bytes(),'pipeline_top10.csv')
    st.download_button('Relatorio HTML Top-10',(run/'report.html').read_bytes(),'pipeline_top10.html')
