"""Panel for controlled improvement experiments and visual inspection."""
from .portable import Path

import pandas as pd
import streamlit as st

from .backbone import MODELS, model_dir
from .core import ARTIFACTS, letterbox, load, read_image
from .data import cohort
from .improvement_pipeline import CONDITIONS
from .weight_search import source_groups


def controls(manifest,launch,busy):
    if not manifest:
        return
    with st.expander('Pipeline de melhorias: executar e comparar',expanded=False):
        st.caption('Novos checkpoints por Top-1 em galerias de ate 10; MRR desempata. Este painel legado compara os cenarios na galeria completa. Para selecionar tambem configuracoes em Gallery-10, use Pipeline Top-1 / 3 / 5. Teste final reservado.')
        a,b = st.columns(2)
        backbone = a.selectbox('Extrator do pipeline',list(MODELS),index=1,key='pipeline_backbone',format_func=lambda k:MODELS[k][0])
        losses = b.multiselect('Losses do pipeline',['infonce','triplet'],default=['infonce','triplet'])
        conditions = st.multiselect('Cenarios do pipeline',list(CONDITIONS),default=list(CONDITIONS),format_func=lambda k:CONDITIONS[k][0])
        st.caption('Original sempre entra como controle. Cada cenario treina neutra, sorriso e lateral nas seeds 42, 43 e 44, e compara as sete combinacoes de vistas e o ensemble das seeds.')
        a,b,c = st.columns(3)
        weights = [column.number_input(label,.05,10.,default,.05,key='pipeline_weight_'+str(i)) for i,(column,label,default) in
                   enumerate(zip((a,b,c),('Pipeline: peso neutra','Pipeline: peso sorrindo','Pipeline: peso lateral'),(.5,.3,.2)))]
        with st.expander('Parametros do pipeline'):
            a,b,c = st.columns(3)
            epochs = a.number_input('Epocas por modelo',1,1000,100,key='pipeline_epochs')
            patience = b.number_input('Paciencia do pipeline',1,100,15)
            batch = c.number_input('Batch do pipeline',2,256,32)
            a,b,c = st.columns(3)
            lr = a.number_input('Learning rate do pipeline',.000001,.1,.001,format='%.6f')
            size = b.selectbox('Resolucao do pipeline',[224,336,448])
            device = c.selectbox('Dispositivo do pipeline',['auto','cpu','mps','cuda'])
            a,b = st.columns(2)
            hard_k = a.number_input('Vizinhos dificeis por ancora',1,64,8)
            warmup = b.number_input('Epocas iniciais sem mineracao',0,100,5)
        fingerprint = cohort(manifest,'E6D','matched')['fingerprint']
        references = {}
        for loss in losses:
            groups = [g for g in source_groups() if g['backbone']==backbone and g['loss']==loss and set(g['sources']) >= {42,43,44}
                      and load(Path(g['sources'][42])/'config.json')['cohort']==fingerprint]
            if groups:
                selected = st.selectbox('Referencia existente: '+loss,range(len(groups)),format_func=lambda i,groups=groups:groups[i]['protocol'][:10]+' | seeds 42, 43, 44',key='pipeline_ref_'+loss)
                references[loss] = [groups[selected]['sources'][s] for s in (42,43,44)]
            else:
                st.caption(loss+': sem trio de referencia existente; o pipeline completo criara novos modelos.')
        n_conditions = len(set(['original',*conditions]))
        st.info(f'Pipeline completo: {n_conditions*len(losses)*9} novos modelos (3 vistas x 3 seeds por cenario/loss). O ensemble e as combinacoes reutilizam esses modelos.')
        inactive_mining = bool(set(conditions)&{'hard','hard_crop'}) and warmup >= epochs
        if inactive_mining:
            st.warning('Com esse limite de epocas, a mineracao nao sera ativada. Diminua as epocas iniciais sem mineracao ou aumente o limite.')
        args = dict(manifest=str(manifest),conditions=conditions,losses=losses,seeds=[42,43,44],weights=weights,
            reference_sources=references,backbone=backbone,epochs=epochs,patience=patience,batch=batch,lr=lr,size=size,
            device=device,hard_k=hard_k,hard_warmup=warmup)
        a,b = st.columns(2)
        if a.button('Rodar somente ensemble existente',disabled=busy() or not losses or len(references)!=len(losses),icon=':material/merge:'):
            launch('improvement_pipeline',dict(args,existing_only=True))
        prepared = (model_dir(backbone)/'model.json').exists()
        if not prepared:
            st.info('Prepare este extrator nos controles de treinamento antes de iniciar o pipeline completo.')
        if b.button('Iniciar pipeline completo',disabled=busy() or not losses or not prepared or inactive_mining,type='primary',icon=':material/play_arrow:'):
            launch('improvement_pipeline',args)
        st.caption('Acompanhe os logs acima. Interromper e Retomar preservam as etapas concluidas e verificam os arquivos antes de continuar.')


def results():
    paths = sorted((ARTIFACTS/'improvement_pipelines').glob('*/summary.json'),reverse=True)
    if not paths:
        st.info('Inicie em Treinamentos > Pipeline de melhorias.')
        return
    path = st.selectbox('Execucao do pipeline',paths,format_func=lambda p:p.parent.name)
    summary = load(path)
    status = load(path.parent/'status.json')
    st.caption(f'{status["state"]} | {status.get("done",0)}/{status.get("total",0)} etapas. '+summary['note'])
    frame = pd.DataFrame(summary['rows'])
    for k in (1,3,5,10):
        frame[f'Recall@{k} (%)'] = 100*frame[f'recall_{k}']
    frame['Ganho Top-1 (p.p.)'] = 100*frame['delta_recall_1']
    frame['IC95% do ganho (p.p.)'] = frame['ci95'].map(lambda value: f'{100*value[0]:+.2f} a {100*value[1]:+.2f}' if isinstance(value,list) else 'Sem comparacao')
    frame['Ensemble vs media individual (p.p.)'] = 100*pd.to_numeric(frame.get('delta_vs_individual_mean',pd.Series(index=frame.index,dtype=float)),errors='coerce')
    frame['Desvio Top-1 (p.p.)'] = 100*pd.to_numeric(frame['recall_1_sd'],errors='coerce')
    overview, plots, examples, subsets, crop = st.tabs(['Comparacao','Graficos','Acertos e erros','Combinacoes de vistas','Enquadramento'])
    with overview:
        st.dataframe(frame[['loss','label','kind','seeds','n','failures','Recall@1 (%)','Desvio Top-1 (p.p.)','Recall@3 (%)','Recall@5 (%)','Recall@10 (%)','mrr','Ganho Top-1 (p.p.)','IC95% do ganho (p.p.)','conclusion','Ensemble vs media individual (p.p.)']],hide_index=True)
        st.caption('Individual: media das seeds. Ensemble: uma unica previsao combinando as seeds. Ganho e intervalo comparados ao Original do mesmo tipo e loss; historico e apenas contexto.')
        st.download_button('Baixar comparacao do pipeline',(path.parent/'results.csv').read_bytes(),'pipeline_resultados.csv')
    with plots:
        metric = st.selectbox('Indicador do pipeline',['Recall@1 (%)','Recall@3 (%)','Recall@5 (%)','Recall@10 (%)','mrr'])
        plotted = frame.assign(Modelo=frame.loss+' | '+frame.label+' | '+frame.kind)
        st.bar_chart(plotted.set_index('Modelo')[[metric]],height=350)
    with examples:
        entries = summary['examples']
        if entries:
            i = st.selectbox('Exemplo do pipeline',range(len(entries)),format_func=lambda i,entries=entries:f'{entries[i]["loss"]} | {entries[i]["condition"]} | {entries[i]["kind"]} | seed {entries[i]["seed"] if entries[i]["seed"] >= 0 else "ensemble"} | {entries[i]["person"]} | posicao {entries[i]["rank"]}')
            entry = entries[i]
            people = {p['person']:p for p in load(Path(entry['output'])/'cohort.json')['people']}
            for col,pid,view,label in zip(st.columns(3),(entry['person'],entry['top_candidate'],entry['person']),('FFN','CF','CF'),('Face','Primeiro colocado','Cranio correto')):
                row = people.get(pid,{}).get('views',{}).get(view)
                if row:
                    try: col.image(letterbox(read_image(row['path']),360)[0],caption=label+' | '+str(pid),width='stretch')
                    except (ValueError,OSError) as exc: col.warning(str(exc))
            st.caption('Exemplo da seed indicada, ou do ensemble; fotos originais. Pesos e vistas podem ser conferidos no Matching selecionando esta execucao: '+Path(entry['output']).name)
    with subsets:
        data = pd.DataFrame(summary.get('subsets',[]))
        if not data.empty:
            metrics = ['recall_1','recall_3','recall_5','recall_10','mrr']
            grouped = data.groupby(['loss','condition','kind','subset'],as_index=False)[metrics].mean().sort_values('recall_1',ascending=False)
            st.caption('Bits na ordem neutra / sorriso / lateral: 100 = so neutra; 111 = todas. Resultados individuais sao medias entre seeds.')
            st.caption('Recall e MRR nesta tabela vao de 0 a 1; Recall@1 de 0,10 significa 10% em primeiro.')
            st.dataframe(grouped,hide_index=True)
    with crop:
        entries = load(path.parent/'previews.json') if (path.parent/'previews.json').exists() else []
        st.caption('Recorte heuristico de bordas uniformes, com margem de 10%. Fundo heterogeneo ou ausencia de borda clara mantem a imagem original. Nao e segmentacao anatomica.')
        if entries:
            i = st.selectbox('Previa do enquadramento',range(len(entries)),format_func=lambda i,entries=entries:entries[i]['person']+' | '+entries[i]['view'])
            entry = entries[i]
            if entry.get('preview'):
                a,b = st.columns(2)
                a.image(letterbox(read_image(entry['source']),420)[0],caption='Original',width='stretch')
                b.image(letterbox(read_image(entry['preview']),420)[0],caption='Entrada recortada',width='stretch')
                st.caption(f'{entry["reason"]} | area removida {entry["removed_fraction"]:.1%} | box original {entry["box"]}')
            else:
                st.warning(entry.get('error','Previa indisponivel'))
