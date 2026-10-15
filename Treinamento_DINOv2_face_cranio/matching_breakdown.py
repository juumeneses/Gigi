"""Side-by-side view ablations, without changing the saved primary ranking."""
from .portable import Path
import numpy as np

from . import fusion
from .core import check_stop, digest, load, save, sha
from .matching_summary import query_summary

SUBSETS = ('100','010','001','110','101','011','111')
DEFAULT_WEIGHTS = (.5,.3,.2)


def label(bits):
    return ' + '.join(name for name,bit in zip(('Neutra','Sorrindo','Lateral'),bits) if bit=='1')


def sources(result):
    run = Path(result['model'])
    cfg = load(run/'config.json')
    if cfg.get('rank_fusion_members'):
        return [(f'Fonte {chr(65+i)}',Path(m['run']),m['view_weights']) for i,m in enumerate(cfg['rank_fusion_members'])]
    return [('Modelo selecionado',run,result.get('weights',cfg['weights']))]


def signature(result_path,result):
    tokens = []
    for person in [result['query']]+result['candidates']:
        for view,row in person['views'].items():
            if view not in ('FFN','FFS','FL','CF','CD'): continue
            try: token=sha(row['path'])
            except OSError: token='ARQUIVO_AUSENTE'
            tokens.append((person['person'],view,row['path'],token))
    from .improvement_pipeline import file_signatures
    model_tokens = [file_signatures(run) for _,run,_ in sources(result)]
    return digest([sha(__file__),sha(result_path),tokens,model_tokens])


def cached_matrices(run,query,candidates):
    """Only reuse validation scores when identities AND actual image hashes match."""
    cohort = {p['person']:p for p in load(run/'cohort.json')['people'] if p['split']=='validacao'}
    ids = list(cohort)
    if query['person'] not in cohort or any(c['person'] not in cohort for c in candidates): return None
    used = [(query,v) for v in ('FFN','FFS','FL')]
    used += [(c,v) for c in candidates for v in ('CF','CD')]
    for person,view in used:
        image=person['views'].get(view)
        if image is None: continue
        original=cohort[person['person']]['views'].get(view)
        try:
            if original is None or sha(image['path'])!=original['hash']: return None
        except OSError:return None
    i=ids.index(query['person']);indices=[ids.index(c['person']) for c in candidates]
    matrices=[]
    with np.load(run/'scores.npz',allow_pickle=False) as arrays:
        for name,fv,sv in zip(fusion.COMPONENTS,('FFN','FFS','FL'),('CF','CF','CD')):
            matrix=arrays[name][i:i+1,indices].copy()
            if fv not in query['views']:matrix[:]=-np.inf
            for j,c in enumerate(candidates):
                if sv not in c['views']:matrix[0,j]=-np.inf
            matrices.append(matrix)
    return matrices


def ranking(scores,candidates):
    rows=[dict(person=c['person'],score=float(score) if np.isfinite(score) else None)
          for c,score in zip(candidates,scores[0])]
    rows.sort(key=lambda r:(-r['score'] if r['score'] is not None else float('inf'),r['person']))
    for i,row in enumerate(rows,1):row['rank']=i if row['score'] is not None else None
    return rows


def build(output,allow_inference=False,stop=None,log=print):
    output=Path(output);result=load(output/'result.json')
    token=signature(output/'result.json',result)
    target=output/'model_breakdown.json'
    if target.exists() and load(target).get('signature')==token:return output
    rows=[];entries=[]
    def append(source,combination,run,weights,ranked):
        correct=next((r for r in ranked if r['person']==result['query']['person']),{})
        row=dict(Fonte=source,Combinacao=combination,Execucao=run.name,
                 Pesos=str(list(weights)),**query_summary(result,ranked),**{'Score do correto':correct.get('score')})
        rows.append(row);entries.append(dict(source=source,combination=combination,model=str(run),weights=list(weights),ranking=ranked))
    cfg=load(Path(result['model'])/'config.json')
    if cfg.get('rank_fusion_members'):
        append('Mistura RRF','Ranking principal',Path(result['model']),[m['weight'] for m in cfg['rank_fusion_members']],result['ranking'])
    for source,run,weights in sources(result):
        check_stop(stop)
        log(f'Comparacao por vistas: {source} / {run.name}',flush=True)
        config=load(run/'config.json')
        positive=[w if w>0 else fallback for w,fallback in zip(weights,DEFAULT_WEIGHTS)]
        matrices=cached_matrices(run,result['query'],result['candidates'])
        if matrices is None:
            if not allow_inference:return None
            local=fusion.compare_fusion(run,result['query'],result['candidates'],stop,log,weights=positive,all_subsets=True)
            combinations=load(local/'result.json')['combinations']
            original_bits=''.join('1' if w>0 else '0' for w in weights)
            original=combinations[original_bits]['ranking']
        else:
            combinations={}
            for bits in SUBSETS:
                selected=[w*int(bit) for w,bit in zip(positive,bits)]
                score=fusion.combine(matrices,config['normalization'],selected)
                combinations[bits]=dict(weights=selected,ranking=ranking(score,result['candidates']))
            original=ranking(fusion.combine(matrices,config['normalization'],weights),result['candidates'])
        append(source,'Configuracao da consulta',run,weights,original)
        for bits,entry in combinations.items():
            append(source,label(bits),run,entry['weights'],entry['ranking'])
    save(target,dict(signature=token,rows=rows,entries=entries,
        note='Mesma consulta e candidatos. Diagnostico separado: nao altera o ranking principal. Vistas com peso zero na consulta usam pesos diagnosticos 50/30/20 no respectivo componente.'))
    return output


def show(output,launch,busy):
    import pandas as pd
    import streamlit as st
    with st.expander('Posicao por modelo e combinacao',expanded=True):
        try:
            ready=build(output,log=lambda *a,**kw:None)
        except (OSError,ValueError,KeyError) as exc:
            st.warning('Comparacao por vistas indisponivel: '+str(exc));return
        if ready is None:
            st.info('Este historico precisa calcular as vistas separadas das imagens enviadas.')
            if st.button('Calcular tabela por modelos e combinacoes',disabled=busy()):
                launch('matching_breakdown',dict(output=str(output)))
            return
        report=load(Path(output)/'model_breakdown.json')
        frame=pd.DataFrame(report['rows'])
        columns=['Fonte','Combinacao','Posicao do correto','Primeiro colocado','Score do primeiro','Score do correto',
                 'Top-1','Top-3','Top-5','Top-10','Pessoa de referencia','Ano','Candidatos com score']
        st.dataframe(frame[[c for c in columns if c in frame]],hide_index=True)
        with st.expander('Execucoes e pesos da tabela'):
            st.dataframe(frame[['Fonte','Combinacao','Execucao','Pesos']],hide_index=True)
        st.caption('Posicao do correto: 1 e melhor; sem referencia conhecida, fica indisponivel. Top-1/3/5/10 sao desta consulta, nao Recall agregado. Scores de modelos diferentes nao sao diretamente comparaveis.')
        st.caption('Fonte A/B: treinos ou ensembles separados que entram na mistura. Configuracao da consulta preserva os pesos originais; as outras linhas testam vistas isoladas e combinacoes sem mudar o ranking principal. Pesos RRF sao entre fontes; nas demais linhas, entre neutra/sorriso/lateral.')
        st.download_button('Baixar comparacao por modelos',pd.DataFrame(report['rows']).to_csv(index=False).encode(),'modelos_combinacoes.csv')
